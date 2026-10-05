"""
run_pipeline.py

Command-line entrypoint for the full "why + where" pipeline:

    record(s) -> ZETA (why: condition + top observation)
              -> anyECG-chat (where: localized span), baseline and/or enriched prompt
              -> crop-and-recheck (does the localized span actually contain
                 more of the claimed evidence than a matched control window?)

This script owns all repo-path / sys.path / working-directory setup, based
on command-line arguments -- the library modules (`ecg_preprocessing.py`,
`zeta_classify.py`, `localize_finding.py`, `crop_and_recheck.py`) assume
that's already been done by the time they're imported, rather than
hardcoding paths themselves. That's why the sys.path inserts and `os.chdir`
below happen BEFORE any of those modules are imported.

Usage (single record):
    python run_pipeline.py \\
        --zeta-repo /path/to/Zeta \\
        --anyecg-repo /path/to/anyECG-chat-main \\
        --projection-ckpt /path/to/stage3_ckpt/projection.pth \\
        --ecg-model-ckpt /path/to/stage3_ckpt/ecg_model.pth \\
        --lora-ckpt /path/to/stage3_ckpt \\
        --ecg-encoder-ckpt /path/to/vit_base_sigmoid/model_epoch9.bin \\
        --llm-model-id /path/to/Meta-Llama-3-8B-Instruct \\
        --record /path/to/some/record \\
        --output results.json

Usage (batch, one WFDB record path per line in a text file):
    ... same flags ...
        --records-file records.txt \\
        --output results.json

See --help for every option, or the argument definitions below for what
each one is doing and why it exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ZETA (why) + anyECG-chat (where) + crop-and-recheck pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- repo locations (own sys.path / CWD setup, see module docstring) ---
    p.add_argument("--zeta-repo", required=True, type=Path,
                    help="Path to the ZETA repo root (contains main.py, configs/, checkpoints/).")
    p.add_argument("--anyecg-repo", required=True, type=Path,
                    help="Path to the anyECG-chat-main repo root (contains anyecg/ package).")
    p.add_argument("--pipeline-code-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory containing ecg_preprocessing.py, zeta_classify.py, "
                         "localize_finding.py, crop_and_recheck.py (defaults to this script's directory).")

    # --- ZETA checkpoint / config ---
    p.add_argument("--observations-path", type=str, default="configs/observations.json",
                    help="Path to ZETA's observation bank, relative to --zeta-repo "
                         "(ZETA's own load_encoders() also expects to run with CWD == zeta-repo).")
    p.add_argument("--conditions", type=str, default="VPC,LBBB,RBBB,NORM",
                    help="Comma-separated ZETA condition keys to classify against. Must include "
                         "at least VPC/LBBB/RBBB (the only conditions with a confirmed anyECG-chat "
                         "phrase mapping) plus at least one more (e.g. NORM) for the pairwise "
                         "comparison to be meaningful.")

    # --- anyECG-chat checkpoints ---
    p.add_argument("--projection-ckpt", required=True, type=Path,
                    help="Path to the Stage 3 projection.pth checkpoint.")
    p.add_argument("--ecg-model-ckpt", type=Path, default=None,
                    help="Path to the Stage 3 ecg_model.pth checkpoint. Strongly recommended -- "
                         "omitting it falls back to the Stage 0 CLEP-pretrained encoder only.")
    p.add_argument("--lora-ckpt", type=Path, default=None,
                    help="Path to the DIRECTORY containing adapter_config.json + "
                         "adapter_model.safetensors (typically the same stage3_ckpt/ folder as "
                         "the two checkpoints above). Strongly recommended -- omitting it leaves "
                         "the LLM at untuned base Llama-3-8B-Instruct.")
    p.add_argument("--ecg-encoder-ckpt", type=Path, default=None,
                    help="Path to the Stage 0 CLEP-pretrained encoder (vit_base_sigmoid/model_epoch9.bin). "
                         "Only needed if you have NOT patched ECG_Language_Model.__init__'s default "
                         "to point here already; passed through as ecg_encoder_ckpt_path.")
    p.add_argument("--llm-model-id", type=str, default=None,
                    help="Local path or HF hub id for Meta-Llama-3-8B-Instruct. Only needed if you "
                         "have NOT patched ECG_Language_Model.__init__'s default already.")

    # --- pipeline behavior ---
    p.add_argument("--record", type=str, default=None,
                    help="A single WFDB record path (no file extension) to run the pipeline on.")
    p.add_argument("--records-file", type=Path, default=None,
                    help="A text file with one WFDB record path per line, for batch mode. "
                         "Mutually exclusive with --record.")
    p.add_argument("--prompt-variant", choices=["baseline", "enriched", "both"], default="both",
                    help="Which anyECG-chat localization prompt(s) to run.")
    p.add_argument("--crop-seconds", type=float, default=10.0,
                    help="How much of each record (in seconds) anyECG-chat searches for the "
                         "localized span. Also used as the default --search-window-seconds.")
    p.add_argument("--search-window-seconds", type=float, default=None,
                    help="Window (in seconds) crop-and-recheck's control window is sampled "
                         "from. MUST match the window anyECG-chat actually searched -- "
                         "defaults to --crop-seconds if not given; only override this if you "
                         "have a specific reason the two should differ.")
    p.add_argument("--pad-to-seconds", type=float, default=10.0,
                    help="Length (in seconds) cropped windows are zero-padded to before "
                         "re-scoring with ZETA (matches ZETA's fixed expected input length).")
    p.add_argument("--min-gap", type=float, default=0.5,
                    help="Buffer (in seconds) kept clear around the localized interval when "
                         "sampling a control window. A single fixed value across all conditions "
                         "is a simplification -- e.g. PVC's compensatory pause arguably needs a "
                         "larger buffer than LBBB/RBBB do. Revisit before treating results as final.")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for control-window sampling, for reproducible runs.")
    p.add_argument("--skip-recheck", action="store_true",
                    help="Skip the crop-and-recheck validation step, running only ZETA + "
                         "anyECG-chat localization (e.g. for a quick smoke test).")

    # --- output ---
    p.add_argument("--output", type=Path, required=True,
                    help="Path to write the JSON results file. Resolved to an absolute path "
                         "BEFORE the working directory is changed to --zeta-repo, so relative "
                         "paths here are relative to where you invoked the script, not to "
                         "--zeta-repo.")

    args = p.parse_args()

    if bool(args.record) == bool(args.records_file):
        p.error("Provide exactly one of --record or --records-file.")

    if args.search_window_seconds is None:
        args.search_window_seconds = args.crop_seconds

    return args


def result_to_jsonable(obj):
    """Recursively converts dataclasses (and things containing them) into JSON-safe structures."""
    if dataclasses.is_dataclass(obj):
        return {k: result_to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: result_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [result_to_jsonable(v) for v in obj]
    return obj


def load_record_list(args: argparse.Namespace) -> List[str]:
    if args.record:
        return [args.record]
    with open(args.records_file) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    args = parse_args()

    # Resolve the output path to absolute BEFORE chdir-ing into the ZETA
    # repo, and before any relative record paths are used, so a relative
    # --output doesn't silently land inside --zeta-repo instead of where
    # the user actually ran the script from.
    output_path = args.output.resolve()
    records = load_record_list(args)

    # --- Own all sys.path / CWD setup here, before importing pipeline code ---
    sys.path.insert(0, str(args.pipeline_code_dir.resolve()))
    sys.path.insert(0, str(args.zeta_repo.resolve()))
    sys.path.insert(0, str(args.anyecg_repo.resolve()))

    # ZETA's own main.py loads checkpoints/config via paths relative to CWD,
    # not relative to the repo itself -- see zeta_classify.py's docstring.
    os.chdir(args.zeta_repo)

    from ecg_preprocessing import ECGPreprocessor
    from zeta_classify import ZetaZeroShotClassifier
    from localise_finding import (
        load_anyecg_model,
        localize_top_finding,
        localize_both_variants,
        CONDITION_NAME_MAP,
    )
    from crop_and_recheck import run_crop_and_recheck, summarize_recheck_results

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unmapped_required = [c for c in CONDITION_NAME_MAP if c not in conditions]
    if unmapped_required:
        print(f"NOTE: --conditions does not include {unmapped_required}, which have confirmed "
              f"anyECG-chat phrase mappings -- ZETA simply won't be able to pick those as its "
              f"top finding for any record.")

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    print("Loading ZETA classifier...")
    zeta_clf = ZetaZeroShotClassifier(observations_path=args.observations_path, conditions=conditions)

    print("Loading anyECG-chat model (this loads all Stage 3 checkpoints -- see warnings below "
          "if any are missing)...")
    anyecg_model = load_anyecg_model(
        projection_ckpt=str(args.projection_ckpt),
        ecg_model_ckpt=str(args.ecg_model_ckpt) if args.ecg_model_ckpt else None,
        lora_ckpt=str(args.lora_ckpt) if args.lora_ckpt else None,
        ecg_encoder_ckpt_path=str(args.ecg_encoder_ckpt) if args.ecg_encoder_ckpt else None,
        llm_model_id=args.llm_model_id,
    )

    preprocessor = ECGPreprocessor()

    all_results = []
    all_recheck_results = {"baseline": [], "enriched": []}
    variants = ["baseline", "enriched"] if args.prompt_variant == "both" else [args.prompt_variant]

    for i, record_path in enumerate(records):
        print(f"\n[{i + 1}/{len(records)}] {record_path}")
        record_result = {"record_path": record_path, "findings": {}, "recheck": {}, "error": None}

        try:
            if args.prompt_variant == "both":
                findings = localize_both_variants(
                    record_path, zeta_clf, anyecg_model, preprocessor, crop_seconds=args.crop_seconds
                )
            else:
                findings = {
                    args.prompt_variant: localize_top_finding(
                        record_path, zeta_clf, anyecg_model, preprocessor,
                        crop_seconds=args.crop_seconds, prompt_variant=args.prompt_variant,
                    )
                }

            for variant, finding in findings.items():
                record_result["findings"][variant] = result_to_jsonable(finding)
                print(f"  [{variant}] condition={finding.condition}  "
                      f"where={finding.where_raw!r}")

                if not args.skip_recheck:
                    recheck = run_crop_and_recheck(
                        finding, zeta_clf, preprocessor,
                        search_window_seconds=args.search_window_seconds,
                        pad_to_seconds=args.pad_to_seconds,
                        min_gap=args.min_gap,
                        rng=rng,
                    )
                    record_result["recheck"][variant] = result_to_jsonable(recheck)
                    all_recheck_results[variant].append(recheck)
                    if recheck is not None:
                        print(f"  [{variant}] recheck: localized={recheck.localized_score:.3f} "
                              f"control={recheck.control_score} diff={recheck.score_difference}")

        except Exception as e:  # noqa: BLE001 -- deliberately broad: one bad record shouldn't kill a batch run
            record_result["error"] = f"{type(e).__name__}: {e}"
            print(f"  ERROR: {record_result['error']}")

        all_results.append(record_result)

    summary = {}
    if not args.skip_recheck:
        for variant in variants:
            summary[variant] = summarize_recheck_results(all_recheck_results[variant])

    output = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "results": all_results,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nWrote results for {len(records)} record(s) to {output_path}")
    if summary:
        print("Summary:")
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
