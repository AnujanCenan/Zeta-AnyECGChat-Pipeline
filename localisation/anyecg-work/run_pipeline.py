"""
run_pipeline.py

Command-line entrypoint for the full "why + where" pipeline:

    record(s) -> ZETA (why: condition + top observation)
              -> anyECG-chat (where: localized span), baseline and/or enriched prompt
              -> crop-and-recheck (does the localized span actually contain
                 more of the claimed evidence than a matched control window?)
              -> ecg_visualization (overlay PNG of the ECG with the localized
                 span -- and the recheck's control window, if computed --
                 shaded on the waveform)

This script owns all repo-path / sys.path / working-directory setup, based
on command-line arguments -- the library modules (`ecg_preprocessing.py`,
`zeta_classify.py`, `localize_finding.py`, `crop_and_recheck.py`,
`ecg_visualization.py`) assume that's already been done by the time they're
imported, rather than hardcoding paths themselves. That's why the sys.path
inserts and `os.chdir` below happen BEFORE any of those modules are
imported.

Overlay generation is ON by default (one PNG per record/variant that
anyECG-chat actually returned a span for -- see --skip-overlays to turn it
off, e.g. for a fast smoke test where you only care about the JSON output).

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

Configuration via .env: copy .env.example to .env (same directory as this
script) and fill in your paths -- repo locations and checkpoint paths no
longer need to be retyped on every invocation. A value given explicitly on
the command line always overrides the corresponding .env value; .env only
supplies what you didn't type. --record/--records-file and --output are
deliberately NOT sourced from .env, since those naturally vary per run
rather than being "set up once" paths.

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
from typing import List, Optional

from dotenv import load_dotenv

# Anchored to this script's own directory, not the current working
# directory -- important because we later os.chdir() into --zeta-repo, and
# because the user may invoke this script from anywhere (e.g. a PBS job
# script's $PBS_O_WORKDIR could differ from where run_pipeline.py lives).
load_dotenv(Path(__file__).resolve().parent / ".env")


def _env_path(name: str) -> Optional[str]:
    """Reads an environment variable (from .env or the shell) as a string,
    or None if unset -- used as argparse `default=` so CLI flags still
    override whatever .env provided."""
    return os.environ.get(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ZETA (why) + anyECG-chat (where) + crop-and-recheck pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- repo locations (own sys.path / CWD setup, see module docstring) ---
    # required=False here (not argparse's required=True) because these can
    # come from .env instead -- genuinely-missing values are caught in the
    # validation step after parsing, with a clear error naming both the
    # flag and the .env variable that can supply it.
    p.add_argument("--zeta-repo", type=Path, default=_env_path("ZETA_REPO"),
                    help="Path to the ZETA repo root (contains main.py, configs/, checkpoints/). "
                         "Falls back to $ZETA_REPO from .env if not given.")
    p.add_argument("--anyecg-repo", type=Path, default=_env_path("ANYECG_REPO"),
                    help="Path to the anyECG-chat-main repo root (contains anyecg/ package). "
                         "Falls back to $ANYECG_REPO from .env if not given.")
    p.add_argument("--pipeline-code-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory containing ecg_preprocessing.py, zeta_classify.py, "
                         "localize_finding.py, crop_and_recheck.py (defaults to this script's directory).")

    # --- ZETA checkpoint / config ---
    p.add_argument("--observations-path", type=str,
                    default=_env_path("OBSERVATIONS_PATH") or "configs/observations.json",
                    help="Path to ZETA's observation bank, relative to --zeta-repo "
                         "(ZETA's own load_encoders() also expects to run with CWD == zeta-repo). "
                         "Falls back to $OBSERVATIONS_PATH from .env, then this relative default.")
    p.add_argument("--conditions", type=str, default="VPC,LBBB,RBBB,NORM",
                    help="Comma-separated ZETA condition keys to classify against. Must include "
                         "at least VPC/LBBB/RBBB (the only conditions with a confirmed anyECG-chat "
                         "phrase mapping) plus at least one more (e.g. NORM) for the pairwise "
                         "comparison to be meaningful.")

    # --- anyECG-chat checkpoints ---
    p.add_argument("--projection-ckpt", type=Path, default=_env_path("PROJECTION_CKPT"),
                    help="Path to the Stage 3 projection.pth checkpoint. "
                         "Falls back to $PROJECTION_CKPT from .env if not given.")
    p.add_argument("--ecg-model-ckpt", type=Path, default=_env_path("ECG_MODEL_CKPT"),
                    help="Path to the Stage 3 ecg_model.pth checkpoint. Strongly recommended -- "
                         "omitting it falls back to the Stage 0 CLEP-pretrained encoder only. "
                         "Falls back to $ECG_MODEL_CKPT from .env if not given.")
    p.add_argument("--lora-ckpt", type=Path, default=_env_path("LORA_CKPT"),
                    help="Path to the DIRECTORY containing adapter_config.json + "
                         "adapter_model.safetensors (typically the same stage3_ckpt/ folder as "
                         "the two checkpoints above). Strongly recommended -- omitting it leaves "
                         "the LLM at untuned base Llama-3-8B-Instruct. "
                         "Falls back to $LORA_CKPT from .env if not given.")
    p.add_argument("--ecg-encoder-ckpt", type=Path, default=_env_path("ECG_ENCODER_CKPT"),
                    help="Path to the Stage 0 CLEP-pretrained encoder (vit_base_sigmoid/model_epoch9.bin). "
                         "Only needed if you have NOT patched ECG_Language_Model.__init__'s default "
                         "to point here already; passed through as ecg_encoder_ckpt_path. "
                         "Falls back to $ECG_ENCODER_CKPT from .env if not given.")
    p.add_argument("--llm-model-id", type=str, default=_env_path("LLM_MODEL_ID"),
                    help="Local path or HF hub id for Meta-Llama-3-8B-Instruct. Only needed if you "
                         "have NOT patched ECG_Language_Model.__init__'s default already. "
                         "Falls back to $LLM_MODEL_ID from .env if not given.")

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
    p.add_argument("--skip-extra-controls", action="store_true",
                    help="Only the original random matched-length control in crop-and-recheck; skip the "
                         "beat-matched control and the max-energy crop.")
    p.add_argument("--skip-recheck", action="store_true",
                    help="Skip the crop-and-recheck validation step, running only ZETA + "
                         "anyECG-chat localization (e.g. for a quick smoke test).")

    # --- visualization ---
    p.add_argument("--skip-overlays", action="store_true",
                    help="Skip generating the ECG localization overlay PNGs (ecg_visualization.py). "
                         "On by default -- one PNG is saved per record/variant that anyECG-chat "
                         "actually returned a span for (nothing is saved for a 'Not Found' answer, "
                         "since there's no interval to shade).")
    p.add_argument("--overlay-dir", type=Path, default=None,
                    help="Directory to save overlay PNGs into. Defaults to '<output-stem>_overlays' "
                         "next to --output if not given.")
    p.add_argument("--overlay-seconds-to-plot", type=float, default=None,
                    help="How much of each record (in seconds) the overlay plots. Defaults to "
                         "--crop-seconds -- i.e. the same window anyECG-chat actually searched, so "
                         "the plotted trace matches what the model saw.")

    # --- output ---
    p.add_argument("--output", type=Path, required=True,
                    help="Path to write the JSON results file. Resolved to an absolute path "
                         "BEFORE the working directory is changed to --zeta-repo, so relative "
                         "paths here are relative to where you invoked the script, not to "
                         "--zeta-repo.")

    args = p.parse_args()

    # Values that used to be argparse's required=True, now validated here
    # instead since they can come from .env -- each error names both the
    # CLI flag and the .env variable that can supply it, since a missing
    # value could be missing from either source.
    required = {
        "--zeta-repo": ("ZETA_REPO", args.zeta_repo),
        "--anyecg-repo": ("ANYECG_REPO", args.anyecg_repo),
        "--projection-ckpt": ("PROJECTION_CKPT", args.projection_ckpt),
    }
    missing = [f"{flag} (or set {env_var} in .env)" for flag, (env_var, value) in required.items() if value is None]
    if missing:
        p.error("Missing required value(s): " + "; ".join(missing))

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

    # Same "resolve before chdir" reasoning as --output: a relative
    # --overlay-dir should land next to where the user ran the script, not
    # inside --zeta-repo.
    overlay_dir = (args.overlay_dir or output_path.parent / f"{output_path.stem}_overlays").resolve()
    if not args.skip_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # --- Own all sys.path / CWD setup here, before importing pipeline code ---
    sys.path.insert(0, str(args.pipeline_code_dir.resolve()))
    sys.path.insert(0, str(args.zeta_repo.resolve()))
    sys.path.insert(0, str(args.anyecg_repo.resolve()))

    # ZETA's own main.py loads checkpoints/config via paths relative to CWD,
    # not relative to the repo itself -- see zeta_classify.py's docstring.
    os.chdir(args.zeta_repo)

    from ecg_preprocessing import ECGPreprocessor
    from zeta_classify import ZetaZeroShotClassifier
    from localize_finding import (
        load_anyecg_model,
        localize_top_finding,
        localize_both_variants,
        CONDITION_NAME_MAP,
    )
    from crop_and_recheck import run_crop_and_recheck, summarize_recheck_results
    from ecg_visualization import plot_localization_overlay
    import matplotlib.pyplot as plt  # noqa: E402 -- Agg backend already set by ecg_visualization's import

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
        record_result = {"record_path": record_path, "findings": {}, "recheck": {}, "overlays": {}, "error": None}

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

                recheck = None
                if not args.skip_recheck:
                    recheck = run_crop_and_recheck(
                        finding, zeta_clf, preprocessor,
                        search_window_seconds=args.search_window_seconds,
                        pad_to_seconds=args.pad_to_seconds,
                        min_gap=args.min_gap,
                        rng=rng,
                        extra_controls=not args.skip_extra_controls,
                    )
                    record_result["recheck"][variant] = result_to_jsonable(recheck)
                    all_recheck_results[variant].append(recheck)
                    if recheck is not None:
                        print(f"  [{variant}] recheck: localized={recheck.localized_score:.3f} "
                              f"control={recheck.control_score} diff={recheck.score_difference}")
                        if recheck.beat_matched_score is not None or recheck.energy_score is not None:
                            print(f"  [{variant}]   beat-matched={recheck.beat_matched_score} "
                                  f"(beats {recheck.localized_n_beats} vs {recheck.beat_matched_n_beats})  "
                                  f"max-energy={recheck.energy_score} (overlap {recheck.energy_overlap})")

                # Overlay generation: skipped entirely for a "Not Found" /
                # unparseable answer (finding.where_intervals is empty/None)
                # -- there's no interval to shade, so plot_localization_overlay
                # would just produce a plain trace with no annotation. That's
                # a meaningful outcome (see crop_and_recheck's no_localization
                # bucket) but not one worth a PNG per record.
                if not args.skip_overlays and finding.where_intervals:
                    overlay_path = overlay_dir / f"{Path(record_path).name}_{finding.condition}_{variant}.png"
                    try:
                        fig = plot_localization_overlay(
                            record_path, finding,
                            preprocessor=preprocessor,
                            crop_result=recheck,
                            seconds_to_plot=args.overlay_seconds_to_plot or args.crop_seconds,
                            output_path=str(overlay_path),
                        )
                        plt.close(fig)
                        record_result["overlays"][variant] = str(overlay_path)
                        print(f"  [{variant}] overlay: {overlay_path}")
                    except Exception as e:  # noqa: BLE001 -- a plotting failure shouldn't drop an
                                             # otherwise-successful classification/localization result
                        record_result["overlays"][variant] = None
                        print(f"  [{variant}] WARNING: overlay generation failed: "
                              f"{type(e).__name__}: {e}")

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
    if not args.skip_overlays:
        print(f"Overlay PNGs (where generated): {overlay_dir}")
    if summary:
        print("Summary:")
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()