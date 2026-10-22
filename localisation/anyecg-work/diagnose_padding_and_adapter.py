"""
diagnose_padding_and_adapter.py

Two independent checks for the "every RBBB record gets classified LBBB, and
anyECG-chat's localized span is always ~the whole window" pattern observed
on 2-lead MIT-BIH records (e.g. record 118).

Run this the same way you'd run run_pipeline.py -- from a shell with the
.env configured (or pass the repo paths as args below), after `cd`-ing to
wherever your pipeline code lives:

    python diagnose_padding_and_adapter.py --record /path/to/mitdb/118 \
        --zeta-repo ... --anyecg-repo ... --pipeline-code-dir ... \
        --projection-ckpt ... --ecg-model-ckpt ... --lora-ckpt ...

CHECK 1 (no GPU/model needed): confirms whether ZETA's zero-padded leads
really do come out as a nonzero plateau, by comparing to_zeta_input()'s
per-lead min/max against to_anyecg_input()'s. This alone is enough to
confirm or rule out Hypothesis 1 without running any model.

CHECK 2 (needs anyECG-chat loaded): confirms whether the loaded LoRA
adapter is actually changing the model's output at all, by generating the
SAME localization question with the adapter enabled vs. disabled (peft's
`disable_adapter()` context manager) and diffing the two answers. If
they're identical, the adapter isn't doing anything -- consistent with
Hypothesis 2. Skipped automatically if the model can't be loaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--record", required=True)
    p.add_argument("--zeta-repo", type=Path, default=os.environ.get("ZETA_REPO"))
    p.add_argument("--anyecg-repo", type=Path, default=os.environ.get("ANYECG_REPO"))
    p.add_argument("--pipeline-code-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--projection-ckpt", type=Path, default=os.environ.get("PROJECTION_CKPT"))
    p.add_argument("--ecg-model-ckpt", type=Path, default=os.environ.get("ECG_MODEL_CKPT"))
    p.add_argument("--lora-ckpt", type=Path, default=os.environ.get("LORA_CKPT"))
    p.add_argument("--ecg-encoder-ckpt", type=Path, default=os.environ.get("ECG_ENCODER_CKPT"))
    p.add_argument("--llm-model-id", type=str, default=os.environ.get("LLM_MODEL_ID"))
    p.add_argument("--abnormal-name", type=str, default="Right bundle branch block beat",
                    help="Condition name to ask anyECG-chat to locate, for CHECK 2.")
    p.add_argument("--skip-model-check", action="store_true",
                    help="Only run CHECK 1 (no GPU / model loading needed).")
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, str(args.pipeline_code_dir.resolve()))
    if args.zeta_repo:
        sys.path.insert(0, str(args.zeta_repo.resolve()))
    if args.anyecg_repo:
        sys.path.insert(0, str(args.anyecg_repo.resolve()))
    if args.zeta_repo:
        os.chdir(args.zeta_repo)

    from ecg_preprocessing import ECGPreprocessor, STANDARD_LEAD_ORDER

    pre = ECGPreprocessor()

    print("=" * 70)
    print("CHECK 1: does ZETA's zero-padding become a nonzero plateau?")
    print("=" * 70)

    rec = pre.load_for_display(args.record)  # raw, pre-normalization, (T, 12)
    zeta_input = pre.to_zeta_input(args.record)        # (12, T), ZETA's normalized view
    anyecg_input = pre.to_anyecg_input(args.record)    # (12, T), anyECG-chat's normalized view

    raw_lead_is_zero = np.all(rec.signal == 0.0, axis=0)  # which raw leads are all-zero (padded)

    print(f"{'lead':>5} | {'padded?':>8} | {'raw min/max':>18} | "
          f"{'ZETA normed value':>20} | {'anyECG normed value':>22}")
    for i, lead in enumerate(STANDARD_LEAD_ORDER):
        padded = bool(raw_lead_is_zero[i])
        raw_range = f"{rec.signal[:, i].min():.3f}/{rec.signal[:, i].max():.3f}"
        zeta_vals = zeta_input[i]      # (12, T) -- lead i across time
        anyecg_vals = anyecg_input[i]
        zeta_str = f"{zeta_vals.min():.3f} .. {zeta_vals.max():.3f}"
        anyecg_str = f"{anyecg_vals.min():.3f} .. {anyecg_vals.max():.3f}"
        flag = "  <-- PADDED" if padded else ""
        print(f"{lead:>5} | {str(padded):>8} | {raw_range:>18} | {zeta_str:>20} | {anyecg_str:>22}{flag}")

    padded_idx = np.where(raw_lead_is_zero)[0]
    if len(padded_idx):
        zeta_padded_vals = zeta_input[padded_idx].flatten()
        anyecg_padded_vals = anyecg_input[padded_idx].flatten()
        print()
        print(f"Padded leads under ZETA's normalization: "
              f"mean={zeta_padded_vals.mean():.4f}, std={zeta_padded_vals.std():.6f} "
              f"(expect ~0 if this is NOT a bug; a nonzero mean confirms Hypothesis 1)")
        print(f"Padded leads under anyECG-chat's normalization: "
              f"mean={anyecg_padded_vals.mean():.4f}, std={anyecg_padded_vals.std():.6f} "
              f"(expect exactly 0 -- nan_to_num fallback)")
    else:
        print("\nNo padded leads detected for this record -- CHECK 1 doesn't apply "
              "(try a known 2-lead MIT-BIH record like 118).")

    if args.skip_model_check:
        return

    print()
    print("=" * 70)
    print("CHECK 2: is the LoRA adapter actually changing anyECG-chat's output?")
    print("=" * 70)

    if not (args.projection_ckpt and args.anyecg_repo):
        print("Skipping -- need --projection-ckpt and --anyecg-repo to load the model.")
        return

    import torch
    from localize_finding import load_anyecg_model, build_localization_question

    model = load_anyecg_model(
        projection_ckpt=str(args.projection_ckpt),
        ecg_model_ckpt=str(args.ecg_model_ckpt) if args.ecg_model_ckpt else None,
        lora_ckpt=str(args.lora_ckpt) if args.lora_ckpt else None,
        ecg_encoder_ckpt_path=str(args.ecg_encoder_ckpt) if args.ecg_encoder_ckpt else None,
        llm_model_id=args.llm_model_id,
    )

    question = build_localization_question(args.abnormal_name, "baseline")
    messages = [[{"role": "user", "content": question}]]
    ecgs = [torch.tensor(anyecg_input, dtype=torch.float32)]

    # --- Which adapter(s) actually exist, and which is active? ---
    # anyECG-chat's own ecg_language_modeling.py calls `model.add_adapter(peft_config)`
    # directly on the AutoModelForCausalLM instance -- transformers' native PeftAdapterMixin,
    # NOT peft.get_peft_model(). This means __init__(use_lora=True) ALREADY creates an
    # untrained adapter (conventionally named "default") before load_anyecg_model() ever
    # runs load_adapter(lora_ckpt). If that call loads the real Stage 3 weights under a
    # DIFFERENT adapter name (e.g. your "Cool_Adapter" workaround) without also calling
    # set_adapter(...) to make it active, generation keeps using the untrained "default"
    # adapter -- no error anywhere, just silently wrong output. This block checks exactly that.
    lm = model.language_model
    try:
        all_adapters = list(lm.peft_config.keys())
    except AttributeError:
        all_adapters = None
    try:
        active = lm.active_adapters()
    except AttributeError:
        active = None

    print(f"All loaded adapter names: {all_adapters}")
    print(f"Currently ACTIVE adapter(s) for generation: {active}")
    if all_adapters and active and set(active) != set(all_adapters) - {None}:
        print("*** More than one adapter is loaded, and the active one may not be the one "
              "load_adapter() just loaded. If the active adapter isn't your Stage 3 checkpoint's "
              "name, call model.language_model.set_adapter('<your adapter name>') after loading. ***")

    answer_with_adapter = model.ecg_chat(ecgs, messages)[0]
    print(f"\nAnswer WITH current active adapter: {answer_with_adapter!r}")

    # transformers' PeftAdapterMixin exposes disable_adapters()/enable_adapters() as plain
    # toggle methods (NOT a context manager, and NOT the same API as peft.PeftModel's
    # disable_adapter() singular context manager -- that's the method name that raised
    # AttributeError here, since anyECG-chat's language_model is not a peft.PeftModel).
    try:
        lm.disable_adapters()
        answer_without_adapter = model.ecg_chat(ecgs, messages)[0]
        lm.enable_adapters()
        print(f"Answer with ALL adapters disabled:  {answer_without_adapter!r}")

        if answer_with_adapter.strip() == answer_without_adapter.strip():
            print("\n*** IDENTICAL -- whatever adapter is active isn't changing the output at all. ***")
            print("Combined with the active-adapter check above, this points to either no adapter "
                  "really being active, or the active one being the untrained 'default' from "
                  "__init__ rather than your loaded Stage 3 weights.")
        else:
            print("\nAnswers differ -- SOME adapter is having an effect. Check the active-adapter "
                  "name printed above against the adapter name your Stage 3 checkpoint was loaded "
                  "under -- if they don't match, you may be running the wrong (untrained) adapter.")
    except AttributeError as e:
        print(f"model.language_model doesn't expose disable_adapters()/enable_adapters(): {e}")
        print("Check anyECG-chat's installed transformers version -- this is part of "
              "PeftAdapterMixin and should be present whenever add_adapter()/load_adapter() are.")


if __name__ == "__main__":
    main()