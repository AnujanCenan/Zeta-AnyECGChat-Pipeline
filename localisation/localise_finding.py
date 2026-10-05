"""
localize_finding.py

Chains ZETA's zero-shot "why" (condition + top clinical observation) into
an anyECG-chat localization request for the "where", from the SAME source
ECG record. Supports two prompt variants so you can test whether telling
anyECG-chat WHICH feature to look for (on top of the condition name it was
actually trained with) helps, hurts, or does nothing to localization
quality:

    "baseline": condition name only -- matches anyECG-chat's actual
                training distribution (see module note below).
    "enriched": condition name + ZETA's top observation text appended --
                untested territory; not something anyECG-chat was trained
                to use, but nothing stops you from sending it.

    ZETA:        record -> to_zeta_input   -> top_finding()   -> condition, observation text
    anyECG-chat: record -> to_anyecg_input -> ecg_chat(...)   -> "Duration: Xs-Ys" (or "Not Found")

The mapping from ZETA's condition keys (e.g. "VPC", "LBBB", "RBBB") to the
human-readable abnormality phrase anyECG-chat's localization prompts expect
is NOT guessed -- it's pulled directly from the actual `question` field of
anyECG-chat's own localization test data (data/location_evaluation/
mit_bih_arrhythmia_test.json), e.g.:
    abnormal_type "V" -> question uses "Premature ventricular contraction"
    abnormal_type "L" -> question uses "Left bundle branch block beat"
    abnormal_type "R" -> question uses "Right bundle branch block beat"
Only the three conditions with a confirmed real overlap are mapped below.
Extend CONDITION_NAME_MAP only after checking the actual phrasing anyECG-chat
was trained on for any new condition -- guessing a phrase that doesn't match
its training distribution risks silently degrading localization quality.

Assumes the caller has already inserted both repo roots onto sys.path
before importing this module (see `run_pipeline.py`, which does this based
on command-line arguments rather than hardcoded paths), that the process's
working directory is ZETA's repo root (its own `main.py` loads
`checkpoints/best.pt` and `configs/*.json` via hardcoded relative paths,
not paths relative to the repo itself), that you have the three released
anyECG-chat Stage 3 checkpoint files (projection.pth, ecg_model.pth, and
the adapter_config.json/adapter_model.safetensors pair, typically all in
one `stage3_ckpt/` folder) on disk, AND that anyECG-chat's
`ECG_Language_Model.__init__` has been patched to accept
`ecg_encoder_ckpt_path`/`llm_model_id` as parameters instead of the
hardcoded, author-machine-specific absolute paths it ships with -- see
`load_anyecg_model` below for the small (4-line) patch this requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch

from anyecg.ecg_language_modeling import ECG_Language_Model  # noqa: E402
from anyecg.utils import parse_intervals  # noqa: E402

from ecg_preprocessing import ECGPreprocessor  # noqa: E402
from zeta_classify import ZetaZeroShotClassifier, ConditionEvidence  # noqa: E402


def load_anyecg_model(
    projection_ckpt: str,
    ecg_model_ckpt: Optional[str] = None,
    lora_ckpt: Optional[str] = None,
    use_lora: bool = True,
    ecg_encoder_ckpt_path: Optional[str] = None,
    llm_model_id: Optional[str] = None,
) -> ECG_Language_Model:
    """
    Constructs ECG_Language_Model AND loads the fine-tuned Stage 3
    checkpoint weights into it -- mirroring anyECG-chat's own
    `inference.py` exactly, because `ECG_Language_Model.__init__` on its
    own does NOT do this: it only loads the CLEP-pretrained ECG encoder
    and initializes `projection` with fresh random weights and (if
    `use_lora=True`) a near-identity, untrained LoRA adapter. Skipping
    this loading step produces a model that runs without error but whose
    localization output is meaningless -- not a fine-tuned model with
    slightly-off behavior, just an untrained projection layer.

    `projection_ckpt` is required (matches `inference.py`'s own assertion
    that it must be provided). `ecg_model_ckpt` and `lora_ckpt` are
    optional there too, but you almost certainly want to provide all
    three released Stage 3 checkpoint files together -- partial loading
    (e.g. projection only) mixes fine-tuned and untrained components in a
    way the original training never produced. `lora_ckpt` should be the
    DIRECTORY containing `adapter_config.json` + `adapter_model.safetensors`
    (that pair together IS the LoRA adapter, in PEFT's own format) --
    e.g. the released `stage3_ckpt/` folder, which conveniently also
    holds `projection.pth` and `ecg_model.pth` alongside it.

    `ecg_encoder_ckpt_path` (the Stage 0 CLEP-pretrained encoder, e.g.
    `vit_base_sigmoid/model_epoch9.bin`) and `llm_model_id` (a local path
    or HF hub id for Meta-Llama-3-8B-Instruct) require anyECG-chat's own
    `ECG_Language_Model.__init__` to be patched first -- the released code
    hardcodes these as absolute paths specific to the original authors'
    machine (`/mnt/sda1/xxxx/...`, `/home/xxxx/...`), which will not exist
    in your environment and will raise FileNotFoundError otherwise. The
    patch just adds these two as constructor parameters with the original
    hardcoded strings as defaults, so it doesn't change behavior for
    anyECG-chat's own `train.py`/`inference.py`, which don't pass them.
    If left as None here, anyECG-chat's (patched) defaults apply --
    only pass these if you're not replicating the original directory
    layout on your own machine/cluster.
    """
    kwargs = {"use_lora": use_lora}
    if ecg_encoder_ckpt_path is not None:
        kwargs["ecg_encoder_ckpt_path"] = ecg_encoder_ckpt_path
    if llm_model_id is not None:
        kwargs["llm_model_id"] = llm_model_id

    model = ECG_Language_Model(**kwargs).cuda()

    res = model.projection.load_state_dict(torch.load(projection_ckpt))
    print(f"Loaded projection checkpoint: {res}")

    if ecg_model_ckpt is not None:
        res = model.ecg_model.load_state_dict(torch.load(ecg_model_ckpt))
        print(f"Loaded ecg_model checkpoint: {res}")
    else:
        print("WARNING: no ecg_model_ckpt provided -- using the CLEP-pretrained "
              "encoder only, not the Stage 3 fine-tuned one. Localization quality "
              "will likely be degraded.")

    if lora_ckpt is not None:
        model.language_model.load_adapter(lora_ckpt)
        print(f"Loaded LoRA adapter: {lora_ckpt}")
    else:
        print("WARNING: no lora_ckpt provided -- the LLM's attention layers are "
              "untouched from base Llama-3-8B-Instruct, not the Stage 3 fine-tuned "
              "adapter. Localization quality will likely be degraded.")

    model.eval()
    return model


# Ground-truth-confirmed mapping only -- see module docstring.
CONDITION_NAME_MAP = {
    "VPC": "Premature ventricular contraction",
    "LBBB": "Left bundle branch block beat",
    "RBBB": "Right bundle branch block beat",
}

PromptVariant = Literal["baseline", "enriched"]


@dataclass
class LocalizedFinding:
    """The full result of one record going through both models."""
    record_path: str
    condition: str                       # ZETA's condition key, e.g. "VPC"
    abnormal_name: str                    # phrase actually sent to anyECG-chat
    why_score: float                      # ZETA's aggregate possibility score
    why_text: str                         # ZETA's top-scoring positive observation
    why_index: int                        # that observation's index (see zeta_classify.ObservationScore)
    prompt_variant: PromptVariant         # which prompt was actually sent
    question: str                         # the exact question text sent to anyECG-chat
    where_raw: str                        # anyECG-chat's raw generated answer
    where_intervals: Optional[List[Tuple[float, float]]]  # parsed spans, or None if "Not Found" / unparseable


def build_localization_question(abnormal_name: str, variant: PromptVariant, why_text: Optional[str] = None) -> str:
    """
    Builds the question text for the requested prompt variant.

    "baseline" reproduces anyECG-chat's own training template exactly
    (condition name only). "enriched" appends ZETA's cited feature as an
    extra instruction -- a template anyECG-chat was never trained on, so
    treat any behavioral difference you observe as a genuine experimental
    result, not an expected capability.
    """
    if variant == "baseline":
        return f"Locate the {abnormal_name} on this ECG for me, please."
    elif variant == "enriched":
        if not why_text:
            raise ValueError("enriched variant requires `why_text`.")
        return (
            f"Locate the {abnormal_name} on this ECG for me, please. "
            f"Look specifically for the following feature: {why_text}."
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant!r}. Expected 'baseline' or 'enriched'.")


def get_zeta_finding(
    record_path: str,
    zeta_classifier: ZetaZeroShotClassifier,
    preprocessor: ECGPreprocessor,
) -> Tuple[ConditionEvidence, str]:
    """
    Runs just the ZETA half: returns (finding, abnormal_name), raising if
    the top condition has no confirmed anyECG-chat phrase mapping. Split
    out from `localize_top_finding` so `localize_both_variants` can compute
    this once and reuse it for both prompt variants, rather than paying
    for ZETA's forward pass twice.
    """
    zeta_ecg = preprocessor.to_zeta_input(record_path)
    finding = zeta_classifier.top_finding(zeta_ecg)

    if finding.condition not in CONDITION_NAME_MAP:
        raise KeyError(
            f"No confirmed anyECG-chat phrase mapping for condition '{finding.condition}'. "
            f"Currently mapped conditions: {list(CONDITION_NAME_MAP)}. "
            f"Restrict `zeta_classifier`'s `conditions` argument to these, or verify and add "
            f"a new mapping entry against anyECG-chat's actual localization question phrasing "
            f"before extending this."
        )
    return finding, CONDITION_NAME_MAP[finding.condition]


def run_anyecg_localization(
    anyecg_ecg,
    abnormal_name: str,
    anyecg_model: ECG_Language_Model,
    variant: PromptVariant = "baseline",
    why_text: Optional[str] = None,
) -> Tuple[str, str, Optional[List[Tuple[float, float]]]]:
    """
    Runs just the anyECG-chat half against an already-preprocessed ECG
    array. Returns (question, raw_answer, parsed_intervals_or_None).
    """
    question = build_localization_question(abnormal_name, variant, why_text)
    messages = [[{"role": "user", "content": question}]]
    ecgs = [torch.tensor(anyecg_ecg, dtype=torch.float32)]

    raw_answer = anyecg_model.ecg_chat(ecgs, messages)[0]

    try:
        intervals = parse_intervals(raw_answer)
    except (ValueError, IndexError):
        # Matches anyECG-chat's own `compute_iou` behavior: an unparseable
        # or "Not Found"-style answer is treated as no localized span,
        # not an error.
        intervals = None

    return question, raw_answer, intervals


def localize_top_finding(
    record_path: str,
    zeta_classifier: ZetaZeroShotClassifier,
    anyecg_model: ECG_Language_Model,
    preprocessor: ECGPreprocessor,
    crop_seconds: float = 10.0,
    prompt_variant: PromptVariant = "baseline",
) -> LocalizedFinding:
    """
    Runs the full "why then where" chain on a single record, using ONE
    prompt variant. For running both variants on the same finding (e.g.
    for an IoU ablation), use `localize_both_variants` instead -- it's
    cheaper, since it computes ZETA's finding only once.

    crop_seconds controls how much of the record anyECG-chat searches --
    it supports dynamic-length input (see `encode_and_project_ecg` in
    anyECG-chat's own code, which chunks anything longer than 10s), so this
    can be set well beyond 10s if you want to search a longer window than
    the single 10s clip ZETA scored.
    """
    finding, abnormal_name = get_zeta_finding(record_path, zeta_classifier, preprocessor)
    anyecg_ecg = preprocessor.to_anyecg_input(record_path, crop_seconds=crop_seconds)

    question, raw_answer, intervals = run_anyecg_localization(
        anyecg_ecg, abnormal_name, anyecg_model, prompt_variant, finding.top_observation.text
    )

    return LocalizedFinding(
        record_path=record_path,
        condition=finding.condition,
        abnormal_name=abnormal_name,
        why_score=finding.score,
        why_text=finding.top_observation.text,
        why_index=finding.top_observation.index,
        prompt_variant=prompt_variant,
        question=question,
        where_raw=raw_answer,
        where_intervals=intervals,
    )


def localize_both_variants(
    record_path: str,
    zeta_classifier: ZetaZeroShotClassifier,
    anyecg_model: ECG_Language_Model,
    preprocessor: ECGPreprocessor,
    crop_seconds: float = 10.0,
) -> Dict[PromptVariant, LocalizedFinding]:
    """
    Runs both "baseline" and "enriched" prompts against the SAME ZETA
    finding and the SAME preprocessed anyECG-chat input, computed once and
    reused -- this is the entry point for the baseline-vs-enriched IoU
    ablation on anyECG-chat's held-out localization test set.
    """
    finding, abnormal_name = get_zeta_finding(record_path, zeta_classifier, preprocessor)
    anyecg_ecg = preprocessor.to_anyecg_input(record_path, crop_seconds=crop_seconds)

    results: Dict[PromptVariant, LocalizedFinding] = {}
    for variant in ("baseline", "enriched"):
        question, raw_answer, intervals = run_anyecg_localization(
            anyecg_ecg, abnormal_name, anyecg_model, variant, finding.top_observation.text
        )
        results[variant] = LocalizedFinding(
            record_path=record_path,
            condition=finding.condition,
            abnormal_name=abnormal_name,
            why_score=finding.score,
            why_text=finding.top_observation.text,
            why_index=finding.top_observation.index,
            prompt_variant=variant,
            question=question,
            where_raw=raw_answer,
            where_intervals=intervals,
        )
    return results


if __name__ == "__main__":
    preprocessor = ECGPreprocessor()
    zeta_clf = ZetaZeroShotClassifier(conditions=list(CONDITION_NAME_MAP.keys()) + ["NORM"])
    anyecg_model = load_anyecg_model(
        projection_ckpt="stage3_ckpt/projection.pth",
        ecg_model_ckpt="stage3_ckpt/ecg_model.pth",
        lora_ckpt="stage3_ckpt",  # folder containing adapter_config.json + adapter_model.safetensors
        ecg_encoder_ckpt_path="vit_base_sigmoid/model_epoch9.bin",  # requires the patch, see docstring above
        llm_model_id="path/to/local/Meta-Llama-3-8B-Instruct",       # or an HF hub id
    )

    both = localize_both_variants(
        record_path="path/to/some/record",
        zeta_classifier=zeta_clf,
        anyecg_model=anyecg_model,
        preprocessor=preprocessor,
        crop_seconds=10.0,
    )

    for variant, result in both.items():
        print(f"\n--- {variant} ---")
        print(f"Question:             {result.question}")
        print(f"Why (ZETA):           \"{result.why_text}\" (score {result.why_score:.3f})")
        print(f"Where (anyECG-chat):  {result.where_raw}")
        print(f"Parsed intervals:     {result.where_intervals}")