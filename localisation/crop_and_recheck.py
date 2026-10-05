"""
crop_and_recheck.py

Validates whether anyECG-chat's localized span actually contains the
evidence ZETA claims justifies the condition:

  1. Crop the record to the localized span, zero-padded up to ZETA's
     expected input length, and re-score it against the SAME specific
     observation ZETA originally cited (by index, not text -- some
     conditions' observation lists contain duplicate sentences, so index
     is the only unambiguous way to re-identify "that exact claim").
  2. Crop a matched-LENGTH control window elsewhere in the same searched
     region (non-overlapping, with a buffer), and score it the same way.
  3. Compare the two scores.

This deliberately re-scores the SAME observation on both crops rather than
re-running full classification on each -- the question being tested is
"does this region support the SPECIFIC claim ZETA already made", not
"does this region independently rediscover the condition from scratch",
which would be a different (and less informative) question.

Interpreting the result:
  - localized score notably > control score: the "where" and "why" are
    mutually consistent evidence for the same finding.
  - localized score <= control score: either anyECG-chat's span doesn't
    contain the claimed feature, ZETA's whole-ECG score isn't localized to
    any particular sub-region, or ZETA's encoder degrades on short,
    out-of-distribution (zero-padded) crops -- this script reports the
    numbers; distinguishing between these explanations is your analysis,
    not something automated here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy import stats

from localisation.ecg_preprocessing import ECGPreprocessor
from localisation.zeta_classify import ZetaZeroShotClassifier
from localise_finding import LocalizedFinding


@dataclass
class CropRecheckResult:
    record_path: str
    condition: str
    observation_text: str
    observation_index: int
    localized_interval: Tuple[float, float]
    localized_score: float
    control_interval: Optional[Tuple[float, float]]
    control_score: Optional[float]
    score_difference: Optional[float]   # localized_score - control_score
    note: Optional[str] = None          # explains a skipped/partial result


def select_primary_interval(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    anyECG-chat can return multiple disjoint spans for one abnormality
    (e.g. several PVC beats in one recording). This check is scoped to ONE
    representative span per finding for now -- the longest interval,
    tie-broken by earliest start -- rather than validating every span,
    which would need a different aggregate scoring scheme. Extend this if
    per-span validation across multi-interval answers becomes important
    for your results.
    """
    return sorted(intervals, key=lambda iv: (-(iv[1] - iv[0]), iv[0]))[0]


def pick_control_window(
    interval: Tuple[float, float],
    search_window_seconds: float,
    min_gap: float = 0.5,
    max_attempts: int = 200,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[float, float]]:
    """
    Picks a window of the SAME duration as `interval`, elsewhere within
    [0, search_window_seconds], that does not overlap `interval` (padded
    by `min_gap` on each side, to avoid picking up edge-of-abnormality
    morphology right at the boundary). Uses simple rejection sampling;
    returns None if no valid window is found within `max_attempts` (this
    happens when `interval` covers most of the searched window -- a real
    condition worth logging, not a bug).

    `search_window_seconds` MUST match whatever `crop_seconds` was passed
    to anyECG-chat when the localized `interval` was produced (i.e. the
    same window it was actually searching), not the full raw record
    length -- otherwise the control window wouldn't be drawn from the same
    space anyECG-chat had access to.
    """
    rng = rng or random
    length = interval[1] - interval[0]
    if length >= search_window_seconds:
        return None  # interval already spans (or exceeds) the whole searched window

    padded_start = max(0.0, interval[0] - min_gap)
    padded_end = min(search_window_seconds, interval[1] + min_gap)

    for _ in range(max_attempts):
        start = rng.uniform(0.0, search_window_seconds - length)
        end = start + length
        if end <= padded_start or start >= padded_end:
            return (start, end)
    return None


def run_crop_and_recheck(
    finding: LocalizedFinding,
    zeta_classifier: ZetaZeroShotClassifier,
    preprocessor: ECGPreprocessor,
    search_window_seconds: float,
    pad_to_seconds: float = 10.0,
    min_gap: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Optional[CropRecheckResult]:
    """
    Runs the crop-and-recheck consistency check for a single LocalizedFinding.

    Returns None if there is nothing to check -- i.e. anyECG-chat returned
    "Not Found" / an unparseable answer (`finding.where_intervals is None`).
    That is a meaningful, reportable outcome (ZETA claimed evidence for a
    condition anyECG-chat couldn't spatially locate at all) -- don't drop
    it silently when aggregating results; count it separately from a
    completed comparison.
    """
    if not finding.where_intervals:
        return None

    interval = select_primary_interval(finding.where_intervals)
    control = pick_control_window(interval, search_window_seconds, min_gap=min_gap, rng=rng)

    localized_crop = preprocessor.to_zeta_input_window(
        finding.record_path, interval[0], interval[1], pad_to_seconds=pad_to_seconds
    )
    localized_obs = zeta_classifier.score_observation_at_index(
        localized_crop, finding.condition, finding.why_index
    )

    if control is None:
        return CropRecheckResult(
            record_path=finding.record_path,
            condition=finding.condition,
            observation_text=finding.why_text,
            observation_index=finding.why_index,
            localized_interval=interval,
            localized_score=localized_obs.score,
            control_interval=None,
            control_score=None,
            score_difference=None,
            note="No valid non-overlapping control window found "
                 "(localized interval likely spans most of the searched window).",
        )

    control_crop = preprocessor.to_zeta_input_window(
        finding.record_path, control[0], control[1], pad_to_seconds=pad_to_seconds
    )
    control_obs = zeta_classifier.score_observation_at_index(
        control_crop, finding.condition, finding.why_index
    )

    return CropRecheckResult(
        record_path=finding.record_path,
        condition=finding.condition,
        observation_text=finding.why_text,
        observation_index=finding.why_index,
        localized_interval=interval,
        localized_score=localized_obs.score,
        control_interval=control,
        control_score=control_obs.score,
        score_difference=localized_obs.score - control_obs.score,
    )


def run_crop_and_recheck_batch(
    findings: List[LocalizedFinding],
    zeta_classifier: ZetaZeroShotClassifier,
    preprocessor: ECGPreprocessor,
    search_window_seconds: float,
    **kwargs,
) -> List[Optional[CropRecheckResult]]:
    """Runs `run_crop_and_recheck` over a list of findings (e.g. your held-out test set)."""
    return [
        run_crop_and_recheck(f, zeta_classifier, preprocessor, search_window_seconds, **kwargs)
        for f in findings
    ]


def summarize_recheck_results(results: List[Optional[CropRecheckResult]]) -> dict:
    """
    Aggregate stats over a batch of crop-and-recheck results, split out by
    outcome type so nothing gets silently averaged together across
    incompatible cases:
      - `no_localization`: anyECG-chat returned no span at all.
      - `no_control`: a span was found but no valid control window existed.
      - `compared`: both scores exist -- the results a paired comparison
        (e.g. localized consistently scoring higher) would be based on.
    """
    no_localization = sum(1 for r in results if r is None)
    completed = [r for r in results if r is not None]
    no_control = [r for r in completed if r.control_score is None]
    compared = [r for r in completed if r.control_score is not None]

    summary = {
        "n_total": len(results),
        "n_no_localization": no_localization,
        "n_no_control_window": len(no_control),
        "n_compared": len(compared),
    }

    if compared:
        localized_scores = [r.localized_score for r in compared]
        control_scores = [r.control_score for r in compared]
        differences = [r.score_difference for r in compared]

        summary["mean_localized_score"] = sum(localized_scores) / len(localized_scores)
        summary["mean_control_score"] = sum(control_scores) / len(control_scores)
        summary["mean_difference"] = sum(differences) / len(differences)
        summary["fraction_localized_higher"] = sum(1 for d in differences if d > 0) / len(differences)

        # Paired test: are localized scores systematically higher than their
        # matched controls, across the whole batch? Reported alongside the
        # raw means/fractions above, not as a substitute for looking at them.
        if len(compared) >= 2:
            t_stat, p_value = stats.ttest_rel(localized_scores, control_scores)
            summary["paired_ttest_statistic"] = float(t_stat)
            summary["paired_ttest_pvalue"] = float(p_value)

    return summary


if __name__ == "__main__":
    from localise_finding import localize_top_finding, load_anyecg_model, CONDITION_NAME_MAP

    preprocessor = ECGPreprocessor()
    zeta_clf = ZetaZeroShotClassifier(conditions=list(CONDITION_NAME_MAP.keys()) + ["NORM"])
    anyecg_model = load_anyecg_model(
        projection_ckpt="path/to/projection.pth",
        ecg_model_ckpt="path/to/ecg_model.pth",
        lora_ckpt="path/to/lora_adapter",
    )

    crop_seconds = 10.0
    finding = localize_top_finding(
        record_path="path/to/some/record",
        zeta_classifier=zeta_clf,
        anyecg_model=anyecg_model,
        preprocessor=preprocessor,
        crop_seconds=crop_seconds,
    )

    result = run_crop_and_recheck(
        finding, zeta_clf, preprocessor, search_window_seconds=crop_seconds
    )

    if result is None:
        print("anyECG-chat found no span for this finding -- nothing to recheck.")
    else:
        print(f"Observation:        \"{result.observation_text}\"")
        print(f"Localized interval: {result.localized_interval}  score={result.localized_score:.3f}")
        print(f"Control interval:   {result.control_interval}  score={result.control_score}")
        print(f"Difference:         {result.score_difference}")
        if result.note:
            print(f"Note: {result.note}")
