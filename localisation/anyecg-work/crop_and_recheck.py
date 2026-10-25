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

Two further controls (added after the localisation study showed ZETA's score is driven mostly by
where the waveform changes, i.e. the beats, so a localized crop containing beats can beat a random
control just by containing beats):
  - BEAT-MATCHED control: a non-overlapping window of the same length containing the same number of
    detected beats as the localized span. If the localized span only beats the random control but not
    this one, "where" and "why" agreeing mostly reflects "the span contains beats".
  - MAX-ENERGY crop: the window of the same length with the highest local signal energy (summed squared
    first difference), a model-free proposal for "where", with no non-overlap requirement. It is a rival
    localiser, not a control: if anyECG-chat's span does not beat it, the localiser adds little over
    "pick the busiest stretch". Its overlap with the localized span is reported.
Beats are found with a simple energy peak detector (a PROXY, not annotation); crops with a count that
cannot be matched are reported, not hidden.

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

import numpy as np
from scipy import stats
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; avoids importing the model code (and package-path issues) at import time
    from ecg_preprocessing import ECGPreprocessor
    from zeta_classify import ZetaZeroShotClassifier
    from localize_finding import LocalizedFinding

ZETA_FS = 500.0


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
    # --- extra controls (None when not computed or not available) ---
    n_beats_detected: Optional[int] = None          # beats found in the searched window
    localized_n_beats: Optional[int] = None
    control_n_beats: Optional[int] = None
    beat_matched_interval: Optional[Tuple[float, float]] = None
    beat_matched_n_beats: Optional[int] = None
    beat_matched_score: Optional[float] = None
    beat_matched_difference: Optional[float] = None  # localized_score - beat_matched_score
    energy_interval: Optional[Tuple[float, float]] = None
    energy_score: Optional[float] = None
    energy_difference: Optional[float] = None        # localized_score - energy_score
    energy_overlap: Optional[float] = None           # fraction of the localized span covered by the max-energy window


# ---------------------------------------------------------------------------
# Model-free helpers for the extra controls
# ---------------------------------------------------------------------------

def _energy_signal(zeta_ecg: np.ndarray) -> np.ndarray:
    """Smoothed squared first difference summed over leads; zeta_ecg is (12, T)."""
    e = (np.diff(zeta_ecg, axis=1) ** 2).sum(0)
    return np.convolve(e, np.ones(20) / 20, mode="same")


def detect_beats(zeta_ecg: np.ndarray, fs: float = ZETA_FS) -> Optional[np.ndarray]:
    """Beat times in seconds (PROXY detector: peaks of the smoothed squared first difference, at least
    0.4 s apart, above the 90th percentile). None if fewer than 3 beats are found."""
    from scipy.signal import find_peaks
    e = _energy_signal(zeta_ecg)
    pk, _ = find_peaks(e, distance=int(0.4 * fs), height=np.percentile(e, 90))
    return None if len(pk) < 3 else pk / fs


def count_beats(peaks_s: np.ndarray, interval: Tuple[float, float]) -> int:
    return int(((peaks_s >= interval[0]) & (peaks_s < interval[1])).sum())


def pick_beat_matched_window(
    interval: Tuple[float, float],
    peaks_s: np.ndarray,
    search_window_seconds: float,
    min_gap: float = 0.5,
    max_attempts: int = 1000,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[int], str]:
    """Like pick_control_window, but also requires the same number of detected beats as `interval`
    (relaxed to +-1 if an exact match is impossible, which is flagged in the returned note).
    Returns (window or None, beats in window, note)."""
    rng = rng or random
    length = interval[1] - interval[0]
    if length >= search_window_seconds:
        return None, None, "span covers the whole searched window"
    target = count_beats(peaks_s, interval)
    padded_start = max(0.0, interval[0] - min_gap)
    padded_end = min(search_window_seconds, interval[1] + min_gap)
    for tol in (0, 1):
        for _ in range(max_attempts):
            start = rng.uniform(0.0, search_window_seconds - length)
            end = start + length
            if end <= padded_start or start >= padded_end:
                n = count_beats(peaks_s, (start, end))
                if abs(n - target) <= tol:
                    return (start, end), n, ("" if tol == 0 else
                                             f"beat count matched only within +-1 ({n} vs {target})")
    return None, None, f"no non-overlapping window with {target} beat(s) found"


def pick_max_energy_window(
    zeta_ecg: np.ndarray, length_s: float, search_window_seconds: float, fs: float = ZETA_FS
) -> Optional[Tuple[float, float]]:
    """The window of `length_s` seconds with the largest summed signal energy within the searched window."""
    e = _energy_signal(zeta_ecg)[: int(round(search_window_seconds * fs))]
    w = int(round(length_s * fs))
    if w <= 0 or w >= len(e):
        return None
    c = np.concatenate([[0.0], np.cumsum(e)])
    sums = c[w:] - c[:-w]
    start = int(np.argmax(sums))
    return (start / fs, (start + w) / fs)


def _overlap_fraction(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Fraction of interval `a` covered by interval `b`."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(a[1] - a[0], 1e-9)



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
    finding: "LocalizedFinding",
    zeta_classifier: "ZetaZeroShotClassifier",
    preprocessor: "ECGPreprocessor",
    search_window_seconds: float,
    pad_to_seconds: float = 10.0,
    min_gap: float = 0.5,
    rng: Optional[random.Random] = None,
    extra_controls: bool = True,
) -> Optional[CropRecheckResult]:
    """
    Runs the crop-and-recheck consistency check for a single LocalizedFinding.

    Returns None if there is nothing to check -- i.e. anyECG-chat returned
    "Not Found" / an unparseable answer (`finding.where_intervals is None`).
    That is a meaningful, reportable outcome (ZETA claimed evidence for a
    condition anyECG-chat couldn't spatially locate at all) -- don't drop
    it silently when aggregating results; count it separately from a
    completed comparison.

    With `extra_controls=True` (default) the beat-matched control and the max-energy crop described
    in the module docstring are also scored; existing fields and behaviour are unchanged.
    """
    if not finding.where_intervals:
        return None

    interval = select_primary_interval(finding.where_intervals)
    control = pick_control_window(interval, search_window_seconds, min_gap=min_gap, rng=rng)

    def score(iv: Tuple[float, float]) -> float:
        crop = preprocessor.to_zeta_input_window(
            finding.record_path, iv[0], iv[1], pad_to_seconds=pad_to_seconds
        )
        return zeta_classifier.score_observation_at_index(crop, finding.condition, finding.why_index).score

    localized_score = score(interval)
    res = CropRecheckResult(
        record_path=finding.record_path,
        condition=finding.condition,
        observation_text=finding.why_text,
        observation_index=finding.why_index,
        localized_interval=interval,
        localized_score=localized_score,
        control_interval=None,
        control_score=None,
        score_difference=None,
    )
    notes = []
    if control is None:
        notes.append("No valid non-overlapping control window found "
                     "(localized interval likely spans most of the searched window).")
    else:
        res.control_interval = control
        res.control_score = score(control)
        res.score_difference = localized_score - res.control_score

    if extra_controls:
        try:
            sig = preprocessor.to_zeta_input(finding.record_path, crop_seconds=search_window_seconds)
            peaks = detect_beats(sig)
            if peaks is None:
                notes.append("fewer than 3 beats detected: beat-matched control skipped")
            else:
                res.n_beats_detected = int(len(peaks))
                res.localized_n_beats = count_beats(peaks, interval)
                if control is not None:
                    res.control_n_beats = count_beats(peaks, control)
                bm, n_bm, bm_note = pick_beat_matched_window(
                    interval, peaks, search_window_seconds, min_gap=min_gap, rng=rng)
                if bm is not None:
                    res.beat_matched_interval, res.beat_matched_n_beats = bm, n_bm
                    res.beat_matched_score = score(bm)
                    res.beat_matched_difference = localized_score - res.beat_matched_score
                if bm_note:
                    notes.append(bm_note)
            ew = pick_max_energy_window(sig, interval[1] - interval[0], search_window_seconds)
            if ew is not None:
                res.energy_interval = ew
                res.energy_score = score(ew)
                res.energy_difference = localized_score - res.energy_score
                res.energy_overlap = _overlap_fraction(interval, ew)
        except Exception as e:  # noqa: BLE001 -- extra controls must never lose the primary result
            notes.append(f"extra controls failed: {type(e).__name__}: {e}")
    res.note = "; ".join(notes) if notes else None
    return res


def run_crop_and_recheck_batch(
    findings: List["LocalizedFinding"],
    zeta_classifier: "ZetaZeroShotClassifier",
    preprocessor: "ECGPreprocessor",
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

    # ---- extra controls (computed on the records where each exists) ----
    def paired_block(prefix: str, pairs):
        if not pairs:
            return
        loc = [p[0] for p in pairs]
        oth = [p[1] for p in pairs]
        diff = [a - b for a, b in pairs]
        summary[f"{prefix}_n"] = len(pairs)
        summary[f"{prefix}_mean_localized"] = float(np.mean(loc))
        summary[f"{prefix}_mean_other"] = float(np.mean(oth))
        summary[f"{prefix}_mean_difference"] = float(np.mean(diff))
        summary[f"{prefix}_fraction_localized_higher"] = float(np.mean([d > 0 for d in diff]))
        if len(pairs) >= 2:
            summary[f"{prefix}_paired_ttest_pvalue"] = float(stats.ttest_rel(loc, oth).pvalue)
            if any(d != 0 for d in diff):
                summary[f"{prefix}_wilcoxon_pvalue"] = float(stats.wilcoxon(diff).pvalue)

    paired_block("vs_beat_matched", [(r.localized_score, r.beat_matched_score) for r in completed
                                     if r.beat_matched_score is not None])
    paired_block("vs_max_energy", [(r.localized_score, r.energy_score) for r in completed
                                    if r.energy_score is not None])
    nb = [(r.localized_n_beats, r.control_n_beats) for r in completed
          if r.localized_n_beats is not None and r.control_n_beats is not None]
    if nb:
        summary["mean_beats_in_localized"] = float(np.mean([a for a, _ in nb]))
        summary["mean_beats_in_random_control"] = float(np.mean([b for _, b in nb]))
    ov = [r.energy_overlap for r in completed if r.energy_overlap is not None]
    if ov:
        summary["mean_overlap_localized_with_max_energy_window"] = float(np.mean(ov))
        summary["fraction_max_energy_overlaps_localized_over_half"] = float(np.mean([o >= 0.5 for o in ov]))
    return summary


if __name__ == "__main__":
    from ecg_preprocessing import ECGPreprocessor
    from zeta_classify import ZetaZeroShotClassifier
    from localize_finding import localize_top_finding, load_anyecg_model, CONDITION_NAME_MAP

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
        print(f"Beat-matched:       {result.beat_matched_interval}  score={result.beat_matched_score}")
        print(f"Max-energy crop:    {result.energy_interval}  score={result.energy_score}")
        if result.note:
            print(f"Note: {result.note}")