"""
ecg_visualization.py

Renders a human-readable visualization of a WFDB ECG record with
anyECG-chat's localized region(s) highlighted, optionally alongside
crop-and-recheck's control window -- the "visual element" half of the
thesis's goal (textual reasoning + an accompanying ECG visual), paired with
`LocalizedFinding`/`CropRecheckResult`'s text output.

Design notes / scope
---------------------
anyECG-chat's localization output is a DISCRETE time span (or a few
disjoint spans) parsed from a "Duration: Xs-Ys" string -- not a per-sample
or per-pixel importance score, and not produced by any attention/saliency
mechanism (this is exactly the capability ZETA's cross-attention attempted
and that the project write-up found too unreliable to use -- see
`design_document.md`). There is no finer-grained signal to render a
literal, continuously-varying attention heatmap from, and fabricating one
here would overstate what the model actually produced. What this module
renders instead is:

  - the localized interval(s), shaded directly on the waveform, across all
    12 leads individually (ECG abnormalities are frequently visible on
    specific leads only, so leads are shown as small multiples rather than
    collapsed to one trace or merged into a single composite signal);
  - a soft alpha taper at each interval's edges (full opacity inside the
    interval, fading to ~0 over a short, configurable window just outside
    it) -- this is a RENDERING CHOICE to make the shaded band read as a
    "highlighted region" rather than a harsh on/off box, not a claim that
    anyECG-chat expressed graded confidence within or around the interval;
  - if a `CropRecheckResult` is supplied, its control window shaded in a
    visibly different style (grey, hatched) so the validation check this
    pipeline actually performs is legible alongside the finding it tests.

Functions/output here deliberately say "overlay" or "localization", never
"heatmap" or "saliency map", since those terms imply a continuous,
model-derived importance signal this pipeline does not have. If per-sample
importance ever becomes available from some other method layered on top
(e.g. a gradient- or attention-based explainer applied to ZETA or
anyECG-chat), extend `_interval_alpha` into a real per-sample weight array
and feed it to a `pcolormesh`/`imshow` strip under each trace instead of
`axvspan` bands -- the lead-layout/annotation code below does not need to
change for that swap.

Usage
-----
    pre = ECGPreprocessor()
    finding = localize_top_finding(record_path, zeta_clf, anyecg_model, pre)
    recheck = run_crop_and_recheck(finding, zeta_clf, pre, search_window_seconds=10.0)

    fig = plot_localization_overlay(
        record_path, finding, preprocessor=pre, crop_result=recheck,
        output_path="overlay.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe (HPC job nodes / no display) -- set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ecg_preprocessing import ECGPreprocessor, STANDARD_LEAD_ORDER
from localize_finding import LocalizedFinding

if TYPE_CHECKING:
    # Only needed for the type hint; importing crop_and_recheck isn't required
    # to use this module without the crop-and-recheck step.
    from crop_and_recheck import CropRecheckResult


LEAD_DISPLAY_NAMES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
}

LOCALIZED_COLOR = "#d62728"   # red -- anyECG-chat's localized span
CONTROL_COLOR = "#7f7f7f"     # grey -- crop-and-recheck's control window

_DEFAULT_MAX_ALPHA = 0.35
_DEFAULT_TAPER_SECONDS = 0.08   # ~1-2 small-ECG-boxes' worth, at typical clinical paper speed
_DEFAULT_N_BANDS = 80           # shading resolution; cosmetic only, not derived from the model


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Standard smoothstep easing, clipped to [0, 1] input. Used only for the visual taper."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _interval_alpha(
    centers: np.ndarray,
    interval: Tuple[float, float],
    max_alpha: float = _DEFAULT_MAX_ALPHA,
    taper_seconds: float = _DEFAULT_TAPER_SECONDS,
) -> np.ndarray:
    """
    Per-band alpha for shading `interval` at a coarse set of band-center
    times `centers` (seconds): flat at `max_alpha` inside the interval,
    smoothly tapering to 0 over `taper_seconds` on each side, 0 elsewhere.
    See module docstring -- this is a display taper, not a model output.
    """
    start, end = interval
    alpha = np.zeros_like(centers, dtype=float)
    alpha[(centers >= start) & (centers <= end)] = max_alpha

    if taper_seconds > 0:
        left = (centers >= start - taper_seconds) & (centers < start)
        alpha[left] = max_alpha * _smoothstep((centers[left] - (start - taper_seconds)) / taper_seconds)
        right = (centers > end) & (centers <= end + taper_seconds)
        alpha[right] = max_alpha * _smoothstep(((end + taper_seconds) - centers[right]) / taper_seconds)

    return alpha


def _shade_interval(
    ax,
    t_min: float,
    t_max: float,
    interval: Tuple[float, float],
    color: str,
    max_alpha: float = _DEFAULT_MAX_ALPHA,
    taper_seconds: float = _DEFAULT_TAPER_SECONDS,
    n_bands: int = _DEFAULT_N_BANDS,
    label: Optional[str] = None,
    hatch: Optional[str] = None,
) -> None:
    """
    Shades `interval` on `ax` as a small number of adjacent `axvspan` bands
    whose alpha follows `_interval_alpha`, approximating a soft-edged band
    without needing a per-sample overlay (cheap even across 12 subplots).

    Bands are only drawn across [interval_start - taper, interval_end +
    taper] (clipped to [t_min, t_max]) -- not across the whole plot -- so
    this stays fast regardless of how long the displayed window is.
    """
    start, end = interval
    span_start = max(t_min, start - taper_seconds)
    span_end = min(t_max, end + taper_seconds)
    if span_end <= span_start:
        return

    centers = np.linspace(span_start, span_end, max(n_bands, 2))
    alphas = _interval_alpha(centers, interval, max_alpha=max_alpha, taper_seconds=taper_seconds)
    band_width = (span_end - span_start) / len(centers)

    added_label = False
    for c, a in zip(centers, alphas):
        if a <= 1e-3:
            continue
        ax.axvspan(
            c - band_width / 2, c + band_width / 2,
            color=color, alpha=float(a), linewidth=0,
            hatch=hatch,
            label=(label if not added_label else None),
        )
        added_label = True


def plot_localization_overlay(
    record_path: str,
    finding: LocalizedFinding,
    preprocessor: Optional[ECGPreprocessor] = None,
    crop_result: Optional["CropRecheckResult"] = None,
    leads: Optional[Sequence[str]] = None,
    seconds_to_plot: Optional[float] = None,
    output_path: Optional[str] = None,
    figsize_per_lead: float = 1.0,
    dpi: int = 150,
):
    """
    Plots the raw, display-scale ECG signal for `record_path` -- one
    subplot per lead -- with `finding.where_intervals` shaded in red and,
    if `crop_result` is supplied, its `control_interval` shaded in grey
    with a hatch pattern.

    Reads the record via `ECGPreprocessor.load_for_display` (native
    sampling rate, physical units, standard lead order) rather than reusing
    `to_zeta_input`/`to_anyecg_input` -- those are model-normalized (global
    [0,1] or per-lead [-1,1]) and would distort the waveform's actual
    amplitude/morphology on a human-readable plot.

    Parameters
    ----------
    leads:
        Subset of `STANDARD_LEAD_ORDER` to plot, in the order given
        (default: all 12, in standard order).
    seconds_to_plot:
        Truncates the displayed window to `[0, seconds_to_plot)` (default:
        the full loaded record). For a record longer than the window
        anyECG-chat actually searched, pass the same `crop_seconds` used
        when producing `finding` -- otherwise the localized interval will
        appear in only a small part of a much longer plotted trace.
    output_path:
        If given, saves the figure (format inferred from the extension,
        e.g. `.png`, `.pdf`) in addition to returning it. Parent
        directories are created if missing.

    Returns
    -------
    The `matplotlib.figure.Figure` (also saved to `output_path` if given,
    and still open -- call `plt.close(fig)` if generating many of these in
    a batch/loop to avoid accumulating open figures).
    """
    preprocessor = preprocessor or ECGPreprocessor()
    rec = preprocessor.load_for_display(record_path)
    signal = rec.signal  # (T, 12), physical units (mV), standard lead order
    fs = rec.fs

    if seconds_to_plot is not None:
        n = int(round(seconds_to_plot * fs))
        signal = signal[:n]

    t = np.arange(signal.shape[0]) / fs
    t_min, t_max = float(t[0]), float(t[-1]) if len(t) else 0.0

    lead_names = list(leads) if leads is not None else list(STANDARD_LEAD_ORDER)
    lead_idx = [STANDARD_LEAD_ORDER.index(l) for l in lead_names]

    intervals: List[Tuple[float, float]] = finding.where_intervals or []
    control_interval = (
        crop_result.control_interval
        if (crop_result is not None and crop_result.control_interval is not None)
        else None
    )

    fig, axes = plt.subplots(
        len(lead_names), 1, sharex=True,
        figsize=(11, max(figsize_per_lead * len(lead_names), 2.0)),
    )
    if len(lead_names) == 1:
        axes = [axes]

    for ax, li in zip(axes, lead_idx):
        ax.plot(t, signal[:, li], color="black", linewidth=0.7)
        for interval in intervals:
            _shade_interval(ax, t_min, t_max, interval, LOCALIZED_COLOR,
                             label="anyECG-chat localization")
        if control_interval is not None:
            _shade_interval(ax, t_min, t_max, control_interval, CONTROL_COLOR,
                             hatch="///", label="crop-and-recheck control")
        lead_key = STANDARD_LEAD_ORDER[li]
        ax.set_ylabel(LEAD_DISPLAY_NAMES.get(lead_key, lead_key),
                       rotation=0, ha="right", va="center", fontsize=10)
        ax.set_yticks([])
        ax.margins(x=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Time (s)")

    legend_handles = []
    if intervals:
        legend_handles.append(
            Patch(facecolor=LOCALIZED_COLOR, alpha=_DEFAULT_MAX_ALPHA,
                  label=f"Localized: {finding.abnormal_name}")
        )
    if control_interval is not None:
        legend_handles.append(
            Patch(facecolor=CONTROL_COLOR, alpha=_DEFAULT_MAX_ALPHA, hatch="///",
                  label="Crop-and-recheck control")
        )
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=False)

    title_lines = [f"{Path(record_path).name} — {finding.condition} ({finding.prompt_variant} prompt)"]
    title_lines.append(f"Why (ZETA): \"{finding.why_text}\" (score={finding.why_score:.3f})")
    if intervals:
        spans = ", ".join(f"{s:.2f}–{e:.2f}s" for s, e in intervals)
        title_lines.append(f"Where (anyECG-chat): {spans}")
    else:
        title_lines.append("Where (anyECG-chat): not found")
    if crop_result is not None and crop_result.control_score is not None:
        title_lines.append(
            f"Crop-recheck: localized={crop_result.localized_score:.3f} "
            f"vs control={crop_result.control_score:.3f} "
            f"(Δ={crop_result.score_difference:+.3f})"
        )
    elif crop_result is not None and crop_result.note:
        title_lines.append(f"Crop-recheck: {crop_result.note}")

    fig.suptitle("\n".join(title_lines), fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, max(0.90, 1.0 - 0.035 * len(title_lines))))

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig


def plot_localization_overlay_batch(
    findings: List[LocalizedFinding],
    preprocessor: Optional[ECGPreprocessor] = None,
    crop_results: Optional[List[Optional["CropRecheckResult"]]] = None,
    output_dir: str = "overlays",
    **kwargs,
) -> List[Optional[str]]:
    """
    Renders one overlay PNG per finding, named `<record-stem>_<condition>_
    <variant>.png` inside `output_dir`. `crop_results`, if given, must be
    the same length as `findings` and in the same order (as produced by
    `run_crop_and_recheck_batch`) -- a `None` entry (no localized span to
    recheck) is passed through as no control window, not skipped.

    Returns the list of saved file paths, with `None` in place of any
    finding that had no localized interval at all (nothing meaningful to
    overlay -- `finding.where_intervals` is empty/None). Figures are closed
    after saving so a large batch doesn't accumulate open matplotlib
    figures in memory.
    """
    preprocessor = preprocessor or ECGPreprocessor()
    crop_results = crop_results or [None] * len(findings)
    if len(crop_results) != len(findings):
        raise ValueError(
            f"crop_results length ({len(crop_results)}) must match findings length ({len(findings)})."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Optional[str]] = []
    for finding, recheck in zip(findings, crop_results):
        if not finding.where_intervals:
            paths.append(None)
            continue
        stem = Path(finding.record_path).name
        out_path = out_dir / f"{stem}_{finding.condition}_{finding.prompt_variant}.png"
        fig = plot_localization_overlay(
            finding.record_path, finding,
            preprocessor=preprocessor, crop_result=recheck,
            output_path=str(out_path), **kwargs,
        )
        plt.close(fig)
        paths.append(str(out_path))

    return paths


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Quick manual smoke test: plots a record with a hand-specified interval "
                    "(no ZETA/anyECG-chat forward pass -- just exercises the plotting code)."
    )
    parser.add_argument("record_path")
    parser.add_argument("--interval", nargs=2, type=float, default=(2.0, 3.2),
                         metavar=("START", "END"))
    parser.add_argument("--control-interval", nargs=2, type=float, default=None,
                         metavar=("START", "END"))
    parser.add_argument("--seconds-to-plot", type=float, default=10.0)
    parser.add_argument("--output", default="overlay_smoke_test.png")
    args = parser.parse_args()

    dummy_finding = LocalizedFinding(
        record_path=args.record_path,
        condition="VPC",
        abnormal_name="Premature ventricular contraction",
        why_score=0.0,
        why_text="(smoke test -- no real ZETA output)",
        why_index=-1,
        prompt_variant="baseline",
        question="(smoke test)",
        where_raw=f"Duration: {args.interval[0]}s-{args.interval[1]}s",
        where_intervals=[tuple(args.interval)],
    )

    dummy_recheck = None
    if args.control_interval is not None:
        # Build a minimal stand-in without importing crop_and_recheck, so this
        # smoke test has no dependency on scipy being installed.
        class _FakeRecheck:
            control_interval = tuple(args.control_interval)
            localized_score = None
            control_score = None
            score_difference = None
            note = None
        dummy_recheck = _FakeRecheck()

    fig = plot_localization_overlay(
        args.record_path, dummy_finding,
        crop_result=dummy_recheck,
        seconds_to_plot=args.seconds_to_plot,
        output_path=args.output,
    )
    print(f"Saved: {args.output}", file=sys.stderr)