"""
ecg_preprocessing.py

Produces two different preprocessed views of the SAME source WFDB ECG record:

  - ZETA format:      native/target 500 Hz, GLOBAL min-max normalization to [0, 1],
                      MIMIC lead-order swap applied (required for the released
                      ZETA/D-BETA checkpoint to work correctly).
  - anyECG-chat format: resampled to 100 Hz, PER-LEAD min-max normalization to [-1, 1],
                      standard clinical lead order (no swap).

Design note
-----------
This deliberately does NOT use an ABC-based Strategy pattern (abstract
`Resampler` / `Normalizer` base classes with per-format subclasses). With only
two known, fixed target formats -- each dictated by how its respective model
was trained, not by freely-combinable options -- that indirection buys nothing
and makes the pipeline harder to trace. Instead, resample/normalize are small,
pure, independently-testable functions, and each target format gets its own
explicit method (`to_zeta_input`, `to_anyecg_input`) that composes them. If a
third format is needed later (e.g. PULSE / LLaVA-Med baselines, which expect a
rendered PNG of the ECG rather than a tensor -- see anyECG-chat's
`inference.py`), add another method reusing the same helpers; no class
hierarchy needs to change.

Both output methods return a numpy array of shape (12, T) -- leads first, time
second -- matching the convention both source repos converge to right after
their own loading step (ZETA: `ecg.T` in `data_load.py`; anyECG-chat:
`ecg.permute(1, 0)` in `utils.py`). Batching / any further permutation
required by a specific `main.py` or `inference.py` call site is the caller's
responsibility -- check that before feeding a batch in, since ZETA's own
`main.py` has a dataset-name-conditional extra permute that's easy to get
backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import wfdb
from scipy.signal import resample as scipy_resample

# ---------------------------------------------------------------------------
# Canonical lead order (standard clinical convention: aVR, aVL, aVF)
# ---------------------------------------------------------------------------
STANDARD_LEAD_ORDER: Sequence[str] = (
    "i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"
)


@dataclass
class LoadedRecord:
    """Raw signal + metadata straight off disk, before any format-specific step."""
    signal: np.ndarray   # shape (T_native, 12), canonical lead order, zero-padded if needed
    fs: float            # native sampling frequency (Hz)


class ECGPreprocessor:
    """
    Loads a single WFDB record and produces model-ready tensors for both
    ZETA and anyECG-chat from it.

    Usage
    -----
        pre = ECGPreprocessor()
        zeta_input   = pre.to_zeta_input(record_path)
        anyecg_input = pre.to_anyecg_input(record_path)

        # or both at once, sharing the single disk read:
        both = pre.preprocess_both(record_path)
    """

    def __init__(self, standard_lead_order: Sequence[str] = STANDARD_LEAD_ORDER):
        self.standard_lead_order = list(standard_lead_order)

    # ------------------------------------------------------------------ #
    # Shared loading / lead-canonicalization step
    # ------------------------------------------------------------------ #
    def _load_record(self, record_path: str) -> LoadedRecord:
        """
        Reads a WFDB record and reorders/pads leads into STANDARD_LEAD_ORDER.

        Adapted directly from anyECG-chat's `utils.get_ecg_from_path`. The
        renaming rules for reduced-lead databases (e.g. MIT-BIH ST Change's
        'ecg'/'ecg' channels -> 'mlii'/'v1') are dataset-specific hacks
        carried over from that repo -- extend `_rename_lead` if you add a
        raw source database not covered here.
        """
        signal, meta = wfdb.rdsamp(record_path)  # signal: (T, n_sig)
        fs = float(meta["fs"])
        leads_present = [self._rename_lead(name) for name in meta["sig_name"]]

        n_sig = signal.shape[1]
        if n_sig < 12:
            # Special-cased renaming for known reduced-lead sources, mirroring
            # anyECG-chat's utils.py handling of MIT-BIH-style records.
            if leads_present == ["ecg", "ecg"]:
                leads_present = ["mlii", "v1"]
            elif leads_present == ["ecg"]:
                leads_present = ["i"]
            missing = [l for l in self.standard_lead_order if l not in leads_present]
            signal = np.concatenate(
                [signal, np.zeros((signal.shape[0], len(missing)), dtype=signal.dtype)],
                axis=1,
            )
            leads_present = leads_present + missing

        if leads_present != self.standard_lead_order:
            reorder_idx = [leads_present.index(l) for l in self.standard_lead_order]
            signal = signal[:, reorder_idx]

        return LoadedRecord(signal=signal, fs=fs)

    @staticmethod
    def _rename_lead(name: str) -> str:
        """Lowercases and applies the ml-prefix / d3-alias fixes from anyECG-chat."""
        return name.lower().replace("ml", "").replace("d3", "iii")

    # ------------------------------------------------------------------ #
    # Pure, swappable building blocks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        """
        Resamples along the time axis. Matches anyECG-chat's approach:
        `scipy.signal.resample` to a sample count derived from the fs ratio.
        No-op if already at target_fs.
        """
        if orig_fs == target_fs:
            return signal
        target_len = int(round(signal.shape[0] * target_fs / orig_fs))
        return scipy_resample(signal, target_len, axis=0)

    @staticmethod
    def _crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
        """Crops to target_len samples, or zero-pads at the end if shorter."""
        t = signal.shape[0]
        if t == target_len:
            return signal
        if t > target_len:
            return signal[:target_len]
        pad = np.zeros((target_len - t, signal.shape[1]), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=0)

    @staticmethod
    def _normalize_global_01(signal: np.ndarray) -> np.ndarray:
        """
        ZETA's normalization: a SINGLE min/max over the whole (T, 12) array,
        not per lead. Matches `data_load.py`:
            (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg) + 1e-8)
        """
        lo, hi = np.min(signal), np.max(signal)
        return (signal - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _normalize_per_lead_pm1(signal: np.ndarray) -> np.ndarray:
        """
        anyECG-chat's normalization: per-lead min-max to [-1, 1], NaN-safe.
        Matches `utils.py`:
            (ecg - min) / (max - min) * 2 - 1, then nan_to_num
        """
        lo = np.min(signal, axis=0, keepdims=True)
        hi = np.max(signal, axis=0, keepdims=True)
        normed = (signal - lo) / (hi - lo) * 2 - 1
        return np.nan_to_num(normed, nan=0.0)

    @staticmethod
    def _apply_mimic_lead_swap(signal: np.ndarray) -> np.ndarray:
        """
        ZETA-specific quirk: its underlying model was pretrained on MIMIC-ECG,
        whose lead order is (..., aVR, aVF, aVL, ...) -- note aVF before aVL --
        while STANDARD_LEAD_ORDER used here is (..., aVR, aVL, aVF, ...).
        `data_load.py` compensates with `ecg[[4, 5]] = ecg[[5, 4]]` on a
        (12, T) array. Forgetting this silently feeds ZETA's encoder
        leads in the wrong order -- it will run without error and produce
        meaningless similarity scores. signal here is (T, 12); we swap the
        corresponding columns (indices 4, 5 = aVL, aVF).
        """
        swapped = signal.copy()
        swapped[:, [4, 5]] = swapped[:, [5, 4]]
        return swapped

    # ------------------------------------------------------------------ #
    # Public: format-specific outputs
    # ------------------------------------------------------------------ #
    def to_zeta_input(
        self,
        record_path: str,
        target_fs: float = 500.0,
        crop_seconds: float = 10.0,
        apply_mimic_lead_swap: bool = True,
    ) -> np.ndarray:
        """
        Returns a (12, T) float32 array matching ZETA's expected input:
        target_fs Hz, cropped/padded to crop_seconds, globally normalized
        to [0, 1], with the MIMIC lead-order swap applied by default.

        `apply_mimic_lead_swap=True` is the correct default when using the
        released ZETA/D-BETA checkpoint. Only set it False if you are
        evaluating against a differently-trained checkpoint that expects
        standard lead order.
        """
        rec = self._load_record(record_path)
        signal = self._resample(rec.signal, rec.fs, target_fs)
        signal = self._crop_or_pad(signal, int(round(crop_seconds * target_fs)))
        signal = self._normalize_global_01(signal)
        if apply_mimic_lead_swap:
            signal = self._apply_mimic_lead_swap(signal)
        return signal.T.astype(np.float32)  # (12, T)

    def to_zeta_input_window(
        self,
        record_path: str,
        start_seconds: float,
        end_seconds: float,
        target_fs: float = 500.0,
        pad_to_seconds: float = 10.0,
        apply_mimic_lead_swap: bool = True,
    ) -> np.ndarray:
        """
        Like `to_zeta_input`, but extracts a specific [start_seconds,
        end_seconds) window instead of the first `crop_seconds` of the
        record, then zero-pads the (typically much shorter) result up to
        `pad_to_seconds` so it matches ZETA's expected fixed input length.

        IMPORTANT CAVEAT: this is NOT a convention ZETA's own code defines
        or was evaluated with -- ZETA was trained and evaluated on whole
        ~10s records, never on short zero-padded crops. Using this to
        re-score a localized sub-span is an explicit test of whether
        ZETA's encoder is meaningfully sensitive at that granularity, not
        an assumption that it will behave well; a degraded or noisy score
        from a very short window is itself a valid (if inconvenient)
        experimental finding, not necessarily a sign this method is wrong.
        Zero-padding (rather than e.g. stretching the signal to fill
        pad_to_seconds) is chosen only for consistency with anyECG-chat's
        own convention for sub-10s ECGs -- it is a design decision made
        here, not one inherited from ZETA.
        """
        rec = self._load_record(record_path)
        resampled = self._resample(rec.signal, rec.fs, target_fs)
        start_idx = int(round(start_seconds * target_fs))
        end_idx = int(round(end_seconds * target_fs))
        window = resampled[start_idx:end_idx]
        window = self._crop_or_pad(window, int(round(pad_to_seconds * target_fs)))
        window = self._normalize_global_01(window)
        if apply_mimic_lead_swap:
            window = self._apply_mimic_lead_swap(window)
        return window.T.astype(np.float32)  # (12, T)

    def to_anyecg_input(
        self,
        record_path: str,
        target_fs: float = 100.0,
        crop_seconds: float = 10.0,
        ecg_start: Optional[int] = None,
        ecg_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns a (12, T) float32 array matching anyECG-chat's expected
        input: resampled to target_fs Hz, per-lead normalized to [-1, 1],
        standard lead order (no swap).

        For the localization task, pass `ecg_start`/`ecg_end` as raw
        sample indices AT THE RECORD'S NATIVE SAMPLING RATE (matching
        anyECG-chat's `utils.get_ecg_from_path` behavior, where these
        indices come directly from the localization dataset's JSON and
        are applied before resampling). If omitted, the first
        `crop_seconds` of native-rate signal is used instead.
        """
        rec = self._load_record(record_path)
        if ecg_start is not None and ecg_end is not None:
            signal = rec.signal[ecg_start:ecg_end]
        else:
            signal = rec.signal[: int(round(crop_seconds * rec.fs))]
        signal = self._resample(signal, rec.fs, target_fs)
        signal = self._normalize_per_lead_pm1(signal)
        return signal.T.astype(np.float32)  # (12, T)

    def preprocess_both(
        self,
        record_path: str,
        zeta_kwargs: Optional[dict] = None,
        anyecg_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Convenience wrapper producing both formats. Note this still reads
        the WFDB file twice (once per call) rather than sharing a single
        `_load_record` call -- deliberately kept simple for a small
        research pipeline. If I/O becomes a bottleneck at your eval-set
        scale, refactor `_load_record` to be called once and passed into
        both format methods instead.
        """
        zeta_kwargs = zeta_kwargs or {}
        anyecg_kwargs = anyecg_kwargs or {}
        return {
            "zeta": self.to_zeta_input(record_path, **zeta_kwargs),
            "anyecg": self.to_anyecg_input(record_path, **anyecg_kwargs),
        }


if __name__ == "__main__":
    # Minimal smoke test -- replace with a real record path to sanity-check shapes.
    import sys

    if len(sys.argv) > 1:
        pre = ECGPreprocessor()
        out = pre.preprocess_both(sys.argv[1])
        print("ZETA input:   ", out["zeta"].shape, out["zeta"].dtype,
              "range:", out["zeta"].min(), out["zeta"].max())
        print("anyECG input: ", out["anyecg"].shape, out["anyecg"].dtype,
              "range:", out["anyecg"].min(), out["anyecg"].max())
    else:
        print("Usage: python ecg_preprocessing.py <path/to/wfdb/record (no extension)>")
