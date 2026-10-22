#!/usr/bin/env python3
"""
zeta_perturb_playground.py -- ZETA zero-shot scoring + occlusion ("where") maps, as one standalone script.

What it does
  1. Loads a 12-lead ECG and scores every condition in ZETA's observation bank
     (the same maths as ZETA's main.py: cosine similarity to each positive/negative sentence,
      pair-wise softmax at temperature 0.5, mean over pairs).
  2. Shows the PER-OBSERVATION probabilities for a condition ("rsr' pattern in v1 and v2" vs its paired negative ...).
  3. Localises WHEN the evidence for a condition / single observation is in the recording by occlusion:
     slide a window along the ECG, replace it with a neutral baseline, re-encode, and record how much the score drops.
     Output = continuous time-importance map (plus optional per-lead map), a plot, and "Duration: a-b" spans
     in the same format anyECG-chat uses.
  4. Checks the map is faithful: deleting the top-ranked time bins should hurt the score far more than deleting
     random bins of the same total length (and the bottom-ranked ones should hurt least).

ZETA's repo is not modified or required to be next to this script: ZETA_REPO in .env is added to sys.path.
The one source patch ZETA needs for transformers >= 4.47 (cross_layer.py import) is applied at runtime instead.

--------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------
  cp .env.example .env     # fill in ZETA_REPO, ZETA_CKPT (best.pt from huggingface.co/jialucode/ZETA)
  pip install torch "transformers==4.50.0" omegaconf wfdb scipy numpy matplotlib sentencepiece
  python zeta_perturb_playground.py                                   # interactive
  python zeta_perturb_playground.py --ecg records500/00000/00172_hr --explain CRBBB
  python zeta_perturb_playground.py --check                           # validate .env + imports

Interactive commands (`help` inside):
  load <record> [start_s [end_s]]     WFDB record (no extension) or .npy/.csv with fs=
  top [n]                             rank conditions by ZETA score
  scores <COND>                       per-observation probabilities for one condition
  explain <COND> [obs#]               occlusion map (+plot, spans, faithfulness) for the condition or its obs# (1-based)
  set window|stride|baseline|metric|perlead|conditions|duration <v>
  info | settings | help | quit

Things to keep in mind (also printed where relevant)
  * ZETA was trained on 500 Hz / 10 s / 12-lead ECGs. Use PTB-XL records500 (_hr), not records100. 100 Hz input is
    upsampled here, with a warning.
  * Occlusion creates inputs ZETA never saw in training. That is why step 4 (faithfulness) is built in, and why you can
    compare two different baselines (`set baseline linear|mean`, or --compare-baselines).
  * The map says where ZETA's score for that observation is sensitive, i.e. "evidence the model uses"; it is not
    proof the feature is physiologically there.
  * ZETA looks at a whole-recording embedding, so the default maps occlude all leads at once. Use `set perlead on` for
    a per-lead map (12x the cost).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SCRIPT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# .env handling
# ----------------------------------------------------------------------------------------------


def _parse_env_file(path: Path) -> dict:
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if val[:1] in ("'", '"') and val.find(val[0], 1) > 0:
            val = val[1:val.find(val[0], 1)]
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        out[key] = val
    return out


def load_env(explicit: Optional[str]) -> Optional[Path]:
    """--env > ./.env > <script dir>/.env. Real environment variables win over the file."""
    candidates = [Path(explicit)] if explicit else [Path.cwd() / ".env", SCRIPT_DIR / ".env"]
    for p in candidates:
        if p.is_file():
            for k, v in _parse_env_file(p).items():
                os.environ.setdefault(k, v)
            os.environ["_ENV_FILE_DIR"] = str(p.resolve().parent)
            return p.resolve()
    if explicit:
        sys.exit(f"[error] --env file not found: {explicit}")
    return None


def env_path(key: str, *aliases: str) -> Optional[str]:
    for k in (key, *aliases):
        v = os.environ.get(k, "").strip()
        if v:
            v = os.path.expanduser(os.path.expandvars(v))
            if not os.path.isabs(v) and os.environ.get("_ENV_FILE_DIR"):
                cand = os.path.join(os.environ["_ENV_FILE_DIR"], v)
                if os.path.exists(cand):
                    return cand
            return v
    return None


@dataclass
class Config:
    repo: str
    ckpt: str
    t5_id: str
    obs_json: str
    ecg_dir: Optional[str]
    output_dir: str
    batch_size: int


def build_config(need_model: bool = True) -> Config:
    repo = env_path("ZETA_REPO")
    ckpt = env_path("ZETA_CKPT") or (os.path.join(repo, "checkpoints", "best.pt") if repo else None)
    obs = env_path("OBSERVATIONS_JSON") or (os.path.join(repo, "configs", "observations.json") if repo else None)
    problems = []
    if not repo:
        problems.append("ZETA_REPO is not set (folder containing main.py, models/ and configs/)")
    else:
        for rel in ("models/cmelt.py", "configs/config.json"):
            if not os.path.isfile(os.path.join(repo, rel)):
                problems.append(f"ZETA_REPO={repo} is missing {rel}")
    if obs and not os.path.isfile(obs):
        problems.append(f"observations file not found: {obs}")
    if need_model and (not ckpt or not os.path.isfile(ckpt)):
        problems.append(f"ZETA_CKPT is not a file: {ckpt} (download best.pt from huggingface.co/jialucode/ZETA)")
    if problems:
        sys.exit("[error] .env problems:\n  - " + "\n  - ".join(problems) + "\n(see .env.example)")
    return Config(repo=repo, ckpt=ckpt or "", t5_id=env_path("T5_MODEL_ID") or "google/flan-t5-base",
                  obs_json=obs, ecg_dir=env_path("ECG_DIR"),
                  output_dir=env_path("OUTPUT_DIR") or str(Path.cwd() / "zeta_outputs"),
                  batch_size=int(os.environ.get("BATCH_SIZE", "32")))


# ----------------------------------------------------------------------------------------------
# Constants from ZETA's code
# ----------------------------------------------------------------------------------------------

ZETA_FS = 500            # Hz the model was trained on
TEMP = 0.5               # softmax temperature in get_diseases_probs()
STD_LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
# ZETA expects MIMIC lead order (I, II, III, aVR, aVF, aVL, V1..V6): rows 4 and 5 swapped vs. standard.
SWAP = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]   # self-inverse permutation

# ----------------------------------------------------------------------------------------------
# ECG loading (mirrors ZETA's data_load.py: first N samples @500 Hz, global min-max to [0,1], AVL/AVF swap)
# ----------------------------------------------------------------------------------------------


@dataclass
class Record:
    name: str
    raw_std: "object"      # (12, T) @500 Hz, un-normalised, standard lead order (for plotting)
    x: "object"            # (12, T) float32, ZETA input (normalised, ZETA lead order)
    present: List[bool]    # per STANDARD lead slot
    source_fs: float
    window_s: Tuple[float, float]
    units: str
    notes: List[str]

    @property
    def seconds(self) -> float:
        return self.x.shape[1] / ZETA_FS


def _canon(names: Sequence[str]) -> List[str]:
    n = [s.strip().lower() for s in names]
    if n == ["ecg", "ecg"]:
        n = ["mlii", "v1"]
    elif n == ["ecg"]:
        n = ["i"]
    return [s.replace("ml", "").replace("d3", "iii") for s in n]


def load_record(spec: str, start_s: Optional[float], end_s: Optional[float], ecg_dir: Optional[str],
                fs: Optional[float] = None, leads: Optional[Sequence[str]] = None,
                duration: float = 10.0) -> Record:
    import numpy as np
    from scipy.signal import resample

    p = Path(os.path.expanduser(spec))
    if not p.is_absolute() and ecg_dir:
        p = Path(ecg_dir) / spec
    suffix = p.suffix.lower()
    start_s = 0.0 if start_s is None else float(start_s)
    notes: List[str] = []
    units = ""
    if suffix in (".npy", ".csv", ".txt"):
        arr = np.load(p) if suffix == ".npy" else np.loadtxt(p, delimiter=",")
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] <= 12 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if fs is None:
            raise ValueError("Array input needs a sampling rate: `load file.npy fs=500`.")
        names = list(leads) if leads else (STD_LEADS if arr.shape[1] == 12 else None)
        if names is None or len(names) != arr.shape[1]:
            raise ValueError(f"{arr.shape[1]} channels: pass leads=<names>, one per channel.")
        a = int(round(start_s * fs))
        b = int(round(end_s * fs)) if end_s is not None else int(round((start_s + duration) * fs))
        sig, source_fs, name = arr[a:b].astype(np.float64), float(fs), p.stem
    else:
        import wfdb
        base = str(p.with_suffix("")) if suffix in (".hea", ".dat") else str(p)
        if not os.path.isfile(base + ".hea"):
            raise FileNotFoundError(f"No WFDB header at {base}.hea (give the record path WITHOUT extension)")
        hdr = wfdb.rdheader(base)
        source_fs = float(hdr.fs)
        a = int(round(start_s * source_fs))
        b = int(round(end_s * source_fs)) if end_s is not None else int(round((start_s + duration) * source_fs))
        rec = wfdb.rdrecord(base, sampfrom=a, sampto=min(b, hdr.sig_len))
        sig = np.nan_to_num(rec.p_signal.astype(np.float64))
        names = list(leads) if leads else list(rec.sig_name)
        units = ",".join(sorted(set(u for u in rec.units if u))) if rec.units else ""
        name = os.path.basename(base)
    if sig.shape[0] < 50:
        raise ValueError("Window is empty/too short -- check start/end seconds.")

    canon = _canon(names)
    sig12 = np.zeros((sig.shape[0], 12))
    present = [False] * 12
    for c, nm in enumerate(canon):
        if nm in STD_LEADS and not present[STD_LEADS.index(nm)]:
            sig12[:, STD_LEADS.index(nm)] = sig[:, c]
            present[STD_LEADS.index(nm)] = True
        else:
            notes.append(f"channel '{names[c]}' not used (unknown or duplicate lead name)")
    if not any(present):
        raise ValueError(f"No channel in {list(names)} maps to a standard lead; pass leads=...")
    if sum(present) < 12:
        notes.append(f"only {sum(present)}/12 leads recorded; the rest are zero-filled. ZETA was trained on 12-lead ECGs, "
                     f"so scores will be unreliable.")

    if source_fs != ZETA_FS:
        n_out = int(round(sig12.shape[0] * ZETA_FS / source_fs))
        sig12 = resample(sig12, n_out, axis=0)
        if source_fs < ZETA_FS:
            notes.append(f"source is {source_fs:g} Hz; upsampled to {ZETA_FS} Hz. Prefer the 500 Hz recording "
                         f"(PTB-XL records500/*_hr) -- ZETA never saw upsampled data.")
    sig12 = sig12.T                                    # (12, T) standard order
    T = sig12.shape[1]
    rec_idx = [i for i, ok in enumerate(present) if ok]
    lo, hi = sig12[rec_idx].min(), sig12[rec_idx].max()   # global min-max over RECORDED leads only (zero-pad stays 0)
    xn = (sig12 - lo) / (hi - lo + 1e-8)
    for i, ok in enumerate(present):
        if not ok:
            xn[i] = 0.0
    x = xn[SWAP].astype(np.float32)                    # ZETA lead order
    return Record(name=name, raw_std=sig12, x=x, present=present, source_fs=source_fs,
                  window_s=(start_s, start_s + T / ZETA_FS), units=units, notes=notes)


# ----------------------------------------------------------------------------------------------
# Model wrapper: ECG embedding + text embedding exactly as in ZETA main.py's zero-shot path
# ----------------------------------------------------------------------------------------------


class ZetaScorer:
    def __init__(self, cfg: Config):
        import torch
        import transformers
        self.torch = torch
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[warn] no CUDA GPU: running on CPU (works, but occlusion maps will be slow).")
        sys.path.insert(0, cfg.repo)    # ZETA's top-level package is the generic name `models`; make sure it wins
        # runtime replacement for the repo patch in README/requirements: these three helpers moved in transformers>=4.47
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu
        for n in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(mu, n) and hasattr(pu, n):
                setattr(mu, n, getattr(pu, n))
        from transformers import T5EncoderModel, T5TokenizerFast
        if cfg.t5_id != "google/flan-t5-base":   # M3AEModel hard-codes the hub id; redirect to a local copy
            orig = T5EncoderModel.from_pretrained

            def _redirect(name, *a, **k):
                return orig(cfg.t5_id if name == "google/flan-t5-base" else name, *a, **k)
            T5EncoderModel.from_pretrained = staticmethod(_redirect)
        from types import SimpleNamespace
        from models.cmelt import M3AEModel
        loaded_from = os.path.abspath(sys.modules["models.cmelt"].__file__)   # `models` itself is a namespace package
        if not loaded_from.startswith(os.path.abspath(cfg.repo)):
            sys.exit(f"[error] `models.cmelt` was imported from {loaded_from}, not ZETA_REPO; another `models` package "
                     f"is shadowing it.")
        mcfg = SimpleNamespace(**json.load(open(os.path.join(cfg.repo, "configs", "config.json")))["model"])
        t0 = time.time()
        print(f"[load] building ZETA (downloads {cfg.t5_id} on first use) ...")
        model = M3AEModel(mcfg)
        try:
            ck = torch.load(cfg.ckpt, map_location="cpu")
        except Exception:      # torch>=2.6 defaults to weights_only=True; the checkpoint may hold non-tensor objects
            ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"]
        sd.pop("ecg_encoder.mask_emb", None)
        print("[load] checkpoint:", model.load_state_dict(sd, strict=True))
        self.model = model.eval().to(self.device)
        self.tok = T5TokenizerFast.from_pretrained(cfg.t5_id)
        print(f"[load] ready in {time.time() - t0:.0f}s on {self.device}")

    # ECG (B,12,T) float -> (B,768) L2-normalised
    def encode_ecg(self, x) -> "object":
        torch = self.torch
        import torch.nn.functional as F
        m = self.model
        with torch.inference_mode():
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            feats, pad = m.ecg_encoder.get_embeddings(x, padding_mask=None)
            cls = m.class_embedding.repeat((len(feats), 1, 1))
            feats = m.ecg_encoder.get_output(torch.cat([cls, feats], dim=1), pad)
            out = m.multi_modal_ecg_proj(feats)
            return F.normalize(m.unimodal_ecg_pooler(out), dim=-1).float()

    def encode_text(self, phrases: Sequence[str]) -> "object":
        torch = self.torch
        import torch.nn.functional as F
        m = self.model
        outs = []
        with torch.inference_mode():
            for t in phrases:     # one at a time, as in ZETA's extract_language_features (no padding effects)
                enc = self.tok(t, truncation=True, return_tensors="pt").to(self.device)
                h = m.multi_modal_language_proj(m.language_encoder(**enc)[0])
                outs.append(m.unimodal_language_pooler(h))
            return F.normalize(torch.cat(outs, 0), dim=-1).float()


class Bank:
    """Observation bank: condition -> paired positive/negative sentences + cached text embeddings."""

    def __init__(self, scorer, path: str):
        self.scorer = scorer
        raw = json.load(open(path))
        self.pairs: Dict[str, Tuple[List[str], List[str]]] = {}
        for c, v in raw.items():
            P, N = [s.lower() for s in v["P"]], [s.lower() for s in v["N"]]
            n = min(len(P), len(N))
            if len(P) != len(N):
                print(f"[warn] {c}: {len(P)} positive vs {len(N)} negative sentences; truncated to {n}")
            if n:
                self.pairs[c] = (P[:n], N[:n])
        self._feats: dict = {}

    def feats(self, cond: str):
        if cond not in self._feats:
            P, N = self.pairs[cond]
            self._feats[cond] = (self.scorer.encode_text(P), self.scorer.encode_text(N))
        return self._feats[cond]

    def find(self, name: str) -> str:
        key = {k.lower(): k for k in self.pairs}.get(name.strip().lower())
        if not key:
            raise KeyError(f"unknown condition '{name}'. Try `top` or `list`.")
        return key


def pair_margins(E, Tp, Tn):
    """(W,D),(n,D),(n,D) -> (W,n) logits; softmax([sp,sn]/T)[0] == sigmoid(margin)."""
    return (E @ Tp.T - E @ Tn.T) / TEMP


def reduce_score(margin, metric: str, obs: Optional[int]):
    """margin (W,n) -> (W,) score. metric 'prob' = ZETA's (mean pair probability); 'margin' = mean logit."""
    import torch
    if obs is not None:
        margin = margin[:, obs:obs + 1]
    return torch.sigmoid(margin).mean(1) if metric == "prob" else margin.mean(1)


# ----------------------------------------------------------------------------------------------
# Occlusion
# ----------------------------------------------------------------------------------------------


def make_windows(T: int, w: int, stride: int) -> List[Tuple[int, int]]:
    w = min(w, T)
    starts = list(range(0, T - w + 1, stride))
    if starts[-1] + w < T:
        starts.append(T - w)
    return [(s, s + w) for s in starts]


def apply_mask(x, a: int, b: int, baseline: str, rows: Optional[Sequence[int]] = None):
    """Return a copy of x (12,T) with samples [a,b) replaced by a neutral baseline on `rows` (default all leads)."""
    import numpy as np
    y = x.copy()
    T = x.shape[1]
    for r in (range(x.shape[0]) if rows is None else rows):
        if baseline == "mean":
            y[r, a:b] = x[r].mean()
        else:  # 'linear': straight line between the samples just outside the window
            left = x[r, a - 1] if a > 0 else x[r, min(b, T - 1)]
            right = x[r, b] if b < T else x[r, a - 1 if a > 0 else 0]
            y[r, a:b] = np.linspace(left, right, b - a)
    return y


def embed_many(scorer, arrays: Sequence, batch: int):
    import torch
    outs = []
    for i in range(0, len(arrays), batch):
        import numpy as np
        outs.append(scorer.encode_ecg(np.stack(arrays[i:i + batch])))
    return torch.cat(outs, 0)


@dataclass
class Occlusion:
    windows: List[Tuple[int, int]]
    groups: List[Optional[int]]     # None = all leads together; else a ZETA-order lead row
    E_full: "object"                # (1,D)
    E_win: Dict[Optional[int], "object"]   # group -> (W,D)
    baseline: str
    T: int


def run_occlusion(scorer, x, window_s: float, stride_s: float, baseline: str, per_lead: bool, batch: int) -> Occlusion:
    T = x.shape[1]
    wins = make_windows(T, int(round(window_s * ZETA_FS)), max(1, int(round(stride_s * ZETA_FS))))
    groups: List[Optional[int]] = [None] + (list(range(12)) if per_lead else [])
    E_full = embed_many(scorer, [x], batch)
    E_win = {}
    for g in groups:
        rows = None if g is None else [g]
        if g is not None and not float(abs(x[g]).sum()):   # zero-filled lead: nothing to occlude
            continue
        E_win[g] = embed_many(scorer, [apply_mask(x, a, b, baseline, rows) for a, b in wins], batch)
    return Occlusion(wins, groups, E_full, E_win, baseline, T)


def importance_map(occ: Occlusion, Tp, Tn, metric: str, obs: Optional[int], group: Optional[int] = None):
    """Per-sample average score drop over all windows covering that sample. Returns (map, s_full)."""
    import numpy as np
    s0 = float(reduce_score(pair_margins(occ.E_full, Tp, Tn), metric, obs)[0])
    sw = reduce_score(pair_margins(occ.E_win[group], Tp, Tn), metric, obs).cpu().numpy()
    acc, cnt = np.zeros(occ.T), np.zeros(occ.T)
    for (a, b), s in zip(occ.windows, sw):
        acc[a:b] += s0 - s
        cnt[a:b] += 1
    return acc / np.maximum(cnt, 1), s0


def extract_spans(m, frac: float = 0.5, min_gap_s: float = 0.15, max_spans: int = 3) -> List[Tuple[float, float, float]]:
    """Contiguous regions where importance >= frac * max (positive only). -> [(start_s, end_s, peak)] strongest first."""
    import numpy as np
    pk = float(m.max())
    if pk <= 0:
        return []
    on = m >= frac * pk
    spans, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            spans.append([i, j + 1])
            i = j + 1
        else:
            i += 1
    merged = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) / ZETA_FS < min_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    out = [(a / ZETA_FS, b / ZETA_FS, float(m[a:b].max())) for a, b in merged]
    return sorted(out, key=lambda t: -t[2])[:max_spans]


# ----------------------------------------------------------------------------------------------
# Faithfulness: delete top / bottom / random time bins and compare score drops
# ----------------------------------------------------------------------------------------------


def faithfulness(scorer, x, m, Tp, Tn, metric: str, obs: Optional[int], baseline: str, batch: int,
                 bin_s: float = 0.25, fractions: Sequence[float] = (0.1, 0.2, 0.3), n_random: int = 30, seed: int = 0):
    import numpy as np
    T = x.shape[1]
    nb = max(1, int(round(T / (bin_s * ZETA_FS))))
    edges = np.linspace(0, T, nb + 1).astype(int)
    imp = np.array([m[edges[i]:edges[i + 1]].mean() for i in range(nb)])
    rng = np.random.default_rng(seed)

    def delete(bins):
        y = x
        for b in bins:
            y = apply_mask(y, int(edges[b]), int(edges[b + 1]), baseline)
        return y

    s_full = float(reduce_score(pair_margins(scorer.encode_ecg(x[None]), Tp, Tn), metric, obs)[0])
    order = np.argsort(-imp)
    rows = []
    for f in fractions:
        k = max(1, int(round(f * nb)))
        variants = [delete(order[:k]), delete(order[::-1][:k])] + [delete(rng.choice(nb, k, replace=False)) for _ in range(n_random)]
        s = reduce_score(pair_margins(embed_many(scorer, variants, batch), Tp, Tn), metric, obs).cpu().numpy()
        drop = s_full - s
        rows.append(dict(fraction=f, top=float(drop[0]), bottom=float(drop[1]),
                         random_mean=float(drop[2:].mean()), random_sd=float(drop[2:].std())))
    for r in rows:
        r["z"] = (r["top"] - r["random_mean"]) / (r["random_sd"] + 1e-9)   # top-ranked vs random deletions, in sd units
    ordered = all(r["top"] > r["random_mean"] > r["bottom"] or (r["top"] > r["random_mean"] and r["top"] > r["bottom"])
                  for r in rows)
    if all(r["z"] > 2 for r in rows) and ordered:
        level = "strong"
    elif ordered and all(r["z"] > 1 for r in rows):
        level = "weak"      # right direction at every fraction, but not clearly beyond random-deletion variability
    else:
        level = "none"
    return dict(score_full=s_full, rows=rows, level=level, faithful=level == "strong", bin_s=bin_s)


# ----------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------


def plot_map(rec: Record, m, lead_maps: Optional[Dict[int, "object"]], spans, title: str, out: Path) -> Path:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(rec.raw_std.shape[1]) / ZETA_FS
    dur = t[-1] + 1 / ZETA_FS
    idx = [i for i, ok in enumerate(rec.present) if ok]
    allm = [m] + list((lead_maps or {}).values())
    v = max(float(np.abs(a).max()) for a in allm) or 1.0
    fig = plt.figure(figsize=(14, 1.35 * len(idx) + 2.6))
    gs = fig.add_gridspec(len(idx) + 1, 1, height_ratios=[1.6] + [1] * len(idx), hspace=0.12)
    top = fig.add_subplot(gs[0])
    top.plot(t, m, color="black", lw=1.2)
    top.axhline(0, color="gray", lw=0.6)
    top.fill_between(t, 0, m, where=m > 0, color="tab:red", alpha=0.35)
    top.fill_between(t, 0, m, where=m < 0, color="tab:blue", alpha=0.35)
    for a, b, _ in spans:
        top.axvspan(a, b, color="gold", alpha=0.25, lw=0)
    top.set_ylabel("score drop\nwhen occluded")
    top.set_xlim(0, dur)
    top.tick_params(labelbottom=False)
    top.set_title(title, fontsize=10)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[k + 1], sharex=top)
        y = rec.raw_std[i]
        pad = 0.08 * (y.max() - y.min() + 1e-9)
        y0, y1 = y.min() - pad, y.max() + pad
        mm = lead_maps.get(SWAP[i], m) if lead_maps else m   # SWAP is its own inverse: standard slot -> ZETA row
        ax.imshow(np.asarray(mm)[None, :], extent=(0, dur, y0, y1), aspect="auto", cmap="RdBu_r",
                  vmin=-v, vmax=v, alpha=0.55, zorder=0, interpolation="nearest")
        ax.plot(t, y, color="black", lw=0.8, zorder=2)
        ax.set_xlim(0, dur)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(STD_LEADS[i].upper())
        if k < len(idx) - 1:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel(f"seconds from window start (source {rec.window_s[0]:.1f}-{rec.window_s[1]:.1f}s). "
                  f"red = occluding this lowers the score (evidence), blue = raises it")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------------------


class Session:
    def __init__(self, cfg: Config, scorer, bank, args):
        self.cfg, self.scorer, self.bank = cfg, scorer, bank
        self.rec: Optional[Record] = None
        self.window = args.window
        self.stride = args.stride
        self.baseline = args.baseline
        self.metric = args.metric
        self.per_lead = args.per_lead
        self.duration = args.duration
        self.conditions: Optional[List[str]] = None
        self.span_frac = args.span_frac
        self.out_dir = Path(args.output_dir or cfg.output_dir)
        self._occ_cache: dict = {}

    def _need(self) -> Record:
        if self.rec is None:
            raise RuntimeError("No ECG loaded. Use: load <record> [start_s [end_s]]")
        return self.rec

    def _log(self, kind: str, **f):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind,
                                     ecg=self.rec.name if self.rec else None, **f)) + "\n")

    # -- commands
    def load(self, spec, start=None, end=None, fs=None, leads=None):
        self.rec = load_record(spec, start, end, self.cfg.ecg_dir, fs=fs, leads=leads, duration=self.duration)
        self._occ_cache.clear()
        self.info()
        for n in self.rec.notes:
            print(f"  [note] {n}")

    def info(self):
        r = self._need()
        print(f"  {r.name}: {r.seconds:.1f}s ({r.window_s[0]:.1f}-{r.window_s[1]:.1f}s of source @ {r.source_fs:g} Hz "
              f"-> {ZETA_FS} Hz), leads recorded: {sum(r.present)}/12")

    def top(self, n: int = 10):
        r = self._need()
        E = self.scorer.encode_ecg(r.x[None])
        names = self.conditions or list(self.bank.pairs)
        res = []
        for c in names:
            Tp, Tn = self.bank.feats(c)
            res.append((float(reduce_score(pair_margins(E, Tp, Tn), "prob", None)[0]), c))
        res.sort(reverse=True)
        print(f"  ZETA condition scores (mean pair probability), top {n} of {len(res)}:")
        for s, c in res[:n]:
            print(f"    {c:<10} {s:.3f}")
        self._log("top", scores=dict((c, s) for s, c in res[:n]))

    def scores(self, cond: str):
        r = self._need()
        c = self.bank.find(cond)
        Tp, Tn = self.bank.feats(c)
        E = self.scorer.encode_ecg(r.x[None])
        mg = pair_margins(E, Tp, Tn)[0].cpu().numpy()
        P, N = self.bank.pairs[c]
        print(f"  {c}: mean pair probability {float((1 / (1 + math.e ** (-mg))).mean()):.3f}")
        for i, (p, n_, g) in enumerate(zip(P, N, mg), 1):
            print(f"   {i}. p={1 / (1 + math.e ** (-g)):.3f}  [+] {p}   vs   [-] {n_}")

    def _occ(self) -> Occlusion:
        key = (self.window, self.stride, self.baseline, self.per_lead)
        if key not in self._occ_cache:
            r = self._need()
            t0 = time.time()
            n_w = len(make_windows(r.x.shape[1], int(round(self.window * ZETA_FS)), max(1, int(round(self.stride * ZETA_FS)))))
            print(f"  occluding: {n_w} windows of {self.window}s (stride {self.stride}s, baseline={self.baseline}"
                  f"{', per-lead x12' if self.per_lead else ''}) ...")
            self._occ_cache[key] = run_occlusion(self.scorer, r.x, self.window, self.stride, self.baseline,
                                                 self.per_lead, self.cfg.batch_size)
            print(f"  done in {time.time() - t0:.1f}s")
        return self._occ_cache[key]

    def explain(self, cond: str, obs: Optional[int] = None, check: bool = True):
        r = self._need()
        c = self.bank.find(cond)
        Tp, Tn = self.bank.feats(c)
        P, N = self.bank.pairs[c]
        if obs is not None and not (1 <= obs <= len(P)):
            raise ValueError(f"{c} has {len(P)} observations; use 1..{len(P)}")
        oi = None if obs is None else obs - 1
        occ = self._occ()
        m, s0 = importance_map(occ, Tp, Tn, self.metric, oi)
        lead_maps = None
        if self.per_lead:
            lead_maps = {g: importance_map(occ, Tp, Tn, self.metric, oi, g)[0] for g in occ.E_win if g is not None}
        spans = extract_spans(m, self.span_frac)
        what = f"{c}" + (f" obs {obs}: '{P[oi]}' vs '{N[oi]}'" if oi is not None else " (mean of its observations)")
        print(f"\n  {what}\n  full-ECG {self.metric} = {s0:.3f}; max drop {m.max():.3f} at {m.argmax() / ZETA_FS:.2f}s; "
              f"min {m.min():.3f}")
        if spans:
            print("  Duration: " + ", ".join(f"{a:.1f}s-{b:.1f}s" for a, b, _ in sorted(spans)) +
                  f"   (regions >= {self.span_frac:.0%} of the peak importance)")
        else:
            print("  No region raises the score when kept: the map has no positive evidence (nothing to localise).")
        faith = None
        if check:
            faith = faithfulness(self.scorer, r.x, m, Tp, Tn, self.metric, oi, self.baseline, self.cfg.batch_size)
            print(f"  faithfulness (delete {faith['bin_s']}s bins; score drop, higher = bigger damage):")
            print("    fraction   top-ranked   random(mean±sd)   bottom-ranked   z(top vs random)")
            for row in faith["rows"]:
                print(f"    {row['fraction']:>7.0%}   {row['top']:>10.3f}   {row['random_mean']:>7.3f}±{row['random_sd']:<6.3f}"
                      f"   {row['bottom']:>10.3f}   {row['z']:>8.1f}")
            msg = {"strong": "map is FAITHFUL: deleting top-ranked bins hurts far more than random bins (z > 2 at every fraction).",
                   "weak": "map is WEAKLY faithful: top-ranked > random > bottom-ranked at every fraction, but z is only 1-2 in "
                           "places. The broad pattern is usable; do not trust exact span edges or fine detail.",
                   "none": "map is NOT faithful to this model's behaviour: do not rely on the spans or plot above (try another "
                           "baseline/window, or this score may not be driven by a localised feature)."}[faith["level"]]
            print("  verdict: " + msg)
        title = f"{r.name} | ZETA {c}" + (f" obs{obs}" if obs else "") + f" | {self.metric}={s0:.2f} | window={self.window}s, {self.baseline}"
        png = plot_map(r, m, lead_maps, spans, title,
                       self.out_dir / f"{r.name}_{re.sub(r'[^A-Za-z0-9]+', '', c)}{'_o' + str(obs) if obs else ''}_{time.strftime('%H%M%S')}.png")
        print(f"  plot: {png}")
        self._log("explain", condition=c, obs=obs, metric=self.metric, window_s=self.window, stride_s=self.stride,
                  baseline=self.baseline, score_full=s0, spans=spans, faithfulness=faith, plot=str(png))

    def compare_baselines(self, cond: str, obs: Optional[int] = None):
        """Maps from 'linear' and 'mean' baselines should agree if the result is not an artefact of the baseline."""
        import numpy as np
        c = self.bank.find(cond)
        Tp, Tn = self.bank.feats(c)
        old = self.baseline
        ms = {}
        for b in ("linear", "mean"):
            self.baseline = b
            ms[b] = importance_map(self._occ(), Tp, Tn, self.metric, None if obs is None else obs - 1)[0]
        self.baseline = old
        r = float(np.corrcoef(ms["linear"], ms["mean"])[0, 1])
        print(f"  baseline agreement for {c}: Pearson r(linear, mean) = {r:.2f}  (>~0.7 = robust to the baseline choice)")

    def set(self, k: str, v: str):
        if k == "window":
            self.window = float(v)
        elif k == "stride":
            self.stride = float(v)
        elif k == "baseline":
            if v not in ("linear", "mean"):
                raise ValueError("baseline: linear | mean")
            self.baseline = v
        elif k == "metric":
            if v not in ("margin", "prob"):
                raise ValueError("metric: margin | prob")
            self.metric = v
        elif k == "perlead":
            self.per_lead = v.lower() in ("on", "true", "1", "yes")
        elif k == "conditions":
            self.conditions = None if v.lower() in ("all", "") else [self.bank.find(x) for x in v.split(",")]
        elif k == "duration":
            self.duration = float(v)
        elif k == "spanfrac":
            self.span_frac = float(v)
        else:
            raise ValueError("settable: window stride baseline metric perlead conditions duration spanfrac")
        print(f"  {k} = {v}")

    def settings(self):
        print(f"  window={self.window}s stride={self.stride}s baseline={self.baseline} metric={self.metric} "
              f"perlead={self.per_lead} duration={self.duration}s spanfrac={self.span_frac} "
              f"conditions={self.conditions or 'all'} outputs={self.out_dir}")


HELP = """\
  load <record> [start_s [end_s]] [fs=500] [leads=i,ii]   WFDB record base path (no extension) / .npy / .csv
  top [n]                       rank conditions by ZETA score
  scores <COND>                 per-observation probabilities
  explain <COND> [obs#]         occlusion map + spans + faithfulness check + plot (obs# is 1-based)
  compare <COND> [obs#]         do the 'linear' and 'mean' baselines give the same map?
  list                          condition names in the bank
  set window|stride|baseline|metric|perlead|conditions|duration|spanfrac <v>
        window 0.6 (s), stride 0.1 (s), baseline linear|mean, metric margin|prob, perlead on|off,
        conditions CRBBB,PVC (restrict `top`) or all, spanfrac 0.5
  info | settings | help | quit
"""


def repl(s: Session):
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    print("\nType `help`. Example:  load records500/00000/00172_hr   then   top   then   explain CRBBB\n")
    while True:
        try:
            line = input("zeta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()
        try:
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd == "help":
                print(HELP)
            elif cmd == "load":
                toks = rest.split()
                if not toks:
                    raise ValueError("usage: load <record> [start_s [end_s]]")
                kv = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in toks[1:] if "=" in t}
                pos = [t for t in toks[1:] if "=" not in t]
                s.load(toks[0], float(pos[0]) if pos else None, float(pos[1]) if len(pos) > 1 else None,
                       fs=float(kv["fs"]) if "fs" in kv else None, leads=kv["leads"].split(",") if "leads" in kv else None)
            elif cmd == "top":
                s.top(int(rest) if rest else 10)
            elif cmd == "scores":
                s.scores(rest)
            elif cmd in ("explain", "compare"):
                toks = rest.split()
                if not toks:
                    raise ValueError(f"usage: {cmd} <COND> [obs#]")
                (s.explain if cmd == "explain" else s.compare_baselines)(toks[0], int(toks[1]) if len(toks) > 1 else None)
            elif cmd == "list":
                print("  " + " ".join(s.bank.pairs))
            elif cmd == "set":
                k, _, v = rest.partition(" ")
                s.set(k, v.strip())
            elif cmd == "info":
                s.info()
            elif cmd == "settings":
                s.settings()
            else:
                print(f"  unknown command '{cmd}' (try `help`)")
        except Exception as e:  # keep the session alive
            print(f"  [error] {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="ZETA zero-shot scoring + occlusion localisation sandbox.")
    ap.add_argument("--env")
    ap.add_argument("--ecg")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fs", type=float)
    ap.add_argument("--leads")
    ap.add_argument("--top", type=int, metavar="N", help="print the top-N conditions for --ecg")
    ap.add_argument("--explain", metavar="COND", help="run occlusion + faithfulness for this condition")
    ap.add_argument("--obs", type=int, help="1-based observation index within --explain's condition")
    ap.add_argument("--compare-baselines", action="store_true")
    ap.add_argument("--window", type=float, default=0.6, help="occlusion window in seconds (default 0.6 ~ one beat)")
    ap.add_argument("--stride", type=float, default=0.1)
    ap.add_argument("--baseline", choices=["linear", "mean"], default="linear")
    ap.add_argument("--metric", choices=["margin", "prob"], default="margin")
    ap.add_argument("--per-lead", action="store_true")
    ap.add_argument("--span-frac", type=float, default=0.5)
    ap.add_argument("--output-dir")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--check", action="store_true", help="validate .env and imports, then exit")
    args = ap.parse_args()

    env = load_env(args.env)
    print(f"[env] {env if env else 'no .env found (shell environment only)'}")
    cfg = build_config(need_model=True)
    print(f"[env] repo: {cfg.repo}")

    if args.check:
        import importlib
        bad = []
        for mod in ("torch", "transformers", "omegaconf", "wfdb", "scipy", "numpy", "matplotlib", "sentencepiece"):
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{mod}: {type(e).__name__}: {e}")
        for b in bad:
            print("  [missing]", b)
        try:
            import torch
            print(f"[check] CUDA: {torch.cuda.is_available()}")
        except Exception:
            pass
        print(f"[check] ckpt={cfg.ckpt}\n[check] observations={cfg.obs_json}\n[check] t5={cfg.t5_id}")
        sys.exit(1 if bad else 0)

    scorer = ZetaScorer(cfg)
    bank = Bank(scorer, cfg.obs_json)
    sess = Session(cfg, scorer, bank, args)
    one_shot = bool(args.top or args.explain)
    if args.ecg:
        try:
            sess.load(args.ecg, args.start, args.end, fs=args.fs, leads=args.leads.split(",") if args.leads else None)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[error] could not load ECG: {e}")
    elif one_shot:
        sys.exit("[error] --top/--explain need --ecg")
    if args.top:
        sess.top(args.top)
    if args.explain:
        sess.explain(args.explain, args.obs)
        if args.compare_baselines:
            sess.compare_baselines(args.explain, args.obs)
    if not one_shot or args.interactive:
        repl(sess)


if __name__ == "__main__":
    main()