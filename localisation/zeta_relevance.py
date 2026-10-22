"""
zeta_relevance.py

Localisation for ZETA's zero-shot observation scores, on the path that actually
produces the score, plus a fixed version of last term's cross-attention map and
an evaluation harness to compare them.

Why this exists (read zeta's main.py: extract_ecg_features / get_diseases_probs)
-------------------------------------------------------------------------------
The score for one observation pair (P_i, N_i) is

    ecg_vec  = normalize( pooler( proj( encoder([CLS; frames]) ) )[token 0] )
    p_i      = sigmoid( (cos(ecg_vec, text_P_i) - cos(ecg_vec, text_N_i)) / 0.5 )

so the quantity to explain is the *logit difference* of the cited pair. The path is
CNN (16 samples = 32 ms per frame) -> 8-layer self-attention encoder -> CLS token.
The BertCrossLayer stack (multi_modal_*_layers) is NOT on this path.

Methods (each returns one non-negative value per CNN frame, length ~312 for 10 s)
---------------------------------------------------------------------------------
  chefer           Chefer et al. 2021 relevance: R <- R + mean_h[(dTarget/dA * A)^+] @ R
                   through the 8 encoder self-attention layers; CLS row.
  rollout          Abnar & Zuidema attention rollout (target-independent); CLS row.
  attn_last        Raw last-layer CLS->frame attention, head mean (target-independent).
  attn_mean        Raw CLS->frame attention averaged over all layers and heads.
  gradcam_cnn      Grad-CAM at the CNN feature-extractor output (channels x frames).
  cross_attn_fixed Last term's text->ECG cross-attention map, with the modality-
                   embedding bug fixed (text = type 0, ECG = type 1), full 10 s
                   input, no extra re-softmax, no per-map min-max.

Evaluation
----------
  --eval      deletion / insertion curves (+ random-order control) on the target logit
  --sanity    Adebayo-style cascading weight randomisation (Spearman vs original map)
  --gt-intervals "2.1-3.4,5.0-6.2"   AUROC / mass-in-GT against annotated spans

Point it at the ZETA repo with --repo or ZETA_REPO in .env (put on sys.path; relative
--config / --observations resolve against it), so it can be run from any directory.
Smoke-tested on a tiny random-weight copy of the architecture only; NOT yet run
against the released checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 500
SOFTMAX_TEMP = 0.5  # ZETA's training/inference temperature (main.py: softmax(i / 0.5))
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ALL_METHODS = ["chefer", "rollout", "attn_last", "attn_mean", "gradcam_cnn", "cross_attn_fixed"]
GRAPH_METHODS = ["chefer", "rollout", "attn_last", "attn_mean", "gradcam_cnn"]  # need only the ECG branch


# ─────────────────────────────────────────────────────────────────────────────
# The ECG branch that produces ZETA's zero-shot score
# ─────────────────────────────────────────────────────────────────────────────

class ECGBranch(nn.Module):
    """The four modules on the zero-shot scoring path (shares weights with the full model)."""

    def __init__(self, ecg_encoder, class_embedding, proj, pooler):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.class_embedding = class_embedding
        self.proj = proj
        self.pooler = pooler

    @classmethod
    def from_model(cls, model):
        return cls(model.ecg_encoder, model.class_embedding,
                   model.multi_modal_ecg_proj, model.unimodal_ecg_pooler)

    def encode(self, x: torch.Tensor):
        """x: (B, 12, T) -> (unit-norm ecg vector (B, D), projected sequence (B, 1+F, D)).
        Mirrors main.py's extract_ecg_features exactly (CLS prepended, token-0 pooler)."""
        feats, pad = self.ecg_encoder.get_embeddings(x, padding_mask=None)
        cls = self.class_embedding.repeat((len(feats), 1, 1))
        seq = self.ecg_encoder.get_output(torch.cat([cls, feats], dim=1), pad)
        out = self.proj(seq)
        vec = F.normalize(self.pooler(out), dim=-1)
        return vec, out


def frame_geometry(branch: ECGBranch):
    """(samples per CNN frame, callable T -> n_frames) from the model's own conv config."""
    layers = eval(branch.ecg_encoder.cfg.conv_feature_layers)  # same eval the repo uses
    spf = int(np.prod([s for (_, _, s) in layers]))
    if any(k != s for (_, k, s) in layers):
        print("WARNING: conv kernel != stride; frame->sample mapping is only approximate")

    def n_frames(T):
        L = T
        for (_, k, s) in layers:
            L = (L - k) // s + 1
        return int(L)

    return spf, n_frames


def target_scalar(vec, pos_vec, neg_vec, mode="logit_diff"):
    sp, sn = vec @ pos_vec, vec @ neg_vec
    if mode == "logit_diff":   # = logit of ZETA's per-pair possibility score
        return (sp - sn) / SOFTMAX_TEMP
    if mode == "pos_sim":
        return sp
    raise ValueError(mode)


# ─────────────────────────────────────────────────────────────────────────────
# Attention taps: explicit softmax(QK^T)V that keeps per-head probs and their grads
# ─────────────────────────────────────────────────────────────────────────────

class _AttnTap:
    """Replaces one MultiHeadAttention.forward (instance-level). Numerically equal to
    F.multi_head_attention_forward in eval mode with no masks (checked in the self-test)."""

    def __init__(self, mha):
        self.mha = mha
        self.attn = None
        self.grad = None
        mha.forward = self._forward

    def restore(self):
        if "forward" in self.mha.__dict__:
            del self.mha.__dict__["forward"]

    def _forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        m = self.mha
        if key_padding_mask is not None or attn_mask is not None:
            raise NotImplementedError("tap supports unmasked, equal-length inputs only")
        if m.training or getattr(m.dropout, "apply_during_inference", False):
            raise RuntimeError("call model.eval() (attention dropout must be off)")
        T, B, C = query.shape
        H, d = m.n_heads, m.d_heads
        q = (m.q_proj(query) * m.scaling).reshape(T, B * H, d).transpose(0, 1)
        k = m.k_proj(key).reshape(T, B * H, d).transpose(0, 1)
        v = m.v_proj(value).reshape(T, B * H, d).transpose(0, 1)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)  # (B*H, T, T)
        a4 = attn.view(B, H, T, T)
        self.attn = a4.detach()
        if a4.requires_grad:
            a4.register_hook(lambda g: setattr(self, "grad", g.detach()))
        # route the value-weighting through the hooked 4-D tensor so its grad is captured
        out = torch.bmm(a4.reshape(B * H, T, T), v).transpose(0, 1).reshape(T, B, C)
        return m.out_proj(out), None


@contextmanager
def attention_taps(branch: ECGBranch):
    taps = [_AttnTap(layer.self_attn) for layer in branch.ecg_encoder.encoder.layers]
    try:
        yield taps
    finally:
        for t in taps:
            t.restore()


# ─────────────────────────────────────────────────────────────────────────────
# Relevance methods
# ─────────────────────────────────────────────────────────────────────────────

def explain(branch: ECGBranch, x_np: np.ndarray, pos_vec, neg_vec, target="logit_diff"):
    """One forward + backward. x_np: (12, T) float in ZETA's global [0,1] normalisation.
    Returns ({method: (n_frames,) array}, target_logit)."""
    dev = next(branch.parameters()).device
    branch.eval()
    branch.requires_grad_(False)  # only the input needs a graph; avoids accumulating param grads
    if branch.ecg_encoder.feature_grad_mult != 1.0:
        raise RuntimeError("feature_grad_mult != 1.0: gradients through the CNN are rescaled/blocked")
    x = torch.tensor(x_np, dtype=torch.float32, device=dev).unsqueeze(0).requires_grad_(True)

    store = {}

    def _fh(mod, inp, out):
        out.retain_grad()
        store["feat"] = out

    h = branch.ecg_encoder.feature_extractor.register_forward_hook(_fh)
    try:
        with attention_taps(branch) as taps, torch.enable_grad():
            vec, _ = branch.encode(x)
            tgt = target_scalar(vec, pos_vec, neg_vec, target)[0]
            tgt.backward()
            maps = _maps_from_taps(taps, store["feat"])
    finally:
        h.remove()
    return maps, float(tgt.detach())


def _maps_from_taps(taps, feat):
    N = taps[0].attn.shape[-1]
    dev = taps[0].attn.device
    eye = torch.eye(N, device=dev)

    # chefer: R <- R + E_h[(grad * A)^+] @ R, layer by layer from the input side
    R = eye.clone()
    for t in taps:
        cam = (t.grad * t.attn).clamp(min=0).mean(dim=1)[0]
        R = R + cam @ R
    chefer = R[0, 1:]

    # rollout: product of (0.5*A_mean + 0.5*I), CLS row
    Rr = eye.clone()
    for t in taps:
        a = 0.5 * t.attn.mean(dim=1)[0] + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        Rr = a @ Rr
    rollout = Rr[0, 1:]

    attn_last = taps[-1].attn.mean(dim=1)[0, 0, 1:]
    attn_mean = torch.stack([t.attn.mean(dim=1)[0, 0, 1:] for t in taps]).mean(0)

    # Grad-CAM on CNN output: feat (1, C, F)
    if feat.grad is None:
        raise RuntimeError("no gradient reached the CNN output")
    w = feat.grad.mean(dim=2, keepdim=True)
    gradcam = torch.relu((w * feat.detach()).sum(dim=1))[0]

    to_np = lambda t: t.detach().float().cpu().numpy()
    return {"chefer": to_np(chefer), "rollout": to_np(rollout), "attn_last": to_np(attn_last),
            "attn_mean": to_np(attn_mean), "gradcam_cnn": to_np(gradcam)}


@torch.no_grad()
def cross_attention_map(model, ecg_out: torch.Tensor, text_seq: torch.Tensor):
    """Last term's heatmap, corrected. ecg_out: (1, 1+F, D) projected ECG sequence;
    text_seq: (1, L, D) projected T5 sequence for ONE observation.
    Differences from the old script: modality ids (text 0 / ECG 1, as in M3AEModel.forward),
    one pass over the full 10 s, no softmax(attn/0.1), no min-max."""
    dev = ecg_out.device
    L, N = text_seq.shape[1], ecg_out.shape[1]
    x = text_seq + model.modality_type_embeddings(torch.zeros(1, L, dtype=torch.long, device=dev))
    y = ecg_out + model.modality_type_embeddings(torch.ones(1, N, dtype=torch.long, device=dev))
    ext_t = torch.zeros(1, 1, 1, L, device=dev)  # all tokens valid (no padding used)
    ext_e = torch.zeros(1, 1, 1, N, device=dev)
    maps = []
    for tl, el in zip(model.multi_modal_language_layers, model.multi_modal_ecg_layers):
        xo = tl(x, y, ext_t, ext_e, output_attentions=True)
        yo = el(y, x, ext_e, ext_t, output_attentions=True)
        maps.append(xo[2][0])  # (H, L, 1+F): text queries -> ECG keys
        x, y = xo[0], yo[0]
    m = torch.stack(maps).mean(dim=(0, 1))  # layers, heads -> (L, 1+F)
    return m.mean(dim=0)[1:].float().cpu().numpy()  # mean over text queries, drop CLS key


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: deletion / insertion
# ─────────────────────────────────────────────────────────────────────────────

def _trapz(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


@torch.no_grad()
def _batched_target(branch, xs: torch.Tensor, pos_vec, neg_vec, target, bs=16):
    out = []
    for i in range(0, len(xs), bs):
        vec, _ = branch.encode(xs[i:i + bs])
        out.append(target_scalar(vec, pos_vec, neg_vec, target))
    return torch.cat(out).float().cpu().numpy()


def faithfulness(branch, x_np, relevance, pos_vec, neg_vec, target="logit_diff",
                 n_steps=20, n_random=10, seed=0):
    """Deletion: fill the top-k frames with the per-lead median, track the target.
    Insertion: start fully filled, restore top-k frames. Frames map to 16-sample blocks
    on all leads. Lower deletion AUC / higher insertion AUC = more faithful. Always read
    against the random-order row. Filling creates off-distribution inputs, and the CNN's
    first layer is a per-channel GroupNorm over the WHOLE record, so editing any frames
    rescales every frame (non-local effect, worst near the fully-filled end). Compare
    methods to each other and to random, not to an absolute number."""
    dev = next(branch.parameters()).device
    spf, n_frames_fn = frame_geometry(branch)
    T = x_np.shape[1]
    Fn = n_frames_fn(T)
    fill = np.median(x_np, axis=1, keepdims=True)
    ks = np.unique(np.round(np.linspace(0, Fn, n_steps + 1)).astype(int))
    fracs = ks / Fn
    rng = np.random.default_rng(seed)

    def curves(order):
        block_rank = np.empty(Fn)
        block_rank[order] = np.arange(Fn)
        sample_rank = np.empty(T)
        sample_rank[:Fn * spf] = np.repeat(block_rank, spf)
        sample_rank[Fn * spf:] = block_rank[-1]  # trailing samples (no frame of their own) follow the last frame
        dels, inss = [], []
        for k in ks:
            m = sample_rank < k
            dels.append(np.where(m[None, :], fill, x_np))
            inss.append(np.where(m[None, :], x_np, fill))
        to_t = lambda L: torch.tensor(np.stack(L), dtype=torch.float32, device=dev)
        return (_batched_target(branch, to_t(dels), pos_vec, neg_vec, target),
                _batched_target(branch, to_t(inss), pos_vec, neg_vec, target))

    res = {}
    for name, rel in relevance.items():
        order = np.argsort(-np.asarray(rel), kind="stable")
        d, i = curves(order)
        res[name] = {"del_curve": d, "ins_curve": i, "del_auc": _trapz(d, fracs), "ins_auc": _trapz(i, fracs)}
    rd, ri = [], []
    for _ in range(n_random):
        d, i = curves(rng.permutation(Fn))
        rd.append(d); ri.append(i)
    d, i = np.mean(rd, 0), np.mean(ri, 0)
    res["random"] = {"del_curve": d, "ins_curve": i, "del_auc": _trapz(d, fracs), "ins_auc": _trapz(i, fracs)}
    res["_fracs"] = fracs
    res["_full"] = float(res["random"]["del_curve"][0])
    res["_blank"] = float(res["random"]["del_curve"][-1])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: sanity (Adebayo et al.) and ground-truth overlap
# ─────────────────────────────────────────────────────────────────────────────

def _randomise_(module, gen):
    for p in module.parameters():
        if p.ndim >= 2:
            p.data.normal_(0.0, 0.02, generator=gen)


def sanity_cascade(branch, x_np, pos_vec, neg_vec, target="logit_diff", seed=0):
    """Randomise from the output end towards the input, cumulatively; report Spearman
    correlation of each method's map with the original. A faithful method's map should
    decorrelate; one that stays high is mostly reading the input, not the model."""
    from scipy.stats import spearmanr
    base, _ = explain(branch, x_np, pos_vec, neg_vec, target)
    b = copy.deepcopy(branch)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    enc = b.ecg_encoder
    stages = [("pooler", [b.pooler]), ("proj", [b.proj])]
    stages += [(f"enc_layer_{i}", [enc.encoder.layers[i]]) for i in reversed(range(len(enc.encoder.layers)))]
    stages += [("cnn+pos", [enc.feature_extractor, enc.conv_pos] + ([enc.post_extract_proj] if enc.post_extract_proj is not None else []))]
    rows = []
    for name, mods in stages:
        for m in mods:
            _randomise_(m, gen)
        maps, logit = explain(b, x_np, pos_vec, neg_vec, target)
        row = {"stage": name, "logit": logit}
        for k in GRAPH_METHODS:
            r = spearmanr(base[k], maps[k]).correlation if np.std(maps[k]) > 0 else float("nan")
            row[k] = float(r)
        rows.append(row)
    return rows


def gt_frame_mask(intervals_s, n_frames, spf, fs=SAMPLE_RATE):
    mask = np.zeros(n_frames, bool)
    for a, b in intervals_s:
        mask[int(a * fs // spf): int(math.ceil(b * fs / spf))] = True
    return mask


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def localisation_metrics(rel, mask):
    rel = np.clip(np.asarray(rel, float), 0, None)
    tot = rel.sum()
    return {"auroc": auroc(rel, mask),
            "mass_in_gt": float(rel[mask].sum() / tot) if tot > 0 else float("nan"),
            "gt_fraction": float(mask.mean()),   # chance level for mass_in_gt
            "argmax_in_gt": bool(mask[int(np.argmax(rel))])}


def relevance_to_intervals(rel, spf, fs=SAMPLE_RATE, top_frac=0.15, merge_gap=2):
    """Top-fraction frames merged into runs -> [(start_s, end_s)]. For crop-and-recheck /
    display only; top_frac is a convention, not a calibrated confidence."""
    rel = np.asarray(rel, float)
    k = max(1, int(round(top_frac * len(rel))))
    keep = np.flatnonzero(rel >= np.sort(rel)[-k])
    runs, start, prev = [], keep[0], keep[0]
    for f in keep[1:]:
        if f - prev > merge_gap + 1:
            runs.append((start, prev)); start = f
        prev = f
    runs.append((start, prev))
    return [(float(s * spf / fs), float((e + 1) * spf / fs)) for s, e in runs]


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_methods(x_np, maps, spf, title, path, lead=1, primary="chefer", fs=SAMPLE_RATE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = x_np.shape[1]
    t = np.arange(T) / fs
    names = list(maps)
    fig, axes = plt.subplots(1 + len(names), 1, figsize=(15, 1.6 + 0.7 * len(names) + 2.2), sharex=True,
                             gridspec_kw={"height_ratios": [3] + [1] * len(names)})

    def to_img(rel):
        r = np.asarray(rel, float)
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
        return np.repeat(r, spf)[None, :]

    ax = axes[0]
    if primary in maps:
        ax.imshow(to_img(maps[primary]), aspect="auto", cmap="Reds", alpha=0.45,
                  extent=[0, len(maps[primary]) * spf / fs, x_np[lead].min(), x_np[lead].max()], zorder=0)
    ax.plot(t, x_np[lead], color="black", lw=0.8, zorder=2)
    ax.set_ylabel(f"lead {LEAD_NAMES[lead]}")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, T / fs)
    for a, n in zip(axes[1:], names):
        a.imshow(to_img(maps[n]), aspect="auto", cmap="magma", extent=[0, len(maps[n]) * spf / fs, 0, 1])
        a.set_yticks([])
        a.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model / data plumbing (mirrors your existing script)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(config_path, checkpoint_path, device):
    from types import SimpleNamespace
    from models.cmelt import M3AEModel
    with open(config_path) as f:
        cfg = SimpleNamespace(**json.load(f)["model"])
    model = M3AEModel(cfg)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:  # torch>=2.6 defaults to weights_only=True
        print(f"torch.load default failed ({type(e).__name__}); retrying weights_only=False "
              f"(only do this for a checkpoint you trust)")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    state.pop("ecg_encoder.mask_emb", None)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def encode_text(model, tokenizer, text, device):
    """Same as main.py's extract_language_features: T5 -> proj -> token-0 pooler -> L2 norm."""
    enc = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    seq = model.multi_modal_language_proj(model.language_encoder(**enc)[0])
    vec = F.normalize(model.unimodal_language_pooler(seq), dim=-1)[0]
    return vec, seq


def load_ecg(args) -> np.ndarray:
    """-> (12, T) float32, GLOBAL min-max over the whole record (as data_load.py does)."""
    if args.npy:
        arr = np.load(args.npy).astype(np.float32)
        if arr.shape[0] == 12 and arr.shape[1] != 12:
            arr = arr.T
    else:
        import pandas as pd
        import wfdb
        root = args.ptbxl_root
        if args.ecg_id is not None:
            db = pd.read_csv(Path(root) / "ptbxl_database.csv", index_col="ecg_id")
            fn = db.loc[args.ecg_id, "filename_hr"]
        else:
            fn = args.filename_hr
        arr, _ = wfdb.rdsamp(str(Path(root) / fn))
        arr = arr.astype(np.float32)
    arr = arr[:5000, :]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr.T.copy()


def parse_intervals(s):
    return [tuple(float(v) for v in part.split("-")) for part in s.split(",") if part.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from dotenv import find_dotenv, load_dotenv
    load_dotenv()                                # .env next to this script (or a parent)
    load_dotenv(find_dotenv(usecwd=True))        # else .env in the current directory; never overrides
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ecg_id", type=int)
    g.add_argument("--filename_hr")
    g.add_argument("--npy", help="(T,12) or (12,T) raw signal at 500 Hz")
    p.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    p.add_argument("--condition", required=True)
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    p.add_argument("--config", default="configs/config.json",
                   help="relative paths resolve against the ZETA repo root")
    p.add_argument("--observations", default="configs/observations.json",
                   help="relative paths resolve against the ZETA repo root")
    p.add_argument("--repo", default=os.getenv("ZETA_REPO"),
                   help="ZETA repo root (for `import models`); default: ZETA_REPO from .env "
                        "(ZETA_REPO_ROOT also accepted)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--obs-index", type=int, default=None, help="positive observation to explain (default: top-scoring)")
    p.add_argument("--target", default="logit_diff", choices=["logit_diff", "pos_sim"])
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--eval", action="store_true", help="deletion/insertion curves")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--sanity", action="store_true", help="cascading weight-randomisation check")
    p.add_argument("--gt-intervals", default=None, help='seconds, e.g. "2.1-3.4,5.0-6.2"')
    p.add_argument("--plot-lead", type=int, default=1)
    p.add_argument("--primary", default="chefer", choices=ALL_METHODS)
    p.add_argument("--save_dir", default=".")
    args = p.parse_args()

    if not args.repo:
        p.error("ZETA repo root not set: pass --repo or set ZETA_REPO in .env")
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "models" / "cmelt.py").is_file():
        p.error(f"{repo} does not look like the ZETA repo root (models/cmelt.py not found)")
    sys.path.insert(0, str(repo))
    in_repo = lambda q: str(q if os.path.isabs(q) else repo / q)
    args.config, args.observations = in_repo(args.config), in_repo(args.observations)
    from transformers import T5TokenizerFast

    dev = torch.device(args.device)
    model = load_model(args.config, args.checkpoint, dev)
    tok = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)
    with open(args.observations) as f:
        obs = json.load(f)
    pos_t = [t.lower() for t in obs[args.condition]["P"]]
    neg_t = [t.lower() for t in obs[args.condition]["N"]]
    n_pairs = min(len(pos_t), len(neg_t))

    x = load_ecg(args)
    branch = ECGBranch.from_model(model)
    spf, _ = frame_geometry(branch)

    with torch.no_grad():
        vec, ecg_out = branch.encode(torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0))
    pv, ps = zip(*[encode_text(model, tok, t, dev) for t in pos_t[:n_pairs]])
    nv, _ = zip(*[encode_text(model, tok, t, dev) for t in neg_t[:n_pairs]])
    probs = [float(torch.sigmoid(target_scalar(vec, pv[i], nv[i])[0])) for i in range(n_pairs)]
    print(f"Condition {args.condition}: possibility score {np.mean(probs):.3f} (ZETA temperature {SOFTMAX_TEMP})")
    for i in range(n_pairs):
        print(f"  [{i}] P(pair)={probs[i]:.3f}  {pos_t[i]}")
    idx = args.obs_index if args.obs_index is not None else int(np.argmax(probs))
    print(f"\nExplaining pair {idx}: \"{pos_t[idx]}\"  vs  \"{neg_t[idx]}\"  (target = {args.target})")

    graph_methods = [m for m in args.methods if m in GRAPH_METHODS]
    maps, logit = explain(branch, x, pv[idx], nv[idx], args.target)
    maps = {k: maps[k] for k in graph_methods}
    if "cross_attn_fixed" in args.methods:
        maps["cross_attn_fixed"] = cross_attention_map(model, ecg_out, ps[idx])
    print(f"target value {logit:.3f}")

    stem = f"ECG-{args.ecg_id if args.ecg_id is not None else 'x'}_{args.condition}_obs{idx}"
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"condition": args.condition, "obs_index": idx, "obs_text": pos_t[idx],
              "target": args.target, "target_value": logit, "score_pair": probs[idx]}

    if args.gt_intervals:
        mask = gt_frame_mask(parse_intervals(args.gt_intervals), len(next(iter(maps.values()))), spf)
        report["gt"] = {k: localisation_metrics(v, mask) for k, v in maps.items()}
        print("\nGround-truth overlap (mass_in_gt should beat gt_fraction):")
        for k, v in report["gt"].items():
            print(f"  {k:<18} AUROC {v['auroc']:.3f}  mass {v['mass_in_gt']:.3f} (chance {v['gt_fraction']:.3f})  argmax_in_gt {v['argmax_in_gt']}")

    if args.eval:
        ev = faithfulness(branch, x, maps, pv[idx], nv[idx], args.target, args.n_steps, args.n_random)
        report["faithfulness"] = {k: {"del_auc": v["del_auc"], "ins_auc": v["ins_auc"]}
                                  for k, v in ev.items() if not k.startswith("_")}
        report["faithfulness"]["_full_blank"] = [ev["_full"], ev["_blank"]]
        print(f"\nDeletion/insertion on target (full={ev['_full']:.2f}, all-filled={ev['_blank']:.2f}); "
              f"want deletion AUC low, insertion AUC high, both beating 'random':")
        for k, v in ev.items():
            if not k.startswith("_"):
                print(f"  {k:<18} del {v['del_auc']:8.3f}   ins {v['ins_auc']:8.3f}")

    if args.sanity:
        rows = sanity_cascade(branch, x, pv[idx], nv[idx], args.target)
        report["sanity"] = rows
        print("\nCascading randomisation, Spearman vs original map (should fall towards 0):")
        print("  " + "stage".ljust(14) + "".join(k[:10].rjust(12) for k in GRAPH_METHODS))
        for r in rows:
            print("  " + r["stage"].ljust(14) + "".join(f"{r[k]:12.3f}" for k in GRAPH_METHODS))

    np.savez(out_dir / f"{stem}_relevance.npz", ecg=x, spf=spf, **maps)
    report["intervals_top15pct"] = {k: relevance_to_intervals(v, spf) for k, v in maps.items()}
    with open(out_dir / f"{stem}_report.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    title = f'{args.condition}  "{pos_t[idx]}"  pair score {probs[idx]:.2f}'
    plot_methods(x, maps, spf, title, out_dir / f"{stem}_methods.png", lead=args.plot_lead, primary=args.primary)
    print(f"\nSaved to {out_dir}/{stem}_*")


if __name__ == "__main__":
    main()