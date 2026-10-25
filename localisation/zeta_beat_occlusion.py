"""
zeta_beat_occlusion.py

Does ZETA's score for an observation depend on WHICH beat carries the feature, or does it treat all
beats alike?  Per-beat occlusion measures this directly: remove one beat at a time and see how much each
observation's score drops.

    python zeta_beat_occlusion.py chimera --obs-filter obs_auroc.csv --out beat_chimera
    python zeta_beat_occlusion.py natural --obs-filter obs_auroc.csv --out beat_natural
    python zeta_beat_occlusion.py summary --dir beat_chimera

chimera  (controlled ground truth)
    Host = a NORM record. Donor = a record with condition C (e.g. CLBBB). k beats of the host are replaced
    by donor beats (real morphology, aligned at the beat peak, boundaries matched), so we KNOW which beats
    carry the feature. Each beat of the chimera is then scored by
        occlusion  drop in the observation's target when that beat is removed (the measured effect)
        chefer     mean Chefer relevance over the beat's frames (observation-specific)
        rollout    mean rollout relevance over the beat's frames (same for every observation)
        energy     beat energy (model-free: do we just find the biggest beats?)
        random     shuffled scores
    and each method's AUROC for picking out the transplanted beats is computed. The SAME is computed on the
    unmodified host with the same beat positions ("null"): a method that favours certain positions or large
    beats gets AUROC > 0.5 there too, so only chimera-minus-null counts as finding the transplanted beat.
    A trial is "responsive" when the chimera's target for the observation is higher than the host's.

natural  (no ground truth)
    Per-beat occlusion on real labelled PTB-XL records: how unequal are beats (spread of the drops), and
    do the drops agree with Chefer / rollout / energy? Writes example figures.

The occlusion removes a window [-pre, +post] s around each detected beat (default -0.20 / +0.40 s: QRS and
the ST-T segment) by per-lead linear interpolation across the window (--fill median uses the lead median
instead). The first CNN layer is a GroupNorm over the whole record, so removing a beat changes every
frame slightly: read the drops relative to the null, not as absolute numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import zeta_relevance as Z
import zeta_relevance_batch as B

FS = Z.SAMPLE_RATE
METHODS = ["occlusion", "chefer", "rollout", "energy", "random"]


# ─────────────────────────────────────────────────────────────────────────────
# beat utilities
# ─────────────────────────────────────────────────────────────────────────────

def valid_peaks(x, pre, post):
    pk = Z.beat_peaks(x, FS)
    if pk is None:
        return np.array([], int)
    T = x.shape[1]
    return np.array([p for p in pk if p - pre - 1 >= 0 and p + post < T], int)


def occlude_beats(x, peaks, pre, post, fill="interp"):
    """(n_beats, 12, T): copy of x with each beat's window removed."""
    out = np.repeat(x[None], len(peaks), axis=0).copy()
    med = np.median(x, axis=1)
    for j, p in enumerate(peaks):
        lo, hi = p - pre, p + post
        if fill == "median":
            out[j, :, lo:hi] = med[:, None]
        else:
            a, b = x[:, lo - 1], x[:, hi]
            ramp = np.linspace(0, 1, hi - lo + 2)[1:-1]
            out[j, :, lo:hi] = a[:, None] + (b - a)[:, None] * ramp[None, :]
    return out


def transplant(host, donor, hp, dp, pre, post, blend=20):
    """Replace host windows around hp with donor windows around dp; offsets are ramped so both window
    edges meet the host exactly (no step), then clipped to the host's [0, 1] range."""
    out = host.copy()
    for h, d in zip(hp, dp):
        seg = donor[:, d - pre:d + post].copy()
        a_h, b_h = host[:, h - pre - 1], host[:, h + post]
        a_d, b_d = donor[:, d - pre - 1], donor[:, d + post]
        ramp = np.linspace(0, 1, post + pre + 2)[1:-1]
        off = (a_h - a_d)[:, None] * (1 - ramp)[None, :] + (b_h - b_d)[:, None] * ramp[None, :]
        out[:, h - pre:h + post] = seg + off
    return np.clip(out, 0.0, 1.0)


def beat_mean(map_, peaks, pre, post, spf):
    n = len(map_)
    out = []
    for p in peaks:
        a, b = max(0, (p - pre) // spf), min(n, -(-(p + post) // spf))
        out.append(float(np.mean(map_[a:b])) if b > a else 0.0)
    return np.array(out)


def beat_energy(x, peaks, pre, post):
    e = np.concatenate([(np.diff(x, axis=1) ** 2).sum(0), [0.0]])
    return np.array([e[p - pre:p + post].sum() for p in peaks])


def auroc(scores, labels):
    return B.fast_auroc(np.asarray(scores, float), labels)


def top1_hit(scores, labels):
    return float(np.asarray(labels, bool)[int(np.argmax(scores))])


class Ctx:
    """Everything that needs the model, so the scoring function is the same for chimera and natural."""

    def __init__(self, args):
        import pandas as pd
        from transformers import T5TokenizerFast
        Z.resolve_repo(self._p(), args)
        if not args.ptbxl_root or not (Path(args.ptbxl_root) / "records500").is_dir():
            raise SystemExit("PTB-XL folder not found: pass --ptbxl_root or set PTBXL_DATASET in .env. Got: %r"
                             % args.ptbxl_root)
        self.args = args
        self.dev = torch.device(args.device)
        self.swap = args.lead_swap == "on"
        self.pre, self.post = int(round(args.pre * FS)), int(round(args.post * FS))
        csv = args.csv or str(Path(args.repo).expanduser() / "data_split" / "ptbxl" / args.split /
                              f"ptbxl_{args.split}_test.csv")
        df = pd.read_csv(csv)
        self.df = df[df["strat_fold"].isin(args.folds)].reset_index(drop=True)
        with open(args.observations) as f:
            self.obs = json.load(f)
        self.label_cols = [c for c in self.df.columns[6:] if c in self.obs]
        self.allowed = B.load_obs_filter(args.obs_filter, args.min_obs_auroc) if args.obs_filter else None
        model = Z.load_model(args.config, args.checkpoint, self.dev)
        tok = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)
        self.model = model
        self.branch = Z.ECGBranch.from_model(model)
        self.spf, self.nf = Z.frame_geometry(self.branch)
        self.bank = B.TextBank(model, tok, self.obs, self.dev)
        self.rng = np.random.RandomState(args.seed)

    @staticmethod
    def _p():
        import argparse as _a
        return _a.ArgumentParser()

    def obs_ids(self, cond):
        n = len(self.bank.get(cond)["pv"])
        ids = self.allowed.get(cond, []) if self.allowed is not None else range(n)
        return [i for i in ids if i < n]

    def read(self, row):
        return Z.read_ptbxl(self.args.ptbxl_root, row["filename_hr"], self.swap)

    @torch.no_grad()
    def encode(self, xs, bs=8):
        out, outs = [], []
        for i in range(0, len(xs), bs):
            v, _ = self.branch.encode(torch.tensor(np.asarray(xs[i:i + bs]), dtype=torch.float32, device=self.dev))
            out.append(v)
        return torch.cat(out)

    def score_beats(self, x, peaks, cond, obs_ids, methods=("occlusion", "chefer", "rollout", "energy")):
        """{method: {obs_index: (n_beats,)}} plus the full-ECG target per observation."""
        b = self.bank.get(cond)
        xs = [x] + list(occlude_beats(x, peaks, self.pre, self.post, self.args.fill))
        vec = self.encode(xs)
        res = {m: {} for m in methods}
        full = {}
        with torch.no_grad():
            vt, eo = self.branch.encode(torch.tensor(x, dtype=torch.float32, device=self.dev).unsqueeze(0))
        roll = None
        for i in obs_ids:
            t = Z.target_scalar(vec, b["pv"][i], b["nv"][i], self.args.target).float().cpu().numpy()
            full[i] = float(t[0])
            if "occlusion" in res:
                res["occlusion"][i] = t[0] - t[1:]
            if "chefer" in res or ("rollout" in res and roll is None):
                mp, _ = Z.explain_pair(self.branch, self.model, x, eo, b["pv"][i], b["nv"][i], b["ps"][i],
                                       ["chefer", "rollout"], self.args.target)
                if "chefer" in res:
                    res["chefer"][i] = beat_mean(mp["chefer"], peaks, self.pre, self.post, self.spf)
                if "rollout" in res:
                    res["rollout"][i] = beat_mean(mp["rollout"], peaks, self.pre, self.post, self.spf)
                    roll = res["rollout"][i]
            elif "rollout" in res:
                res["rollout"][i] = roll
        if "energy" in res:
            e = beat_energy(x, peaks, self.pre, self.post)
            for i in obs_ids:
                res["energy"][i] = e
        return res, full


def _add_common(p):
    p.add_argument("--split", default="sub_class", choices=["sub_class", "super_class", "form", "rhythm"])
    p.add_argument("--csv", default=None)
    p.add_argument("--folds", type=int, nargs="+", default=[10])
    p.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    p.add_argument("--config", default="configs/config.json")
    p.add_argument("--observations", default="configs/observations.json")
    p.add_argument("--repo", default=os.getenv("ZETA_REPO") or os.getenv("ZETA_REPO_ROOT"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lead-swap", choices=["on", "off"], default="on")
    p.add_argument("--target", default="logit_diff", choices=["logit_diff", "pos_sim"])
    p.add_argument("--obs-filter", default=None, help="obs_auroc.csv: keep observations with pair AUROC >= --min-obs-auroc")
    p.add_argument("--min-obs-auroc", type=float, default=0.75)
    p.add_argument("--pre", type=float, default=0.20, help="seconds before the beat peak removed/transplanted")
    p.add_argument("--post", type=float, default=0.40, help="seconds after the beat peak")
    p.add_argument("--fill", choices=["interp", "median"], default="interp")
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)


# ─────────────────────────────────────────────────────────────────────────────
# chimera
# ─────────────────────────────────────────────────────────────────────────────

def cmd_chimera(args):
    C = Ctx(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s_i, s_n = (int(v) for v in args.shard.split("/"))
    df = C.df
    if "NORM" not in df.columns:
        raise SystemExit("no NORM column in the split CSV: hosts are NORM records")
    conds = [c for c in (args.conditions or C.label_cols) if c in C.label_cols and c != "NORM" and c not in ("SR",)]
    if C.allowed is not None:
        conds = [c for c in conds if C.allowed.get(c)]
    print(f"donor conditions: {conds}; k = {args.k}; trials per condition {args.trials}")
    hosts = np.flatnonzero((df["NORM"].values == 1))
    f = open(out / f"chimera_shard{s_i}of{s_n}.jsonl", "a")
    n_rows, t0 = 0, time.time()
    trial_id = 0
    for cond in conds:
        ids = C.obs_ids(cond)
        # --donor-norm: control. Donors are NORMAL records, so transplanted beats carry no condition feature;
        # whatever a method finds then is splice artefact / atypicality, not the observation.
        didx = np.flatnonzero(df["NORM" if args.donor_norm else cond].values == 1)
        scored = []
        for i in C.rng.permutation(didx)[: max(args.trials * 3, args.trials)]:      # confident donors
            try:
                x_ = C.read(df.loc[i])
                with torch.no_grad():
                    v_, _ = C.branch.encode(torch.tensor(x_, dtype=torch.float32, device=C.dev).unsqueeze(0))
                pp = B.pair_probs(v_, C.bank.get(cond))
                scored.append((float(np.mean([pp[k] for k in ids])), int(i)))
            except Exception as e:
                print(f"  donor {i}: {e!r}")
        scored.sort(reverse=True)
        if args.donor_norm:
            scored = [(0.0, i) for _, i in scored]        # any normal record; no confidence ranking
        donors = [i for _, i in scored[: args.trials]]
        for t_i, d_i in enumerate(donors):
            trial_id += 1
            if (trial_id - 1) % s_n != s_i:
                continue
            if args.max_rows and n_rows >= args.max_rows:
                break
            try:
                donor = C.read(df.loc[d_i])
                dpk = valid_peaks(donor, C.pre, C.post)
                for _attempt in range(10):
                    h_i = int(C.rng.choice(hosts))
                    if h_i == d_i:
                        continue
                    host = C.read(df.loc[h_i])
                    hpk = valid_peaks(host, C.pre, C.post)
                    if len(hpk) >= 6 and len(dpk) >= max(args.k):
                        break
                else:
                    continue
                for k in args.k:
                    # k host positions, pairwise >= pre+post apart, and k donor beats
                    for _ in range(30):
                        hs = np.sort(C.rng.choice(len(hpk), k, replace=False))
                        if k == 1 or np.all(np.diff(hpk[hs]) >= C.pre + C.post):
                            break
                    ds = C.rng.choice(len(dpk), k, replace=False)
                    chim = transplant(host, donor, hpk[hs], dpk[ds], C.pre, C.post)
                    labels = np.zeros(len(hpk), bool)
                    labels[hs] = True
                    sc_c, full_c = C.score_beats(chim, hpk, cond, ids)
                    sc_h, full_h = C.score_beats(host, hpk, cond, ids)
                    rng = np.random.RandomState(args.seed + trial_id)
                    rows = []
                    for i in list(ids) + [-1]:
                        r = {"trial": trial_id, "condition": cond, "host": int(df.loc[h_i, "ecg_id"]),
                             "donor": int(df.loc[d_i, "ecg_id"]), "k": k, "n_beats": len(hpk),
                             "obs_index": i, "transplanted": [int(v) for v in hs]}
                        if i >= 0:
                            r["obs_text"] = C.bank.get(cond)["pos_t"][i]
                            r["delta_target"] = full_c[i] - full_h[i]
                        else:
                            r["obs_text"] = "(mean over observations)"
                            r["delta_target"] = float(np.mean([full_c[j] - full_h[j] for j in ids]))
                        for m in ("occlusion", "chefer", "rollout", "energy"):
                            pick = (lambda s: s[m][i]) if i >= 0 else (lambda s: np.mean([s[m][j] for j in ids], 0))
                            r[f"{m}_auroc"] = auroc(pick(sc_c), labels)
                            r[f"{m}_auroc_null"] = auroc(pick(sc_h), labels)
                            r[f"{m}_top1"] = top1_hit(pick(sc_c), labels)
                            r[f"{m}_top1_null"] = top1_hit(pick(sc_h), labels)
                        rs = rng.rand(len(hpk))
                        r["random_auroc"], r["random_top1"] = auroc(rs, labels), top1_hit(rs, labels)
                        rows.append(r)
                    for r in rows:
                        f.write(json.dumps(B.clean(r)) + "\n")
                    f.flush()
                    n_rows += len(rows)
            except Exception as e:
                print(f"  trial {trial_id} ({cond}): {e!r}")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(f"[{cond}] trials so far {trial_id}, rows {n_rows}, {(time.time() - t0) / 60:.1f} min", flush=True)
    f.close()
    print(f"done: {n_rows} rows -> {out}")
    if s_n == 1:
        summarise_chimera(out)


def summarise_chimera(d):
    import pandas as pd
    d = Path(d)
    rows = []
    for fn in sorted(glob.glob(str(d / "chimera*.jsonl"))):
        rows += [json.loads(l) for l in open(fn) if l.strip()]
    if not rows:
        raise SystemExit(f"no chimera*.jsonl in {d}")
    df = pd.DataFrame(rows)
    ms = ["occlusion", "chefer", "rollout", "energy"]
    for m in ms:
        df[f"{m}_gain"] = df[f"{m}_auroc"] - df[f"{m}_auroc_null"]
    df["responsive"] = df["delta_target"] > 0
    out = []

    def block(g, label):
        r = {**label, "n_trials": int(g["trial"].nunique()), "n_rows": len(g),
             "frac_responsive": float(g["responsive"].mean()), "mean_delta_target": float(g["delta_target"].mean()),
             "chance_top1": float((g["k"] / g["n_beats"]).mean())}
        for m in ms + ["random"]:
            for col, nm in ((f"{m}_auroc", "auroc"),) + (((f"{m}_auroc_null", "auroc_null"), (f"{m}_gain", "gain"),
                                                          (f"{m}_top1", "top1"), (f"{m}_top1_null", "top1_null"))
                                                         if m != "random" else ()):
                v = g[col].dropna().values
                if len(v):
                    lo, hi = B.boot_ci(v)
                    r[f"{m}_{nm}"], r[f"{m}_{nm}_lo"], r[f"{m}_{nm}_hi"] = float(v.mean()), lo, hi
        return r

    g_all = df[df["obs_index"] == -1]
    for k, g in g_all.groupby("k"):
        out.append(block(g, {"scope": "all conditions (mean over observations)", "k": k}))
        out.append(block(g[g["responsive"]], {"scope": "all conditions, responsive trials only", "k": k}))
    for (c, k), g in g_all.groupby(["condition", "k"]):
        out.append(block(g, {"scope": f"{c} (mean over observations)", "k": k}))
    for (c, i, k), g in df[df["obs_index"] >= 0].groupby(["condition", "obs_index", "k"]):
        out.append(block(g, {"scope": f"{c} obs {i}: {g['obs_text'].iloc[0][:48]}", "k": k}))
    T = pd.DataFrame(out)
    T.to_csv(d / "chimera_table.csv", index=False)
    show = ["scope", "k", "n_trials", "frac_responsive", "mean_delta_target"] + [
        f"{m}_{s}" for m in ms for s in ("auroc", "auroc_null")] + ["random_auroc"]
    with pd.option_context("display.width", 280, "display.max_columns", None, "display.max_colwidth", 52,
                           "display.float_format", "{:.2f}".format):
        print("AUROC for picking out the transplanted beats (0.5 = chance); *_null = same beats, unmodified host")
        print(T[[c for c in show if c in T]].head(24).to_string(index=False))
    print("\nGain over null (chimera minus host, mean [95% CI]) -- all conditions, all trials:")
    for k, g in g_all.groupby("k"):
        for m in ms:
            v = g[f"{m}_gain"].dropna().values
            lo, hi = B.boot_ci(v)
            print(f"  k={k}  {m:10s} {v.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   top1 {g[f'{m}_top1'].mean():.2f} "
                  f"(null {g[f'{m}_top1_null'].mean():.2f}, chance {(g['k'] / g['n_beats']).mean():.2f})")
    _chimera_fig(g_all, ms, d / "chimera_summary.png")
    print(f"\nwrote {d}/chimera_table.csv and chimera_summary.png")
    return T


def _chimera_fig(g, ms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"occlusion": "#2a78d6", "chefer": "#eb6834", "rollout": "#1baf7a", "energy": "#9b6b3a"}
    ks = sorted(g["k"].unique())
    conds = sorted(g["condition"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks) + 1.5, 0.45 * len(conds) * len(ms) / 2 + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        gk = g[g["k"] == k]
        y = 0
        ticks, labels = [], []
        for c in conds:
            gc = gk[gk["condition"] == c]
            for j, m in enumerate(ms):
                v = gc[f"{m}_gain"].dropna().values
                if len(v) < 2:
                    continue
                lo, hi = B.boot_ci(v)
                ax.hlines(y, lo, hi, color=cols[m], lw=1.6)
                ax.scatter([v.mean()], [y], color=cols[m], s=24, zorder=3, label=m if c == conds[0] else None)
                y -= 1
            ticks.append(y + len(ms) / 2 + 0.5)
            labels.append(c)
            y -= 1
        ax.axvline(0, color="#6b6a66", lw=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"k = {k} transplanted beats", fontsize=9, loc="left")
        ax.set_xlabel("AUROC gain over unmodified host (95% CI)\n> 0 = finds the transplanted beats beyond position/size bias")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# natural
# ─────────────────────────────────────────────────────────────────────────────

def natural_figure(x, peaks, pre, post, ids, texts, occ, chef, path, cond, ecg_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    t = np.arange(x.shape[1]) / FS
    n = len(ids)
    fig, axes = plt.subplots(2 + 1, 1, figsize=(14, 2.2 + 0.45 * n * 2), gridspec_kw={"height_ratios": [2.2, 0.45 * n, 0.45 * n]})
    ax = axes[0]
    ax.plot(t, x[1], color="#0b0b0b", lw=0.8)
    for j, p in enumerate(peaks):
        ax.axvspan((p - pre) / FS, (p + post) / FS, color="#cccccc", alpha=0.25, lw=0)
        ax.text(p / FS, ax.get_ylim()[1], str(j), ha="center", va="top", fontsize=8)
    ax.set_title(f"ECG {ecg_id} | {cond} | lead II, beats numbered", fontsize=10, loc="left")
    ax.set_xlim(0, x.shape[1] / FS)
    ax.set_yticks([])
    for a, M, ttl, cmap in ((axes[1], np.array([occ[i] for i in ids]), "occlusion drop (target falls when beat removed)", "magma"),
                            (axes[2], np.array([chef[i] for i in ids]), "Chefer beat mean", "viridis")):
        im = a.imshow(M, aspect="auto", cmap=cmap)
        a.set_xticks(range(len(peaks)))
        a.set_yticks(range(n))
        a.set_yticklabels([textwrap.shorten(texts[i], 40, placeholder="...") for i in ids], fontsize=7)
        a.set_title(ttl, fontsize=9, loc="left")
        for r in range(n):
            for c_ in range(len(peaks)):
                a.text(c_, r, f"{M[r, c_]:.2f}", ha="center", va="center", fontsize=6, color="white")
    axes[2].set_xlabel("beat number")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_natural(args):
    import pandas as pd
    from scipy.stats import spearmanr
    C = Ctx(args)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    s_i, s_n = (int(v) for v in args.shard.split("/"))
    df = C.df
    conds = [c for c in (args.conditions or C.label_cols) if c in C.label_cols and c not in ("NORM", "SR")]
    if C.allowed is not None:
        conds = [c for c in conds if C.allowed.get(c)]
    jobs = []
    for cond in conds:
        ids = C.obs_ids(cond)
        idx = C.rng.permutation(np.flatnonzero(df[cond].values == 1))[: args.candidates]
        scored = []
        for i in idx:
            try:
                x_ = C.read(df.loc[i])
                with torch.no_grad():
                    v_, _ = C.branch.encode(torch.tensor(x_, dtype=torch.float32, device=C.dev).unsqueeze(0))
                pp = B.pair_probs(v_, C.bank.get(cond))
                scored.append((float(np.mean([pp[k] for k in ids])), int(i)))
            except Exception as e:
                print(f"  candidate {i}: {e!r}")
        scored.sort(reverse=True)
        jobs += [(cond, i) for _, i in scored[: args.per_condition]]
    jobs = jobs[s_i::s_n]
    print(f"{len(jobs)} (condition, ECG) pairs")
    f = open(out / f"natural_shard{s_i}of{s_n}.jsonl", "a")
    figs = {}
    n_rows = 0
    for cond, i in jobs:
        if args.max_rows and n_rows >= args.max_rows:
            break
        try:
            x = C.read(df.loc[i])
            pk = valid_peaks(x, C.pre, C.post)
            if len(pk) < 5:
                continue
            ids = C.obs_ids(cond)
            sc, full = C.score_beats(x, pk, cond, ids)
            texts = {j: C.bank.get(cond)["pos_t"][j] for j in ids}
            for j in ids:
                o, ch, ro, en = sc["occlusion"][j], sc["chefer"][j], sc["rollout"][j], sc["energy"][j]
                sp = lambda a, b: float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else None
                f.write(json.dumps(B.clean({
                    "ecg_id": int(df.loc[i, "ecg_id"]), "condition": cond, "obs_index": j, "obs_text": texts[j],
                    "n_beats": len(pk), "target_full": full[j], "drops": o.tolist(), "chefer": ch.tolist(),
                    "cv_drop": float(np.std(o) / (abs(np.mean(o)) + 1e-9)), "max_over_median": float(np.max(o) / (abs(np.median(o)) + 1e-9)),
                    "top_beat": int(np.argmax(o)), "share_top": float(np.max(np.clip(o, 0, None)) / (np.clip(o, 0, None).sum() + 1e-12)),
                    "rho_energy": sp(o, en), "rho_chefer": sp(o, ch), "rho_rollout": sp(o, ro)})) + "\n")
                n_rows += 1
            f.flush()
            if figs.get(cond, 0) < args.figures:
                safe = cond.replace("/", "-")
                natural_figure(x, pk, C.pre, C.post, ids, texts, sc["occlusion"], sc["chefer"],
                               out / "figures" / f"beats_{safe}_{int(df.loc[i, 'ecg_id'])}.png", cond, int(df.loc[i, "ecg_id"]))
                figs[cond] = figs.get(cond, 0) + 1
        except Exception as e:
            print(f"  ECG {i} ({cond}): {e!r}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    f.close()
    print(f"done: {n_rows} rows -> {out}")
    if s_n == 1:
        summarise_natural(out)


def summarise_natural(d):
    import pandas as pd
    d = Path(d)
    rows = []
    for fn in sorted(glob.glob(str(d / "natural*.jsonl"))):
        rows += [json.loads(l) for l in open(fn) if l.strip()]
    df = pd.DataFrame(rows)
    g = df.groupby(["condition", "obs_index", "obs_text"]).agg(
        n_ecg=("ecg_id", "nunique"), cv_drop=("cv_drop", "median"), share_top=("share_top", "median"),
        max_over_median=("max_over_median", "median"), rho_energy=("rho_energy", "median"),
        rho_chefer=("rho_chefer", "median"), rho_rollout=("rho_rollout", "median")).reset_index()
    g.to_csv(d / "natural_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 44,
                           "display.float_format", "{:.2f}".format):
        print("Per-beat occlusion on real records (medians over ECGs). cv_drop = spread of drops across beats; "
              "share_top = share of the total drop carried by the single most important beat (1/n_beats if all beats equal)")
        print(g.to_string(index=False))
    print(f"\nwrote {d}/natural_table.csv")
    return g


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chimera", help="transplant donor beats into a NORM host; do methods find them?")
    _add_common(c)
    c.add_argument("--trials", type=int, default=40, help="donor records per condition")
    c.add_argument("--k", type=int, nargs="+", default=[1, 3], help="beats transplanted per trial")
    c.add_argument("--donor-norm", action="store_true",
                   help="CONTROL: donor beats come from NORM records (no condition feature); methods that still "
                        "'find' them are detecting the splice, not the observation")
    n = sub.add_parser("natural", help="per-beat occlusion on real labelled records (no ground truth)")
    _add_common(n)
    n.add_argument("--per-condition", type=int, default=20)
    n.add_argument("--candidates", type=int, default=60)
    n.add_argument("--figures", type=int, default=3)
    s = sub.add_parser("summary", help="rebuild tables/figures from a finished run")
    s.add_argument("--dir", required=True)
    s.add_argument("--kind", choices=["chimera", "natural"], default="chimera")
    return p


def main(argv=None):
    from dotenv import find_dotenv, load_dotenv
    load_dotenv()
    load_dotenv(find_dotenv(usecwd=True))
    a = build_parser().parse_args(argv)
    if a.cmd == "chimera":
        cmd_chimera(a)
    elif a.cmd == "natural":
        cmd_natural(a)
    else:
        (summarise_chimera if a.kind == "chimera" else summarise_natural)(a.dir)


if __name__ == "__main__":
    main()