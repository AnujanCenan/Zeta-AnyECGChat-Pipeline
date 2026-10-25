"""
zeta_demo.py -- the end-to-end output for a PTB-XL ECG: prediction, cited observations, beat-level evidence.

    python zeta_demo.py --ecg-ids 1275 2409 --obs-filter obs_auroc.csv --out demo
    python zeta_demo.py --condition CLBBB --n 3 --obs-filter obs_auroc.csv --out demo

For each ECG it writes <out>/ecg_<id>.png, ecg_<id>.md and ecg_<id>.json:
  * ZETA's top conditions (zero-shot scores) and, for the chosen condition, the observations it cites
    (positive sentence, pair score);
  * per-beat evidence for those observations:
      - atypical-beat map (attention rollout over each beat; the same for every observation): which beats the
        model attends to as unusual;
      - per-beat occlusion (a measured effect, observation-specific): how much each observation's score drops
        when that beat is removed;
  * a generated, template-only sentence per observation naming the beats whose removal lowers it most.

What the figure does and does not say (from the chimera and per-observation tests): rollout and Chefer flag
atypical beats but are NOT observation-specific; occlusion shows modest observation dependence for
bundle-branch patterns only; nothing localises WITHIN a beat; none of it is validated against clinical
annotation. The figure states this on its face.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch

import zeta_relevance as Z
import zeta_relevance_batch as B
import zeta_beat_occlusion as BO

FS = Z.SAMPLE_RATE
CAVEAT = ("Beat-level evidence only. Rollout shows relative attention across beats (same for every observation; in the chimera test it ranked transplanted abnormal beats highly); "
          "occlusion shows how much each observation's score falls when a beat is removed. Neither localises within a beat, "
          "and neither is validated against clinical annotation.")


def beat_sentence(text, score, drops, peaks, top_n=2, min_drop=0.05):
    order = np.argsort(-drops)[:top_n]
    order = [j for j in order if drops[j] >= min_drop]
    if not order:
        return (f'"{text}" (score {score:.2f}): no single beat accounts for it; removing any one beat changes its target '
                f'by under {min_drop:.2f} (largest drop {float(np.max(drops)):.2f}).')
    parts = ", ".join(f"beat {j + 1} at {peaks[j] / FS:.1f} s (drop {drops[j]:.2f})" for j in order)
    tot = np.clip(drops, 0, None).sum()
    share = float(np.clip(drops[order[0]], 0, None) / tot) if tot > 0 else 0.0
    return (f'"{text}" (score {score:.2f}): removing {parts} lowers it most; '
            f'the top beat carries {share:.0%} of the total drop (equal share would be {1 / len(peaks):.0%}).')


def make_figure(x, peaks, pre, post, rollout, occ, texts, scores, title, path, lead_names, z_flag=1.5, vmax_min=0.15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(x.shape[1]) / FS
    ids = list(occ)
    n = len(ids)
    fig = plt.figure(figsize=(14, 6.2 + 0.5 * n))
    gs = fig.add_gridspec(4 + 1, 1, height_ratios=[1.3, 1.0, 1.0, 0.35 + 0.42 * n, 0.9], hspace=0.6)
    r = np.asarray(rollout, float)
    rz = (r - r.mean()) / (r.std() + 1e-9)
    ax0 = fig.add_subplot(gs[0])
    for lead in (1,):
        ax0.plot(t, x[lead], color="#0b0b0b", lw=0.8)
    for j, p in enumerate(peaks):
        ax0.axvspan((p - pre) / FS, (p + post) / FS, color="#eb6834", alpha=float(np.clip(0.08 + 0.18 * max(rz[j], 0), 0, 0.6)), lw=0)
        ax0.text(p / FS, ax0.get_ylim()[1], str(j + 1), ha="center", va="top", fontsize=8)
    ax0.set_xlim(0, t[-1])
    ax0.set_yticks([])
    ax0.set_ylabel(f"lead {lead_names[1]}", rotation=0, ha="right", va="center")
    ax0.set_title(title, fontsize=10, loc="left")
    ax0.set_xticks([])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t, x[6], color="#0b0b0b", lw=0.8)
    for j, p in enumerate(peaks):
        ax1.axvspan((p - pre) / FS, (p + post) / FS, color="#eb6834", alpha=float(np.clip(0.08 + 0.18 * max(rz[j], 0), 0, 0.6)), lw=0)
    ax1.set_yticks([])
    ax1.set_ylabel(f"lead {lead_names[6]}", rotation=0, ha="right", va="center")
    ax1.set_xlabel("time (s); shading = rollout attention per beat (darker = more attention)", fontsize=8)
    axb = fig.add_subplot(gs[2])
    axb.bar(np.arange(1, len(peaks) + 1), rz, color=["#eb6834" if v > z_flag else "#b3b2ae" for v in rz])
    axb.axhline(z_flag, color="#6b6a66", lw=0.8, ls="--")
    axb.set_ylabel("rollout\n(z-score)", fontsize=8)
    axb.set_xticks(range(1, len(peaks) + 1))
    axb.tick_params(labelsize=7)
    axb.set_title(f"relative attention across beats (rollout z-score); orange = above the dashed line (z > {z_flag}); a flat map means no beat stands out", fontsize=8, loc="left")
    axm = fig.add_subplot(gs[3])
    M = np.array([occ[i] for i in ids])
    im = axm.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(vmax_min, float(M.max())))
    axm.set_xticks(range(len(peaks)))
    axm.set_xticklabels(range(1, len(peaks) + 1))
    axm.set_yticks(range(n))
    axm.set_yticklabels([textwrap.shorten(texts[i], 46, placeholder="...") + f"  ({scores[i]:.2f})" for i in ids], fontsize=7)
    axm.set_title("occlusion: drop in each observation's target when that beat is removed (shared scale, 0 to at least 0.15: dark = negligible)", fontsize=8, loc="left")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            axm.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6, color="white")
    axt = fig.add_subplot(gs[4])
    axt.axis("off")
    axt.text(0, 1, "\n".join(textwrap.wrap(CAVEAT, 170)), va="top", fontsize=7.5, color="#52514e")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_one(C, ecg_row, args, out):
    ecg_id = int(ecg_row["ecg_id"])
    x = C.read(ecg_row)
    with torch.no_grad():
        vec, _ = C.branch.encode(torch.tensor(x, dtype=torch.float32, device=C.dev).unsqueeze(0))
    cond_scores = {}
    for c in C.obs:
        if c in args.skip:
            continue
        cond_scores[c] = float(np.mean(B.pair_probs(vec, C.bank.get(c))))
    ranked = sorted(cond_scores.items(), key=lambda kv: -kv[1])
    cond = args.condition or ranked[0][0]
    b = C.bank.get(cond)
    probs = B.pair_probs(vec, b)
    ids_all = C.obs_ids(cond)
    ids = sorted(ids_all, key=lambda i: -probs[i])[: args.n_obs]
    pk = BO.valid_peaks(x, C.pre, C.post)
    if len(pk) < 3 or not ids:
        print(f"ECG {ecg_id}: fewer than 3 usable beats or no observations; skipped")
        return None
    sc, full = C.score_beats(x, pk, cond, ids)
    texts = {i: b["pos_t"][i] for i in ids}
    scores = {i: probs[i] for i in ids}
    roll = sc["rollout"][ids[0]]
    rz = (roll - roll.mean()) / (roll.std() + 1e-9)
    flagged = [int(j + 1) for j in np.flatnonzero(rz > args.z_flag)]
    sentences = [beat_sentence(texts[i], scores[i], sc["occlusion"][i], pk, min_drop=args.min_drop) for i in ids]
    lead_names = Z.LEAD_NAMES_ZETA if C.swap else Z.LEAD_NAMES
    labels = [c for c in C.label_cols if int(ecg_row.get(c, 0)) == 1] if hasattr(ecg_row, "get") else []
    title = (f"ECG {ecg_id} | ZETA's top conditions: " + ", ".join(f"{c} {s:.2f}" for c, s in ranked[:3])
             + f" | explained: {cond}" + (f" | dataset labels: {', '.join(labels)}" if labels else ""))
    png = out / f"ecg_{ecg_id}.png"
    make_figure(x, pk, C.pre, C.post, roll, {i: sc["occlusion"][i] for i in ids}, texts, scores, title, png, lead_names, args.z_flag)
    md = [f"# ECG {ecg_id}", "",
          "**Prediction (ZETA zero-shot):** " + ", ".join(f"{c} ({s:.2f})" for c, s in ranked[:3]),
          f"**Explained condition:** {cond}" + (f"  \n**Dataset labels:** {', '.join(labels)}" if labels else ""), "",
          "## Cited observations and the beats behind them", ""] + [f"- {s}" for s in sentences] + [
          "", f"**Beats with the highest attention (rollout z > {args.z_flag}):** "
          + (", ".join(f"beat {j} ({pk[j - 1] / FS:.1f} s)" for j in flagged) if flagged else "none stand out (attention is spread evenly across beats)"), "",
          f"**Detected beats:** {len(pk)}", "", f"_{CAVEAT}_", "", f"![figure](ecg_{ecg_id}.png)"]
    (out / f"ecg_{ecg_id}.md").write_text("\n".join(md))
    (out / f"ecg_{ecg_id}.json").write_text(json.dumps(B.clean({
        "ecg_id": ecg_id, "condition_scores": cond_scores, "explained": cond, "labels": labels,
        "beats_s": [float(p / FS) for p in pk], "rollout_z": rz.tolist(), "flagged_beats": flagged,
        "observations": [{"index": i, "text": texts[i], "score": scores[i], "occlusion_drops": sc["occlusion"][i].tolist(),
                          "chefer_beat_means": sc["chefer"][i].tolist(), "sentence": s}
                         for i, s in zip(ids, sentences)]}), indent=1))
    print(f"ECG {ecg_id}: {cond}; flagged beats {flagged}; wrote {png}")
    return ecg_id


def main(argv=None):
    from dotenv import find_dotenv, load_dotenv
    load_dotenv()
    load_dotenv(find_dotenv(usecwd=True))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    BO._add_common(p)
    p.add_argument("--ecg-ids", type=int, nargs="*", default=None)
    p.add_argument("--condition", default=None, help="explain this condition (default: ZETA's top one per ECG); with --n and no --ecg-ids, picks the most confident labelled records")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--n-obs", type=int, default=3, help="observations shown per ECG")
    p.add_argument("--skip", nargs="*", default=["NORM", "SR"])
    p.add_argument("--z-flag", type=float, default=1.5, help="rollout z-score above which a beat is called out")
    p.add_argument("--min-drop", type=float, default=0.05, help="smallest occlusion drop (target logit units) worth naming a beat for")
    args = p.parse_args(argv)
    C = BO.Ctx(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = C.df
    if args.ecg_ids:
        rows = [df[df["ecg_id"] == e].iloc[0] for e in args.ecg_ids if (df["ecg_id"] == e).any()]
    else:
        cond = args.condition
        if not cond:
            raise SystemExit("give --ecg-ids, or --condition with --n")
        idx = C.rng.permutation(np.flatnonzero(df[cond].values == 1))[: max(args.n * 6, 12)]
        scored = []
        for i in idx:
            x_ = C.read(df.loc[i])
            with torch.no_grad():
                v_, _ = C.branch.encode(torch.tensor(x_, dtype=torch.float32, device=C.dev).unsqueeze(0))
            pp = B.pair_probs(v_, C.bank.get(cond))
            scored.append((float(np.mean([pp[k] for k in C.obs_ids(cond)])), int(i)))
        scored.sort(reverse=True)
        rows = [df.loc[i] for _, i in scored[: args.n]]
    done = [run_one(C, r, args, out) for r in rows]
    print(f"{sum(d is not None for d in done)} demo(s) in {out}")


if __name__ == "__main__":
    main()