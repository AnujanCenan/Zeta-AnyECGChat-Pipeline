"""
zeta_relevance_batch.py

Run the ZETA localisation methods over many PTB-XL records and compare them.

    python zeta_relevance_batch.py run       --split sub_class --out results/ [--shard 0/8]
    python zeta_relevance_batch.py aggregate --dir results/
    python zeta_relevance_batch.py check-swap --split sub_class      # aVL/aVF swap on vs off, ZETA AUROC

`check-swap` computes ZETA's own per-condition zero-shot AUROC on a sample of the test fold with the
aVL<->aVF swap on and off (no relevance maps, forward passes only) so you can pick --lead-swap with data.
'off' is also what ZETA's own PTB-XL evaluation would amount to if its swap line acts on time samples.

What `run` does, per (ECG, condition):
  * picks the CITED observation (ZETA's top-scoring positive pair for that condition) and computes the
    maps from zeta_relevance.py: chefer, rollout, attn_last, attn_mean, gradcam_cnn, cross_attn_fixed;
  * scores each map by deletion / insertion on the model's own target (no ground truth needed), always
    against a random-order control on the SAME record, so numbers are comparable across records;
  * records a beat-locking PROXY (AUROC of the map vs. detected QRS windows; NOT ground truth);
  * with >=2 positive observations, computes how much each method's map changes between observations
    (target dependence, "xobs");
  * stores peakiness, top-15% mass, and the Spearman agreement between every pair of methods;
  * optionally (--sanity-every K) the cascading weight-randomisation check on every K-th row.
Results go to one JSON line per row (resumable, shardable), maps to small .npz files.

Which ECGs / conditions
  --condition-source label     (default) use the labels in ZETA's own PTB-XL split CSVs
                               (<repo>/data_split/ptbxl/<split>/ptbxl_<split>_test.csv): every record is
                               explained for each true condition it carries, capped at --per-condition.
  --condition-source zeta-top  no labels: explain whatever condition ZETA scores highest. Rows then mix
                               right and wrong calls; use `aggregate --min-cond-score` to restrict.

Cost: each row needs one forward/backward per observation plus (n_methods + n_random) x (n_steps + 1) x 2
forward passes for the deletion/insertion curves. NOT timed on GPU yet; time the first ~20 rows
(--max-rows 20) before launching the full job.

Example PBS array job (adapt the resource line to Katana):
    #PBS -J 0-7
    #PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
    cd $PBS_O_WORKDIR; source venv/bin/activate
    python zeta_relevance_batch.py run --split sub_class --per-condition 50 \
        --shard ${PBS_ARRAY_INDEX}/8 --out results/
  then, once all shards finish:  python zeta_relevance_batch.py aggregate --dir results/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import zeta_relevance as Z

METHODS = Z.ALL_METHODS
# validated categorical slots 1-6 of the reference palette, fixed order (blue, orange, aqua, yellow, magenta, green)
METHOD_COLOURS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def clean(o):
    """NaN/inf -> None so the JSONL is strict JSON; numpy scalars -> python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

class TextBank:
    """Encodes each condition's observation sentences once (frozen model -> reusable across ECGs)."""

    def __init__(self, model, tok, obs, dev):
        self.model, self.tok, self.obs, self.dev, self.cache = model, tok, obs, dev, {}

    def get(self, cond):
        if cond not in self.cache:
            P = [t.lower() for t in self.obs[cond]["P"]]
            N = [t.lower() for t in self.obs[cond]["N"]]
            n = min(len(P), len(N))
            pv, ps = zip(*[Z.encode_text(self.model, self.tok, t, self.dev) for t in P[:n]])
            nv, _ = zip(*[Z.encode_text(self.model, self.tok, t, self.dev) for t in N[:n]])
            self.cache[cond] = {"pos_t": P[:n], "neg_t": N[:n], "pv": pv, "nv": nv, "ps": ps}
        return self.cache[cond]


def pair_probs(vec, bank):
    return [float(torch.sigmoid(Z.target_scalar(vec, bank["pv"][i], bank["nv"][i])[0]))
            for i in range(len(bank["pv"]))]


def row_metrics(maps, ev, qrs_mask, xsim):
    """Per-method scalars for one row."""
    rnd = ev["random"]
    out = {}
    for m, rel in maps.items():
        e = ev[m]
        r = np.clip(np.asarray(rel, float), 0, None)
        tot = r.sum()
        d = {"del_auc": e["del_auc"], "ins_auc": e["ins_auc"],
             "del_adv": rnd["del_auc"] - e["del_auc"],   # >0: removing its top frames hurts MORE than random
             "ins_adv": e["ins_auc"] - rnd["ins_auc"],   # >0: restoring its top frames helps MORE than random
             "peak_ratio": float(r.max() / (r.mean() + 1e-12)),
             "top15_mass": float(np.sort(r)[::-1][: int(0.15 * len(r))].sum() / tot) if tot > 0 else None}
        if qrs_mask is not None:
            lm = Z.localisation_metrics(rel, qrs_mask)
            d["qrs_auroc"], d["qrs_mass"] = lm["auroc"], lm["mass_in_gt"]
        if xsim is not None:
            d["xobs_mean"], d["xobs_min"] = xsim[m]["mean_offdiag"], xsim[m]["min_offdiag"]
        out[m] = d
    return out


def method_agreement(maps):
    from scipy.stats import spearmanr
    names = list(maps)
    out = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = maps[names[a]], maps[names[b]]
            out[f"{names[a]}|{names[b]}"] = (spearmanr(x, y).correlation
                                             if np.std(x) > 0 and np.std(y) > 0 else float("nan"))
    return out


def load_done(out_dir):
    done = set()
    for f in glob.glob(str(out_dir / "results*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((int(r["ecg_id"]), r["condition"]))
                except Exception:
                    pass
    return done


def cmd_run(args, parser):
    import pandas as pd

    Z.resolve_repo(parser, args)
    from transformers import T5TokenizerFast

    out_dir = Path(args.out)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    s_i, s_n = (int(v) for v in args.shard.split("/"))
    tag = f"shard{s_i}of{s_n}"

    csv = args.csv or str(Path(args.repo).expanduser() / "data_split" / "ptbxl" / args.split /
                          f"ptbxl_{args.split}_test.csv")
    df = pd.read_csv(csv)
    df = df[df["strat_fold"].isin(args.folds)].reset_index(drop=True)
    with open(args.observations) as f:
        obs = json.load(f)
    label_cols = [c for c in df.columns[6:] if c in obs]
    conds = [c for c in (args.conditions or label_cols) if c in label_cols and c not in args.skip]
    print(f"{len(df)} records in {csv} (folds {args.folds}); conditions: {len(conds)} -> {conds}")

    rng = np.random.RandomState(args.seed)
    jobs = {}
    if args.condition_source == "label":
        for c in conds:
            idx = np.flatnonzero(df[c].values == 1)
            if args.per_condition and len(idx) > args.per_condition:
                idx = rng.choice(idx, args.per_condition, replace=False)
            for i in idx:
                jobs.setdefault(int(df.loc[i, "ecg_id"]), []).append(c)
        print(f"{sum(len(v) for v in jobs.values())} (ECG, condition) rows over {len(jobs)} ECGs")
    else:
        pick = rng.choice(len(df), min(args.n_ecgs, len(df)), replace=False)
        jobs = {int(df.loc[i, "ecg_id"]): None for i in pick}
        print(f"{len(jobs)} ECGs; condition = ZETA's top-scoring")
    by_id = df.set_index("ecg_id")
    items = sorted(jobs.items())[s_i::s_n]
    done = load_done(out_dir)

    dev = torch.device(args.device)
    model = Z.load_model(args.config, args.checkpoint, dev)
    tok = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)
    branch = Z.ECGBranch.from_model(model)
    spf, nf_fn = Z.frame_geometry(branch)
    bank = TextBank(model, tok, obs, dev)
    if args.condition_source == "zeta-top":
        all_conds = [c for c in obs if c not in args.skip]
        print(f"encoding observation text for {len(all_conds)} conditions ...")
        for c in all_conds:
            bank.get(c)

    f_res = open(out_dir / f"results_{tag}.jsonl", "a")
    f_san = open(out_dir / f"sanity_{tag}.jsonl", "a")
    f_err = open(out_dir / f"errors_{tag}.jsonl", "a")
    n_rows, t0 = 0, time.time()
    for k, (ecg_id, want) in enumerate(items):
        if args.max_rows and n_rows >= args.max_rows:
            break
        try:
            x = Z.read_ptbxl(args.ptbxl_root, by_id.loc[ecg_id, "filename_hr"], args.lead_swap == "on")
            with torch.no_grad():
                vec, ecg_out = branch.encode(torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0))
            if want is None:
                sc = {c: float(np.mean(pair_probs(vec, bank.get(c)))) for c in obs if c not in args.skip}
                want = [max(sc, key=sc.get)]
            qrs_mask, n_beats = Z.qrs_frame_mask(x, spf, nf_fn(x.shape[1]))
            for cond in want:
                if (ecg_id, cond) in done:
                    continue
                b = bank.get(cond)
                probs = pair_probs(vec, b)
                cond_score, idx = float(np.mean(probs)), int(np.argmax(probs))
                if cond_score < args.min_cond_score:
                    continue
                obs_ids = list(range(len(probs))) if (not args.no_xobs and len(probs) > 1) else [idx]
                all_maps, logits = {}, {}
                for i in obs_ids:
                    all_maps[i], logits[i] = Z.explain_pair(branch, model, x, ecg_out, b["pv"][i], b["nv"][i],
                                                           b["ps"][i], METHODS, args.target)
                maps = all_maps[idx]
                xsim = Z.cross_observation_similarity(all_maps) if len(obs_ids) > 1 else None
                ev = Z.faithfulness(branch, x, maps, b["pv"][idx], b["nv"][idx], args.target,
                                    args.n_steps, args.n_random, seed=args.seed)
                row = {"ecg_id": ecg_id, "strat_fold": int(by_id.loc[ecg_id, "strat_fold"]),
                       "condition": cond, "cond_score": cond_score, "obs_index": idx,
                       "obs_text": b["pos_t"][idx], "pair_score": probs[idx], "target_value": logits[idx],
                       "n_obs": len(probs), "full": ev["_full"], "blank": ev["_blank"],
                       "range": ev["_full"] - ev["_blank"], "n_beats": n_beats,
                       "qrs_frac": float(qrs_mask.mean()) if qrs_mask is not None else None,
                       "random": {"del_auc": ev["random"]["del_auc"], "ins_auc": ev["random"]["ins_auc"]},
                       "methods": row_metrics(maps, ev, qrs_mask, xsim), "agree": method_agreement(maps),
                       "lead_swap": args.lead_swap}
                f_res.write(json.dumps(clean(row)) + "\n")
                f_res.flush()
                if not args.no_save_maps:
                    safe = cond.replace("/", "-")
                    np.savez_compressed(out_dir / "maps" / f"{ecg_id}_{safe}.npz", spf=spf, obs_index=idx,
                                        **{m: v.astype(np.float16) for m, v in maps.items()})
                n_rows += 1
                if args.sanity_every and n_rows % args.sanity_every == 0:
                    rows = Z.sanity_cascade(branch, x, b["pv"][idx], b["nv"][idx], args.target)
                    f_san.write(json.dumps(clean({"ecg_id": ecg_id, "condition": cond, "rows": rows})) + "\n")
                    f_san.flush()
                if args.max_rows and n_rows >= args.max_rows:
                    break
        except Exception as e:  # keep the batch alive; log and move on
            f_err.write(json.dumps({"ecg_id": ecg_id, "error": repr(e), "trace": traceback.format_exc()[-800:]}) + "\n")
            f_err.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  ECG {ecg_id}: {e!r} (logged)")
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            el = time.time() - t0
            print(f"[{tag}] ECG {k + 1}/{len(items)}  rows written this run: {n_rows}  "
                  f"{el / 60:.1f} min elapsed, ~{el / (k + 1) * (len(items) - k - 1) / 60:.0f} min left", flush=True)
    for fh in (f_res, f_san, f_err):
        fh.close()
    print(f"done: {n_rows} rows -> {out_dir}/results_{tag}.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# aggregate
# ─────────────────────────────────────────────────────────────────────────────

def boot_ci(v, B=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = v[rng.randint(0, len(v), (B, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cmd_aggregate(args, parser):
    import pandas as pd
    from scipy.stats import wilcoxon
    d = Path(args.dir)
    rows = []
    for f in sorted(glob.glob(str(d / "results*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    if not rows:
        parser.error(f"no results*.jsonl in {d}")
    n_all = len(rows)
    rows = [r for r in rows if (r["cond_score"] or 0) >= args.min_cond_score
            and (not args.conditions or r["condition"] in args.conditions)]
    print(f"{n_all} rows read; {len(rows)} after filters (min_cond_score={args.min_cond_score})")

    recs, agree_rows = [], []
    for r in rows:
        rng_ = abs(r["range"]) if r["range"] is not None else float("nan")
        ok = rng_ >= args.min_range
        for m, v in r["methods"].items():
            recs.append({"ecg_id": r["ecg_id"], "condition": r["condition"], "method": m, "range": rng_,
                         "cond_score": r["cond_score"], **{k: (np.nan if x is None else x) for k, x in v.items()},
                         "del_adv_n": v["del_adv"] / rng_ if ok else np.nan,
                         "ins_adv_n": v["ins_adv"] / rng_ if ok else np.nan})
        agree_rows.append({k: (np.nan if x is None else x) for k, x in r["agree"].items()})
    df = pd.DataFrame(recs)
    n_small = int((df.groupby(["ecg_id", "condition"])["range"].first() < args.min_range).sum())
    print(f"{n_small} rows have |full-blank| < {args.min_range}: excluded from the normalised "
          f"advantage columns (their change in target is too small to normalise)")
    cols = ["del_adv", "ins_adv", "del_adv_n", "ins_adv_n", "qrs_auroc", "qrs_mass", "peak_ratio",
            "top15_mass", "xobs_mean", "xobs_min"]
    cols = [c for c in cols if c in df.columns]
    # unit of analysis = ECG (a record may appear under several conditions): average within ECG first
    ecg = df.groupby(["ecg_id", "method"])[cols].mean().reset_index()
    n_ecg = ecg["ecg_id"].nunique()
    print(f"{n_ecg} ECGs, {df.groupby(['ecg_id', 'condition']).ngroups} (ECG, condition) rows\n")

    methods = [m for m in METHODS if m in set(df["method"])]
    summ = []
    for m in methods:
        e = ecg[ecg["method"] == m]
        s = {"method": m, "n_ecg": len(e)}
        for c in ("del_adv_n", "ins_adv_n", "qrs_auroc", "xobs_mean"):
            if c in e:
                lo, hi = boot_ci(e[c])
                s[c], s[c + "_lo"], s[c + "_hi"] = float(e[c].mean()), lo, hi
        s["del_win_rate"] = float((e["del_adv"] > 0).mean())
        s["ins_win_rate"] = float((e["ins_adv"] > 0).mean())
        s["peak_ratio_median"] = float(e["peak_ratio"].median())
        s["top15_mass_mean"] = float(e["top15_mass"].mean())
        summ.append(s)
    S = pd.DataFrame(summ).set_index("method")
    S.to_csv(d / "summary_by_method.csv")
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print("Advantage over random order (normalised by each row's |full - blank|; >0 = better than random)")
        print(S[[c for c in ["n_ecg", "del_adv_n", "del_adv_n_lo", "del_adv_n_hi", "ins_adv_n", "ins_adv_n_lo",
                             "ins_adv_n_hi", "del_win_rate", "ins_win_rate"] if c in S]])
        print("\nBeat-locking proxy, concentration, target dependence")
        print(S[[c for c in ["qrs_auroc", "qrs_auroc_lo", "qrs_auroc_hi", "peak_ratio_median", "top15_mass_mean",
                             "xobs_mean"] if c in S]])

    # paired Wilcoxon between methods, ECG-level
    pw = []
    for c in ("del_adv_n", "ins_adv_n"):
        piv = ecg.pivot(index="ecg_id", columns="method", values=c)[methods].dropna()
        pr, ps = [], []
        for a in range(len(methods)):
            for b in range(a + 1, len(methods)):
                diff = piv[methods[a]] - piv[methods[b]]
                try:
                    p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
                except ValueError:
                    p = float("nan")
                pr.append({"metric": c, "a": methods[a], "b": methods[b], "n": len(diff),
                           "median_a_minus_b": float(diff.median()), "p": p})
                ps.append(p)
        padj = holm(np.nan_to_num(ps, nan=1.0))
        for r_, q in zip(pr, padj):
            r_["p_holm"] = float(q)
        pw += pr
    PW = pd.DataFrame(pw)
    PW.to_csv(d / "pairwise_wilcoxon.csv", index=False)
    print("\nPairwise paired Wilcoxon (ECG level, Holm-adjusted across pairs within each metric) -> pairwise_wilcoxon.csv")

    by_c = df.groupby(["condition", "method"])[[c for c in ("del_adv_n", "ins_adv_n", "qrs_auroc") if c in df]] \
             .agg(["mean", "count"])
    by_c.to_csv(d / "by_condition.csv")

    A = pd.DataFrame(agree_rows).mean()
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for k, v in A.items():
        a, b = k.split("|")
        if a in mat.index and b in mat.index:
            mat.loc[a, b] = mat.loc[b, a] = v
    mat.to_csv(d / "method_agreement.csv")

    san = []
    for f in sorted(glob.glob(str(d / "sanity*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                for st in r["rows"]:
                    san.append({"ecg_id": r["ecg_id"], "condition": r["condition"], **st})
    if san:
        SN = pd.DataFrame(san)
        order = list(dict.fromkeys(SN["stage"]))
        SS = SN.groupby("stage")[[m for m in Z.GRAPH_METHODS if m in SN]].median().loc[order]
        SS.to_csv(d / "sanity_summary.csv")
        with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.2f}".format):
            print(f"\nCascading randomisation, median Spearman vs original map over {SN.groupby(['ecg_id','condition']).ngroups} rows")
            print(SS)

    make_figure(S, mat, methods, d / "summary.png", n_ecg)
    print(f"\nWrote summary_by_method.csv, pairwise_wilcoxon.csv, by_condition.csv, method_agreement.csv, "
          f"{'sanity_summary.csv, ' if san else ''}summary.png to {d}")


def fast_auroc(scores, labels):
    from scipy.stats import rankdata
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_check_swap(args, parser):
    """ZETA's zero-shot AUROC per condition with the aVL/aVF swap on vs off."""
    import pandas as pd
    import wfdb

    Z.resolve_repo(parser, args)
    from transformers import T5TokenizerFast

    csv = args.csv or str(Path(args.repo).expanduser() / "data_split" / "ptbxl" / args.split /
                          f"ptbxl_{args.split}_test.csv")
    df = pd.read_csv(csv)
    df = df[df["strat_fold"].isin(args.folds)].reset_index(drop=True)
    if args.n_ecgs and len(df) > args.n_ecgs:
        df = df.iloc[np.random.RandomState(args.seed).choice(len(df), args.n_ecgs, replace=False)].reset_index(drop=True)
    with open(args.observations) as f:
        obs = json.load(f)
    conds = [c for c in df.columns[6:] if c in obs and c not in args.skip and (not args.conditions or c in args.conditions)]
    conds = [c for c in conds if df[c].sum() >= args.min_pos and (len(df) - df[c].sum()) >= args.min_pos]
    print(f"{len(df)} ECGs from {csv}; {len(conds)} conditions with >= {args.min_pos} positives and negatives")

    dev = torch.device(args.device)
    model = Z.load_model(args.config, args.checkpoint, dev)
    tok = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)
    branch = Z.ECGBranch.from_model(model).eval()
    bank = TextBank(model, tok, obs, dev)
    PV = {c: torch.stack(list(bank.get(c)["pv"])) for c in conds}
    NV = {c: torch.stack(list(bank.get(c)["nv"])) for c in conds}

    variants = {"on": True, "off": False}
    vecs = {k: [] for k in variants}
    t0 = time.time()
    for i in range(0, len(df), args.batch):
        raws = []
        for fn in df["filename_hr"].iloc[i:i + args.batch]:
            sig, _ = wfdb.rdsamp(str(Path(args.ptbxl_root) / fn))
            raws.append(sig)
        for k, sw in variants.items():
            xb = torch.tensor(np.stack([Z.prepare_ecg(r, sw) for r in raws]), dtype=torch.float32, device=dev)
            with torch.no_grad():
                vecs[k].append(branch.encode(xb)[0].cpu())
        if (i // args.batch) % 10 == 0:
            print(f"  {min(i + args.batch, len(df))}/{len(df)} ECGs, {(time.time() - t0) / 60:.1f} min", flush=True)
    V = {k: torch.cat(v) for k, v in vecs.items()}

    def cond_scores(vec):
        out = {}
        for c in conds:
            sp, sn = vec @ PV[c].cpu().T, vec @ NV[c].cpu().T
            out[c] = torch.sigmoid((sp - sn) / Z.SOFTMAX_TEMP).mean(1).numpy()
        return out

    S = {k: cond_scores(v) for k, v in V.items()}
    y = {c: df[c].values == 1 for c in conds}
    rows = [{"condition": c, "n_pos": int(y[c].sum()), "on": fast_auroc(S["on"][c], y[c]),
             "off": fast_auroc(S["off"][c], y[c])} for c in conds]
    res = pd.DataFrame(rows)
    res["delta_on_minus_off"] = res["on"] - res["off"]
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print("\nZETA zero-shot AUROC per condition")
        print(res.to_string(index=False))
    m_on, m_off = res["on"].mean(), res["off"].mean()
    # bootstrap over ECGs of the macro-mean difference
    rng = np.random.RandomState(args.seed)
    diffs = []
    for _ in range(args.boot):
        idx = rng.randint(0, len(df), len(df))
        d = [fast_auroc(S["on"][c][idx], y[c][idx]) - fast_auroc(S["off"][c][idx], y[c][idx]) for c in conds]
        diffs.append(np.nanmean(d))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nmacro-mean AUROC  swap on {m_on:.3f}   swap off {m_off:.3f}   difference (on - off) {m_on - m_off:+.3f} "
          f"[95% bootstrap CI {lo:+.3f}, {hi:+.3f}]")
    print(f"conditions where on > off: {(res['delta_on_minus_off'] > 0).sum()} / {len(res)}")
    if lo > 0:
        print("-> swap ON scores better; keep --lead-swap on (the default).")
    elif hi < 0:
        print("-> swap OFF scores better; use --lead-swap off.")
    else:
        print("-> no clear difference on this sample; the CI includes 0. Use more ECGs (--n-ecgs 0 = whole fold) "
              "before deciding, or keep the default (on) on the grounds that it matches ZETA's pretraining order.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


def make_figure(S, mat, methods, path, n_ecg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6), facecolor=SURFACE)
    y = np.arange(len(methods))[::-1]

    def dots(a, col, title, xlabel, ref=None, note=None):
        a.set_facecolor(SURFACE)
        for yi, m in zip(y, methods):
            if col not in S or not np.isfinite(S.loc[m, col]):
                continue
            a.plot([S.loc[m, col + "_lo"], S.loc[m, col + "_hi"]], [yi, yi], color=METHOD_COLOURS[m], lw=2, zorder=2)
            a.plot(S.loc[m, col], yi, "o", ms=8, color=METHOD_COLOURS[m], mec=SURFACE, mew=2, zorder=3)
        if ref is not None:
            a.axvline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        a.set_yticks(y)
        a.set_yticklabels(methods)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.set_xlabel(xlabel + (f"\ndashed line = {note}" if note else ""))
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            a.spines[s_].set_visible(False)
        a.tick_params(axis="y", length=0)

    dots(ax[0, 0], "del_adv_n", "Deletion: harm beyond random", "(random − method) / |full − blank|   (mean, 95% CI)",
         ref=0, note="random-order control")
    dots(ax[0, 1], "ins_adv_n", "Insertion: recovery beyond random", "(method − random) / |full − blank|   (mean, 95% CI)",
         ref=0, note="random-order control")
    dots(ax[1, 0], "qrs_auroc", "Beat-locking proxy (not ground truth)", "AUROC vs detected QRS windows   (mean, 95% CI)",
         ref=0.5, note="chance (0.5)")

    a = ax[1, 1]
    a.set_facecolor(SURFACE)
    vals = mat.values.astype(float).copy()
    np.fill_diagonal(vals, np.nan)           # self-agreement is 1 by definition; keep it off the colour scale
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(GRID)
    im = a.imshow(vals, cmap=cmap, vmin=-1, vmax=1)
    a.set_xticks(range(len(methods)))
    a.set_xticklabels(methods, rotation=40, ha="right")
    a.set_yticks(range(len(methods)))
    a.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = vals[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if abs(v) > 0.6 else INK)
    a.set_title("Agreement between methods (mean Spearman)", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    fig.suptitle(f"ZETA localisation methods across {n_ecg} PTB-XL ECGs", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute maps + metrics for many ECGs")
    r.add_argument("--split", default="sub_class", choices=["sub_class", "super_class", "form", "rhythm"])
    r.add_argument("--csv", default=None, help="explicit split CSV (same layout as ZETA's data_split files)")
    r.add_argument("--folds", type=int, nargs="+", default=[10], help="strat_fold values to use (10 = test)")
    r.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    r.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    r.add_argument("--config", default="configs/config.json")
    r.add_argument("--observations", default="configs/observations.json")
    r.add_argument("--repo", default=os.getenv("ZETA_REPO") or os.getenv("ZETA_REPO_ROOT"))
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--lead-swap", choices=["on", "off"], default="on")
    r.add_argument("--condition-source", choices=["label", "zeta-top"], default="label")
    r.add_argument("--conditions", nargs="+", default=None, help="restrict to these conditions")
    r.add_argument("--skip", nargs="*", default=["NORM", "SR"],
                   help="conditions with no meaningful 'where' (default: NORM SR; pass --skip with no values for none)")
    r.add_argument("--per-condition", type=int, default=50, help="max records per condition (0 = all)")
    r.add_argument("--n-ecgs", type=int, default=200, help="records to sample in zeta-top mode")
    r.add_argument("--min-cond-score", type=float, default=0.0)
    r.add_argument("--target", default="logit_diff", choices=["logit_diff", "pos_sim"])
    r.add_argument("--n-steps", type=int, default=20)
    r.add_argument("--n-random", type=int, default=5)
    r.add_argument("--no-xobs", action="store_true", help="skip the all-observations target-dependence check")
    r.add_argument("--sanity-every", type=int, default=0, help="cascading randomisation on every K-th row (0 = off)")
    r.add_argument("--no-save-maps", action="store_true")
    r.add_argument("--shard", default="0/1", help="i/N for job arrays")
    r.add_argument("--max-rows", type=int, default=0, help="stop after this many rows (for timing/tests)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)

    a = sub.add_parser("aggregate", help="summarise results*.jsonl into tables and a figure")
    a.add_argument("--dir", required=True)
    a.add_argument("--min-cond-score", type=float, default=0.0,
                   help="only rows where ZETA's own possibility score for the condition is at least this")
    a.add_argument("--min-range", type=float, default=0.2,
                   help="drop rows from the normalised columns when |full - blank| (logit units) is below this")
    a.add_argument("--conditions", nargs="+", default=None)

    c = sub.add_parser("check-swap", help="ZETA zero-shot AUROC with the aVL/aVF swap on vs off")
    c.add_argument("--split", default="sub_class", choices=["sub_class", "super_class", "form", "rhythm"])
    c.add_argument("--csv", default=None)
    c.add_argument("--folds", type=int, nargs="+", default=[10])
    c.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    c.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    c.add_argument("--config", default="configs/config.json")
    c.add_argument("--observations", default="configs/observations.json")
    c.add_argument("--repo", default=os.getenv("ZETA_REPO") or os.getenv("ZETA_REPO_ROOT"))
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c.add_argument("--conditions", nargs="+", default=None)
    c.add_argument("--skip", nargs="*", default=[])
    c.add_argument("--n-ecgs", type=int, default=1000, help="random sample of the fold (0 = all)")
    c.add_argument("--min-pos", type=int, default=10, help="skip conditions with fewer positives (or negatives)")
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--boot", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default=None, help="optional CSV of the per-condition table")
    return p


def main(argv=None):
    from dotenv import find_dotenv, load_dotenv
    load_dotenv()
    load_dotenv(find_dotenv(usecwd=True))
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args, p)
    elif args.cmd == "check-swap":
        cmd_check_swap(args, p)
    else:
        cmd_aggregate(args, p)


if __name__ == "__main__":
    main()