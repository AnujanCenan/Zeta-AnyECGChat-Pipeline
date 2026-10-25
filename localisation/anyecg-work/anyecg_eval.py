#!/usr/bin/env python3
"""
anyecg_eval.py -- score anyECG-chat's localisation against MIT-BIH beat annotations.

No ZETA involved. For each record, 10 s windows are cut; the TRUE condition comes
from the .atr beat labels (L -> LBBB, R -> RBBB, V -> VPC), and anyECG-chat is asked
its own training question ("Locate the <name> on this ECG for me, please.").
Its returned span(s) are compared against where the labelled beats actually are.

Window roles (per window x question):
  pos : >= 2 beats of the asked type in the window
  neg : 0 beats of the asked type  (includes clean-normal windows and "cross"
        windows, e.g. an LBBB window asked about RBBB)
Metrics
  detection        pos: fraction answered with a span (not "Not Found")
  false-positive   neg: fraction answered with a span
  On "mixed" pos windows (target beats AND other beats present -- mainly PVCs among
  normal beats), where localisation is non-trivial:
    recall      = target beats inside predicted span(s) / target beats
    specificity = non-target beats outside span(s) / non-target beats
    balanced    = (recall + specificity)/2   (0.5 for "whole window" and for "Not Found")
    iou         = IoU of predicted span vs union of [t-0.25, t+0.25 s] around target beats
  vs baselines: random contiguous span of the same total length (Monte-Carlo), whole window.
  Pure windows (all beats are the target type, e.g. LBBB records) cannot show localisation,
  only detection, so they are reported separately.

Usage (on Katana, same .env as run_pipeline.py; MITBIH_ROOT optional):
  python anyecg_eval.py --records-file records.txt --out anyecg_eval
  python anyecg_eval.py --records-file records.txt --out selftest --dry-run oracle   # no GPU/model
"""
from __future__ import annotations
import argparse, csv, json, os, random, struct, sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

FS = 360.0
WIN_S = 10.0
SYM = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S",
       10: "E", 11: "j", 12: "/", 13: "Q", 34: "e"}
COND = {"LBBB": ("L", "Left bundle branch block beat"),
        "RBBB": ("R", "Right bundle branch block beat"),
        "VPC": ("V", "Premature ventricular contraction")}


def read_atr(path):
    """[(sample, symbol)] from a MIT-BIH .atr file (own reader; wfdb rdann breaks on numpy 2)."""
    b = open(path, "rb").read()
    i, t, out = 0, 0, []
    while i + 2 <= len(b):
        w = struct.unpack_from("<H", b, i); w = w[0]
        i += 2
        code, data = w >> 10, w & 1023
        if code == 0 and data == 0:
            break
        if code == 59:
            hi, lo = struct.unpack_from("<HH", b, i)
            i += 4
            t += (hi << 16) | lo
        elif code in (60, 61, 62):
            pass
        elif code == 63:
            i += data + (data & 1)
        else:
            t += data
            if code in SYM:
                out.append((t, SYM[code]))
    return out


def merge(spans):
    spans = sorted((max(0.0, a), min(WIN_S, b)) for a, b in spans if b > a)
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def covered(t, spans):
    return any(a <= t <= b for a, b in spans)


def length(spans):
    return sum(b - a for a, b in spans)


def iou(pred, gt):
    pred, gt = merge(pred), merge(gt)
    inter = 0.0
    for a, b in pred:
        for c, d in gt:
            inter += max(0.0, min(b, d) - max(a, c))
    union = length(pred) + length(gt) - inter
    return inter / union if union > 0 else float("nan")


def metrics(spans, tgt, oth):
    """recall / specificity / balanced for one window. spans may be [] (not found)."""
    rec = sum(covered(t, spans) for t in tgt) / len(tgt)
    spec = (sum(not covered(t, spans) for t in oth) / len(oth)) if oth else float("nan")
    return rec, spec, (rec + spec) / 2 if oth else float("nan")


def random_baseline(spans, tgt, oth, rng, n=200):
    L = length(spans)
    if L <= 0:
        return 0.5
    if L >= WIN_S:
        return 0.5
    vals = []
    for _ in range(n):
        s = rng.uniform(0, WIN_S - L)
        vals.append(metrics([(s, s + L)], tgt, oth)[2])
    return float(np.mean(vals))


def build_windows(records, args, rng):
    """Returns list of dict(record, start, beats[(t_sec, sym)]) sampled per record."""
    wins = []
    for rec in records:
        ann = read_atr(rec + ".atr")
        n_samp = ann[-1][0] if ann else 0
        n_win = int(n_samp // (WIN_S * FS))
        by_win = {}
        for s, sym in ann:
            w = int(s // (WIN_S * FS))
            if w < n_win:
                by_win.setdefault(w, []).append(((s - w * WIN_S * FS) / FS, sym))
        chosen = {}
        for cname, (sym, _) in COND.items():
            cand = [w for w, bl in by_win.items() if sum(1 for _, x in bl if x == sym) >= 2]
            rng.shuffle(cand)
            cand.sort(key=lambda w: 0 if any(x != sym for _, x in by_win[w]) else 1)  # mixed first
            for w in cand[: args.pos_windows]:
                chosen[w] = by_win[w]
        clean = [w for w, bl in by_win.items() if len(bl) >= 5 and not any(x in "LRV" for _, x in bl)]
        rng.shuffle(clean)
        for w in clean[: args.neg_windows]:
            chosen[w] = by_win[w]
        for w, bl in sorted(chosen.items()):
            wins.append({"record": rec, "window": w, "start": int(w * WIN_S * FS), "beats": bl})
    return wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-file", required=True)
    ap.add_argument("--out", default="anyecg_eval")
    ap.add_argument("--pos-windows", type=int, default=4, help="positive windows per record per condition")
    ap.add_argument("--neg-windows", type=int, default=3, help="clean (no L/R/V) windows per record")
    ap.add_argument("--conditions", default="LBBB,RBBB,VPC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", choices=["oracle", "whole", "none"], default=None,
                    help="skip the model; fake answers to self-test the metrics")
    ap.add_argument("--zeta-repo", default=os.environ.get("ZETA_REPO"))
    ap.add_argument("--anyecg-repo", default=os.environ.get("ANYECG_REPO"))
    ap.add_argument("--projection-ckpt", default=os.environ.get("PROJECTION_CKPT"))
    ap.add_argument("--ecg-model-ckpt", default=os.environ.get("ECG_MODEL_CKPT"))
    ap.add_argument("--lora-ckpt", default=os.environ.get("LORA_CKPT"))
    ap.add_argument("--ecg-encoder-ckpt", default=os.environ.get("ECG_ENCODER_CKPT"))
    ap.add_argument("--llm-model-id", default=os.environ.get("LLM_MODEL_ID"))
    args = ap.parse_args()

    args.out = str(Path(args.out).resolve())  # before chdir into the ZETA repo
    rng = random.Random(args.seed)
    records = [l.strip() for l in open(args.records_file) if l.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    wins = build_windows(records, args, rng)
    print(f"{len(records)} records -> {len(wins)} windows x {len(conds)} questions")

    model = pre = run_loc = None
    if args.dry_run is None:
        here = Path(__file__).resolve().parent
        for p in (args.anyecg_repo, args.zeta_repo, str(here)):
            if p:
                sys.path.insert(0, str(Path(p).resolve()))
        os.chdir(args.zeta_repo)
        from ecg_preprocessing import ECGPreprocessor
        from localize_finding import load_anyecg_model, run_anyecg_localization
        model = load_anyecg_model(projection_ckpt=str(args.projection_ckpt),
                                  ecg_model_ckpt=str(args.ecg_model_ckpt) if args.ecg_model_ckpt else None,
                                  lora_ckpt=str(args.lora_ckpt) if args.lora_ckpt else None,
                                  ecg_encoder_ckpt_path=args.ecg_encoder_ckpt, llm_model_id=args.llm_model_id)
        pre, run_loc = ECGPreprocessor(), run_anyecg_localization

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, w in enumerate(wins):
        ecg = None
        for cname in conds:
            sym, phrase = COND[cname]
            tgt = [t for t, x in w["beats"] if x == sym]
            oth = [t for t, x in w["beats"] if x != sym]
            if len(tgt) == 1:
                continue
            role = "pos" if len(tgt) >= 2 else "neg"
            kind = ("pure" if not oth else "mixed") if role == "pos" else (
                "cross" if any(x in "LR" for _, x in w["beats"]) else "clean")
            raw, spans, err = "", [], None
            try:
                if args.dry_run:
                    if args.dry_run == "oracle":
                        spans = merge([(t - 0.3, t + 0.3) for t in tgt]) if role == "pos" else []
                    elif args.dry_run == "whole":
                        spans = [(0.1, WIN_S)]
                    raw = f"dry-run:{args.dry_run}"
                else:
                    if ecg is None:
                        ecg = pre.to_anyecg_input(w["record"], crop_seconds=WIN_S,
                                                  ecg_start=w["start"], ecg_end=w["start"] + int(WIN_S * FS))
                    _, raw, iv = run_loc(ecg, phrase, model, "baseline")
                    spans = merge(iv) if iv else []
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            found = bool(spans)
            row = {"record": os.path.basename(w["record"]), "window": w["window"], "condition": cname,
                   "role": role, "kind": kind, "n_target": len(tgt), "n_other": len(oth),
                   "raw": raw, "found": found, "span_len": round(length(spans), 2), "error": err}
            if role == "pos" and kind == "mixed" and err is None:
                rec, spec, bal = metrics(spans, tgt, oth)
                gt = [(t - 0.25, t + 0.25) for t in tgt]
                row.update(recall=rec, specificity=spec, balanced=bal,
                           random_balanced=random_baseline(spans, tgt, oth, rng),
                           iou=iou(spans, gt) if found else 0.0)
            rows.append(row)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(wins)} windows")

    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "windows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    summ = {"n_rows": len(rows), "errors": sum(r["error"] is not None for r in rows)}
    ok = [r for r in rows if r["error"] is None]
    for cname in conds:
        c = [r for r in ok if r["condition"] == cname]
        pos = [r for r in c if r["role"] == "pos"]
        s = {"pos_windows": len(pos),
             "detection_all_pos": float(np.mean([r["found"] for r in pos])) if pos else None}
        for kind in ("pure", "mixed"):
            kk = [r for r in pos if r["kind"] == kind]
            s[f"detection_{kind}"] = float(np.mean([r["found"] for r in kk])) if kk else None
        for kind in ("clean", "cross"):
            kk = [r for r in c if r["role"] == "neg" and r["kind"] == kind]
            s[f"false_positive_{kind}"] = float(np.mean([r["found"] for r in kk])) if kk else None
        mx = [r for r in pos if r["kind"] == "mixed"]
        if mx:
            for m in ("recall", "specificity", "balanced", "random_balanced", "iou"):
                s[f"mixed_{m}"] = float(np.nanmean([r[m] for r in mx]))
            s["mixed_n"] = len(mx)
            s["mixed_n_records"] = len({r["record"] for r in mx})
            # per-record paired comparison (avoids windows from one record counting as independent)
            per = {}
            for r in mx:
                per.setdefault(r["record"], []).append(r["balanced"] - r["random_balanced"])
            d = np.array([np.mean(v) for v in per.values()])
            s["mixed_balanced_minus_random_per_record"] = {k: float(np.mean(v)) for k, v in per.items()}
            if len(d) >= 6 and np.any(d != 0):
                from scipy.stats import wilcoxon
                s["mixed_wilcoxon_p"] = float(wilcoxon(d).pvalue)
        summ[cname] = s
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"\nWrote {out_dir}/windows.csv and {out_dir}/summary.json")


if __name__ == "__main__":
    main()