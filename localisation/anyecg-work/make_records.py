#!/usr/bin/env python3
"""Generate records.txt for run_pipeline.py from MIT-BIH .atr annotations.

Usage: python make_records.py [--root /path/to/MIT_BIH]  (or set MITBIH_ROOT in .env) [--per-class 5] [--out records.txt]
Needs: pip install wfdb. Writes one absolute record path per line (no extension).
Selection (from beat annotations): LBBB = mostly 'L' beats, RBBB = mostly 'R',
PVC = >=100 'V' beats with the rest mostly 'N', NORM = all 'N' and ~no V/A/L/R.
"""
import argparse, collections, os, wfdb
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("MITBIH_ROOT"),
                help="MIT-BIH folder (default: MITBIH_ROOT from .env)")
ap.add_argument("--per-class", type=int, default=5)
ap.add_argument("--out", default="records.txt")
a = ap.parse_args()
if not a.root:
    ap.error("Set MITBIH_ROOT in .env or pass --root.")

rows = {}
for f in sorted(os.listdir(a.root)):
    if f.endswith(".atr"):
        r = f[:-4]
        c = collections.Counter(wfdb.rdann(os.path.join(a.root, r), "atr").symbol)
        beats = sum(c[k] for k in "NLRVAaJjSEFfQe")
        rows[r] = (c, max(beats, 1))

def pick(score, k):
    return [r for r, _ in sorted(((r, score(*v)) for r, v in rows.items()), key=lambda t: -t[1])[:k] if _ > 0]

lbbb = pick(lambda c, n: c["L"] / n if c["L"] / n > .8 else 0, a.per_class)
rbbb = pick(lambda c, n: c["R"] / n if c["R"] / n > .5 and c["V"] < 100 else 0, a.per_class)
pvc = pick(lambda c, n: c["V"] if c["V"] >= 100 and c["L"] + c["R"] + c["/"] < 10 else 0, a.per_class)
norm = pick(lambda c, n: c["N"] / n if c["N"] / n > .98 and c["V"] + c["A"] <= 3 and c["/"] == 0 else 0, a.per_class)
sel = {"LBBB": lbbb, "RBBB": rbbb, "PVC": pvc, "NORM": norm}
with open(a.out, "w") as fh:
    for k, v in sel.items():
        print(k, v)
        for r in v:
            fh.write(os.path.join(os.path.abspath(a.root), r) + "\n")
print("wrote", a.out)