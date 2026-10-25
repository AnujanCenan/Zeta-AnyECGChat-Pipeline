#!/usr/bin/env python3
"""Generate records.txt for run_pipeline.py from MIT-BIH .atr annotations.

Usage: python make_records.py [--root /path/to/MIT_BIH]  (or set MITBIH_ROOT in .env) [--per-class 5] [--out records.txt]
No wfdb needed (own .atr reader; wfdb breaks on some numpy 2 installs). Writes one absolute record path per line (no extension).
Selection (from beat annotations): LBBB = mostly 'L' beats, RBBB = mostly 'R',
PVC = >=100 'V' beats with the rest mostly 'N', NORM = all 'N' and ~no V/A/L/R.
"""
import argparse, collections, os, struct
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

SYM = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S",
       10: "E", 11: "j", 12: "/", 13: "Q", 34: "e"}


def read_atr_symbols(path):
    """Beat-symbol list from a MIT-BIH .atr file (16-bit LE words: code<<10 | data)."""
    b = open(path, "rb").read()
    i, out = 0, []
    while i + 2 <= len(b):
        w = struct.unpack_from("<H", b, i)[0]
        i += 2
        code, data = w >> 10, w & 1023
        if code == 0 and data == 0:
            break
        if code == 59:      # SKIP: 4-byte interval follows
            i += 4
        elif code in (60, 61, 62):   # NUM / SUB / CHN
            pass
        elif code == 63:    # AUX: `data` bytes, padded to even
            i += data + (data & 1)
        elif code in SYM:
            out.append(SYM[code])
    return out


rows = {}
for f in sorted(os.listdir(a.root)):
    if f.endswith(".atr"):
        r = f[:-4]
        c = collections.Counter(read_atr_symbols(os.path.join(a.root, f)))
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