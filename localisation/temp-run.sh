for c in CLBBB CRBBB IRBBB LAFB/LPFB; do
  d=$(echo $c | tr '/' '-')
  python zeta_demo.py --condition "$c" --n 3 --obs-filter obs_auroc.csv --out demo_$d
done