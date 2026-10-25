python zeta_relevance_batch.py run --folds 10 --lead-swap on \
  --condition-source label --conditions CRBBB IRBBB CLBBB "LAFB/LPFB" \
  --obs-filter obs_auroc.csv --min-obs-auroc 0.75 \
  --select top --per-condition 40 --candidates 150 --sanity-every 5 --out focus_filtered
python zeta_relevance_batch.py aggregate --dir focus_filtered