python zeta_relevance_batch.py run --folds 10 --lead-swap on \
  --condition-source label --conditions CRBBB IRBBB CLBBB "LAFB/LPFB" \
  --select top --per-condition 40 --candidates 150 --sanity-every 5 --out focus
python zeta_relevance_batch.py aggregate --dir focus