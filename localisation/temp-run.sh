python zeta_relevance_batch.py run \
  --folds 10 --lead-swap on --condition-source label \
  --select top --per-condition 30 --candidates 120 \
  --sanity-every 5 --out full
python zeta_relevance_batch.py aggregate --dir full