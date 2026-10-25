python zeta_relevance_batch.py run --folds 10 --lead-swap on --condition-source label \
  --select top --per-condition 30 --candidates 120 --sanity-every 5 --out full2
python zeta_relevance_batch.py aggregate --dir full2


python zeta_relevance.py --ecg_id 172 --condition CRBBB --all-obs --eval \
  --specificity --lead-attr --plot-lead 6 --save_dir crbbb_v2