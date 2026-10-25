#!/bin/sh
python run_pipeline.py --records-file records.txt --prompt-variant baseline \
  --conditions VPC,LBBB,RBBB,NORM --seed 0 --output recheck_first.json