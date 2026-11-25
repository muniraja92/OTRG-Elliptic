#!/bin/bash
# Sequential pipeline; each heavy step takes the shared CPU lock separately.
R=/home/claude/TRUST_GRAF_Research; E=$R/experiments/second_dataset; DD=$R/data/second_dataset; L=$R/results/second_dataset
cd /tmp
flock /tmp/cpu.lock python3 -I $E/run_second.py yelp > $L/run_yelp.log 2>&1 || exit 1
flock /tmp/cpu.lock python3 -I $E/run_second.py amazon > $L/run_amazon.log 2>&1 || exit 1
flock /tmp/cpu.lock python3 -I $E/evaluate_second.py yelp amazon > $L/evaluate.log 2>&1
echo CHAIN_DONE >> $L/evaluate.log
