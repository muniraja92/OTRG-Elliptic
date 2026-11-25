#!/bin/bash
# Re-run of the rewired control with configuration-model rewiring, remaining degree-only control, evaluation.
R=/home/claude/TRUST_GRAF_Research; E=$R/experiments/second_dataset; L=$R/results/second_dataset; cd /tmp
flock /tmp/cpu.lock python3 -I $E/run_second.py yelp rewired_only > $L/run_yelp_rewired_v2.log 2>&1 || exit 1
flock /tmp/cpu.lock python3 -I $E/run_second.py amazon rewired_only > $L/run_amazon_rewired_v2.log 2>&1 || exit 1
flock /tmp/cpu.lock python3 -I $E/run_degree_only.py amazon > $L/run_deg_amazon.log 2>&1 || exit 1
flock /tmp/cpu.lock python3 -I $E/evaluate_second.py yelp amazon > $L/evaluate.log 2>&1
echo CHAIN3_DONE >> $L/evaluate.log
