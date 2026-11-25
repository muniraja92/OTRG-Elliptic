#!/bin/bash
# GNN baselines on the second datasets (run_gnn_second.py), one lock acquisition per dataset.
R=/home/claude/TRUST_GRAF_Research; E=$R/experiments/second_dataset; L=$R/logs; cd /tmp
for ds in amazon yelp; do
  echo "=== $ds wait $(date +%T)" >> $L/gnn_second_chain.log
  flock /tmp/cpu.lock python3 -I $E/run_gnn_second.py $ds > $L/gnn_second_$ds.log 2>&1
  echo "=== $ds rc=$? $(date +%T)" >> $L/gnn_second_chain.log
done
