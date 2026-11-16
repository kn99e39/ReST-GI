#!/bin/bash
# WL97 E5 chain 2 (sequential, one GPU process at a time): OpenMP wait-policy control, repeat of s 1.0, scaling s 1.5 / 2.0 / 3.0
B=$HOME/rest_x97/E5
OMP_WAIT_POLICY=passive bash $B/code/run_e5.sh run2_passive_s1.0 1.0 --lamonly --noprof
bash $B/code/run_e5.sh run3_s1.0 1.0
bash $B/code/run_e5.sh run4_s1.5 1.5 --quick
bash $B/code/run_e5.sh run5_s2.0 2.0 --quick
bash $B/code/run_e5.sh run6_s3.0 3.0 --quick
touch $B/out/chain2.done
