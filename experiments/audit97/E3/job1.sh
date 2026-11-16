#!/bin/bash
cd ~/rest_x97/E3
C=/home/nd/rest_x97/E3/code; O=/home/nd/rest_x97/E3/out
rm -f job1.done
./run_e3.sh quad97.py tables $C $O a 16 384 256 8
./run_e3.sh quad97.py tables $C $O c 32 384 256 32
./run_e3.sh quad97.py deliver $C $O s8r12 8 12 --tables=$O/tables_a.npz
./run_e3.sh quad97.py deliver $C $O s4r12 4 12 --tables=$O/tables_a.npz
touch job1.done
