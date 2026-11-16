#!/bin/bash
# WL97 E3 job 2: anchor fix rerun, bindir model, convergence ladder (own dir only)
cd ~/rest_x97/E3
C=/home/nd/rest_x97/E3/code; O=/home/nd/rest_x97/E3/out
rm -f job2.done
./run_e3.sh quad97.py deliver $C $O s8r12 8 12 --tables=$O/tables_a.npz --bindir=384,256
./run_e3.sh quad97.py deliver $C $O s16r12 16 12 --tables=$O/tables_a.npz --bindir=384,256
./run_e3.sh quad97.py deliver $C $O s16r12hv 16 12 --tables=$O/tables_a.npz --bindir=768,512
./run_e3.sh quad97.py deliver $C $O s4r12 4 12 --tables=$O/tables_a.npz
./run_e3.sh quad97.py deliver $C $O s4r12c 4 12 --tables=$O/tables_c.npz
./run_e3.sh quad97.py deliver $C $O s2r12 2 12 --tables=$O/tables_a.npz
./run_e3.sh quad97.py deliver $C $O s4r24 4 24 --tables=$O/tables_a.npz
./run_e3.sh quad97.py deliver $C $O s4r6 4 6 --tables=$O/tables_a.npz
./run_e3.sh quad97.py deliver $C $O s8r12o1 8 12 --tables=$O/tables_a.npz --off=0.25,0.25 --roff=0.25,0.75
./run_e3.sh quad97.py deliver $C $O s8r12o2 8 12 --tables=$O/tables_a.npz --off=0.75,0.75 --roff=0.75,0.25
./run_e3.sh quad97.py tables $C $O b 16 768 512 8
./run_e3.sh quad97.py deliver $C $O s4r12b 4 12 --tables=$O/tables_b.npz
touch job2.done
