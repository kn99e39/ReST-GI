#!/bin/bash
cd ~/rest_x97/E3
rm -f job3.done
./run_e3.sh mc97.py /home/nd/rest_x97/E3/code /home/nd/rest_x97/E3/out 23 32 97001
touch job3.done
