#!/bin/bash
experiment=$1
./run.sh /dev/ttyACM1 "s0_1_$experiment.dat" 2>/dev/null &
./run.sh /dev/ttyACM3 "s0_2_$experiment.dat" 2>/dev/null &
./run.sh /dev/ttyACM5 "s0_3_$experiment.dat" 2>/dev/null &
./run.sh /dev/ttyACM7 "s0_4_$experiment.dat" 2>/dev/null &
./run.sh /dev/ttyACM9 "s0_5_$experiment.dat" 2>/dev/null &
./run.sh /dev/ttyACM11 "s1_$experiment.dat" 2>/dev/null &
#watch -n 1 'ls *dat -l'
