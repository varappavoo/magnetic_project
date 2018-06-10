#!/bin/bash
filename=$1
./plot_align_trim.py -f1 s1_3.5cm_200cm.dat -f2  s2_3.5cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s1_7cm_200cm.dat -f2  s2_7cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s1_17cm_200cm.dat -f2  s2_17cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s1_34.9cm_200cm.dat -f2  s2_34.9cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s1_69.5cm_200cm.dat -f2  s2_69.5cm_200cm.dat >> $filename &
./plot_align_trim.py  -f1 s1_5.2cm_300cm.dat  -f2 s2_5.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s1_10.5cm_300cm.dat  -f2 s2_10.5cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s1_26.2cm_300cm.dat  -f2 s2_26.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s1_52.3cm_300cm.dat  -f2 s2_52.3cm_300cm.dat  >> $filename &
./plot_align_trim.py -f1 s1_6.1cm_350cm.dat  -f2 s2_6.1cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s1_12.2cm_350cm.dat  -f2 s2_12.2cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s1_30.5cm_350cm.dat  -f2 s2_30.5cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s1_61cm_350cm.dat  -f2 s2_61cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s1_121cm_350cm.dat  -f2 s2_121cm_350cm.dat >> $filename &

./plot_align_trim.py -f1 s0_3.5cm_200cm.dat -f2  s1_3.5cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_7cm_200cm.dat -f2  s1_7cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_17cm_200cm.dat -f2  s1_17cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_34.9cm_200cm.dat -f2  s1_34.9cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_69.5cm_200cm.dat -f2  s1_69.5cm_200cm.dat >> $filename &
./plot_align_trim.py  -f1 s0_5.2cm_300cm.dat  -f2 s1_5.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_10.5cm_300cm.dat  -f2 s1_10.5cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_26.2cm_300cm.dat  -f2 s1_26.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_52.3cm_300cm.dat  -f2 s1_52.3cm_300cm.dat  >> $filename &
./plot_align_trim.py -f1 s0_6.1cm_350cm.dat  -f2 s1_6.1cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_12.2cm_350cm.dat  -f2 s1_12.2cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_30.5cm_350cm.dat  -f2 s1_30.5cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_61cm_350cm.dat  -f2 s1_61cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_121cm_350cm.dat  -f2 s1_121cm_350cm.dat >> $filename &

./plot_align_trim.py -f1 s0_3.5cm_200cm.dat -f2  s2_3.5cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_7cm_200cm.dat -f2  s2_7cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_17cm_200cm.dat -f2  s2_17cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_34.9cm_200cm.dat -f2  s2_34.9cm_200cm.dat >> $filename &
./plot_align_trim.py -f1 s0_69.5cm_200cm.dat -f2  s2_69.5cm_200cm.dat >> $filename &
./plot_align_trim.py  -f1 s0_5.2cm_300cm.dat  -f2 s2_5.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_10.5cm_300cm.dat  -f2 s2_10.5cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_26.2cm_300cm.dat  -f2 s2_26.2cm_300cm.dat  >> $filename &
./plot_align_trim.py  -f1 s0_52.3cm_300cm.dat  -f2 s2_52.3cm_300cm.dat  >> $filename &
./plot_align_trim.py -f1 s0_6.1cm_350cm.dat  -f2 s2_6.1cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_12.2cm_350cm.dat  -f2 s2_12.2cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_30.5cm_350cm.dat  -f2 s2_30.5cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_61cm_350cm.dat  -f2 s2_61cm_350cm.dat >> $filename &
./plot_align_trim.py -f1 s0_121cm_350cm.dat  -f2 s2_121cm_350cm.dat >> $filename &