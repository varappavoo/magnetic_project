#!/bin/bash
file_s1=$1
file_s2=$2
gnuplot -e "stats '$file_s1' using 1 prefix 'T' nooutput;print 'T min: ', T_min ;stats '$file_s1' using 3 prefix 'A' nooutput;print 's1 std dev: ',A_stddev;  print 's1 min: ', A_min ;stats '$file_s2' using 3 prefix 'B' nooutput;print 's2 std dev: ',B_stddev;  print 's2 min: ', B_min ;set term wxt;set xrange[0:60];plot '$file_s1' using (column(1)-T_min):(column(3)-A_min) with lines, '$file_s2' using (column(1)-T_min):(column(3)-B_min) with lines" -p

