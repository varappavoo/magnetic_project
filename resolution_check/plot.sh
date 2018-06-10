#!/bin/bash
file_s1=$1
file_s2=$2
data_col=4
extension='.avg'
#cat $file_s1 |awk  -F "\"* \"*"  -v avg=1 N=1000{'avg-=avg/N;avg+=$3/N;print $1,$2,$3,avg;fflush();'}|awk 'NR>6000;fflush();' > $file_s1$extension
#cat $file_s2 |awk  -F "\"* \"*"  -v avg=1 N=1000{'avg-=avg/N;avg+=$3/N;print $1,$2,$3,avg;fflush();'}|awk 'NR>6000;fflush();' > $file_s2$extension
file_s1=$file_s1$extension
file_s2=$file_s2$extension
#gnuplot -e "set term wxt;set xrange[0:60];stats '$file_s1' using 1 prefix 'T' nooutput;print 'T min: ', T_min ;stats '$file_s1' using $data_col prefix 'A' nooutput;print 's1 std dev: ',A_stddev;  print 's1 min: ', A_min ;stats '$file_s2' using $data_col prefix 'B' nooutput;print 's2 std dev: ',B_stddev;  print 's2 min: ', B_min ;set term wxt;plot '$file_s1' using (column(1)-T_min):(column($data_col)-A_min) with lines, '$file_s2' using (column(1)-T_min):(column($data_col)-B_min) with lines" -p
gnuplot -e "set term wxt;stats '$file_s1' using 1 prefix 'T' nooutput;print 'T min: ', T_min ;stats '$file_s1' using $data_col prefix 'A' nooutput;print 's1 std dev: ',A_stddev;  print 's1 min: ', A_min ;stats '$file_s2' using $data_col prefix 'B' nooutput;print 's2 std dev: ',B_stddev;  print 's2 min: ', B_min ;set term wxt;plot '$file_s1' using (column(1)-T_min):(column($data_col)-A_min) with lines, '$file_s2' using (column(1)-T_min):(column($data_col)-B_min) with lines" -p
