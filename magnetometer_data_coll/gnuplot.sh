#!/bin/bash
#gnuplot -e "set xdata time;set timefmt '%s';plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines" -p
#gnuplot -e "set xdata time;set timefmt '%s';set format x '%m/%d/%Y %H:%M:%S';plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines" -p

#gnuplot -e "set xdata time;set timefmt '%s';set format x '%m/%d/%Y %H:%M:%S';plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines, 'magnetometer_data_1524224981_gnu.csv' u 1:2 with lines, 'magnetometer_data_1524224981_gnu.csv' u 1:3 with lines,'magnetometer_data_1524224981_gnu.csv' u 1:4 with lines" -p
#gnuplot -e "set xdata time;set timefmt '%s';set xrange [1524225100:1524225200];plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines, 'magnetometer_data_1524224981_gnu.csv' u 1:2 with lines, 'magnetometer_data_1524224981_gnu.csv' u 1:3 with lines,'magnetometer_data_1524224981_gnu.csv' u 1:4 with lines,'magnetometer_data_1524224981_gnu.csv' u 1:5 smooth acsplines" -p
#gnuplot -e "set xdata time;set timefmt '%s';set xrange [1524225000:];plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines title 'raw', 'magnetometer_data_1524224981_gnu.csv' u 1:5 smooth acsplines title 'acsplines'" -p
# gnuplot -e "set terminal png size 1600, 400;set size 1, 1;set output 'magneto_zoom.png';set xdata time;set timefmt '%s';set yrange [55:65];set xrange [1524225100:1524225300];plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines title 'raw', 'magnetometer_data_1524224981_gnu.csv' u 1:5 smooth acsplines title 'acsplines'" -p
# gnuplot -e "set terminal png size 1600, 400;set size 1, 1;set output 'magneto_zoom.png';set xdata time;set timefmt '%s';set yrange [55:65];set xrange [1524225100:1524225300];plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines title 'raw', 'magnetometer_data_1524224981_gnu.csv' u 1:5 smooth acsplines title 'acsplines'" -p

# gnuplot -e "set terminal png size 1600, 400;set size 1, 1;set output 'magneto_avg.png';set xdata time;set timefmt '%s';set yrange [58:60];set xrange [1524225100:1524225300];plot 'magnetometer_data_1524224981_gnu.csv' u 1:5 with lines title 'raw', 'magnetometer_data_1524224981_gnu_m_avg.csv' u 1:2 with lines title 'avg'" -p

# gnuplot -e "stats 'magnetometer_data_1524224981_gnu_m_avg.csv'  using 2 prefix 'A'; print 'mean: ',A_mean; print 'std dev: ',A_stddev;  print 'min: ',A_min; ;print 'lo quartile: ',A_lo_quartile; print 'median: ', A_median; print 'upper quartile: ', A_up_quartile;print 'max: ',  A_max"
# log_20180514_1451.txt.gnu > log_20180514_1451.txt.gnu.avg


gnuplot -e "set terminal png size 1600, 400;set size 1, 1;set output 'magneto_mobile_exp.png';set xdata time;set timefmt '%s'; set yrange[127:132]; set xrange[1526280800:1526281122];plot 'log_20180514_1451.txt.gnu' u 1:2 with lines title 'raw', 'log_20180514_1451.txt.gnu.avg' u 1:2 with lines title 'avg'" -p