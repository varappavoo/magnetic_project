samples(x) = ((column(1)) > 4) ? 5 : (column(1)+1)
avg5(x) = (shift5(x), (back1+back2+back3+back4+back5)/samples(column(1)))
shift5(x) = (back5 = back4, back4 = back3, back3 = back2, back2 = back1, back1 = x)

#
# Initialize a running sum
#
init(x) = (back1 = back2 = back3 = back4 = back5 = sum = 0)
#
# Plot data, running average and cumulative average
#

datafile = 's1_0cm.dat'
# set xrange [0:57]

set style data lines


#stats '$file_s1' using 1 prefix 'T' nooutput;print 'T min: ', T_min ;stats '$file_s1' using 3 prefix 'A' nooutput;print 's1 std dev: ',A_stddev;  print 's1 min: ', A_min ;stats '$file_s2' using 3 prefix 'B' nooutput;print 's2 std dev: #',B_stddev;  print 's2 min: ', B_min ;set term wxt;

plot sum = init(0), \
     datafile using 0:2 title 'data' lw 2 lc rgb 'forest-green', \
     '' using 1:(avg5(column(3))) title "running mean over previous 5 points" pt 7 ps 0.5 lw 1 lc rgb "blue", \
     '' using 1:(sum = sum + (column(3)), sum/(column(3)+1)) title "cumulative mean" pt 1 lw 1 lc rgb "dark-red"