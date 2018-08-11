#!/bin/bash
ref=6.25 #400*0.25^3
gnuplot -e 'set xrange [0:4]; set yrange [0:7];set parametric; set grid ytics lt 0 lw 1 lc rgb "#bbbbbb";set grid xtics lt 0 lw 1 lc rgb "#bbbbbb"; plot  t, (1/(t*t)) title "1/r^2", t, (1/(t*t*t)) title "1/r^3";' -p
# gnuplot -e 'set xrange [0:4]; set yrange [0:4];set parametric; set grid ytics lt 0 lw 1 lc rgb "#bbbbbb";set grid xtics lt 0 lw 1 lc rgb "#bbbbbb"; plot t, ($ref/(t*t)) title "$ref/r^2", t,($ref/(t*t*t)) title "$ref/r^3";' -p
gnuplot -e "set xrange [0:4]; set yrange [0:7];set parametric; set grid ytics lt 0 lw 1 lc rgb '#bbbbbb';set grid xtics lt 0 lw 1 lc rgb '#bbbbbb'; plot t, ($ref/(t*t)) title '$ref/r^2', t,($ref/(t*t*t)) title '$ref/r^3';" -p
