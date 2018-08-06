#!/bin/bash
file=$1
# gnuplot -e "plot  '$file' using 1:3 with lines, '$file' using 1:5 with lines" -p
gnuplot -e "plot  '$file' using 1:2 with lines, '$file' using 1:3 with lines,'$file' using 1:4 with lines,'$file' using 1:5 with lines" -p
