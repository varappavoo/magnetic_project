#!/bin/bash
#echo "RUN EXAMPLE: ./run.sh /dev/ttyACM0 test.dat"
device=$1
filename=$2
echo "DEVICE: $device	filename: $filename"
~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 $device|ts "%.s"|awk -v avg=$2 -v N=20 {'avg=avg-(avg/N);avg=avg+($2/N);print $1,$2/100,avg/100;fflush();'}|awk 'NR>500;fflush();'|tee $filename
