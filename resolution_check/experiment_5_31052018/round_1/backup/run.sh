#!/bin/bash
device=$1
filename=$2
echo "DEVICE: $device	filename: $filename"
# ~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 /dev/ttyACM0|ts "%.s"|awk -v avg=$2 -v N=20 {'avg=avg-(avg/N);avg=avg+($2/N);print $1,$2/100,avg/100;fflush();'}|awk 'NR>500;fflush();'|tee /dev/tty|awk {'print $1%10000,$2,$3;fflush();'}
~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 $1|ts "%.s"|awk -v avg=$2 -v N=20 {'avg=avg-(avg/N);avg=avg+($2/N);print $1,$2/100,avg/100;fflush();'}|awk 'NR>500;fflush();'|awk {'print $1%10000,$2,$3;fflush();'} |tee >(awk '{print $1,$3;fflush();}'|feedgnuplot --lines --stream 0.01 --ylabel 'utesla' --xlabel 'time' --domain --xlen 300) >(feedgnuplot --lines --stream 0.01 --ylabel 'utesla' --xlabel 'time' --domain --xlen 30)  > $filename


