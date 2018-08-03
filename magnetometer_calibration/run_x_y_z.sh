#!/bin/bash
# ~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 /dev/ttyACM0|ts "%.s"|awk -v avg=$2 -v N=20 {'avg=avg-(avg/N);avg=avg+($2/N);print $1,$2/100,avg/100;fflush();'}|awk 'NR>500;fflush();'|tee /dev/tty|awk {'print $1%10000,$2,$3;fflush();'}

device=$1
file=$2

# offset_x=0;
# offset_y=0;
# offset_z=0;
# scale_x=1;
# scale_y=1;
# scale_z=1;

# offset_x=29.159999999999997
# offset_y=10.0
# offset_z=-28.08
# scale_x=1.2296969696969697
# scale_y=1.0019753086419751
# scale_z=0.8412106135986733

offset_x=-250.0
offset_y=6083.0
offset_z=-3891.5
scale_x=0.967051282051282
scale_y=0.9335858210802515
scale_z=1.1175808219989924


~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 $device|ts "%.s"|awk -v offset_x="$offset_x" -v offset_y="$offset_y" -v offset_z="$offset_z" -v scale_x="$scale_x" -v scale_y="$scale_y" -v scale_z="$scale_z" {'print $1%10000,($2-offset_x)*scale_x,($3-offset_y)*scale_y,($4-offset_z)*scale_z,sqrt((($2-offset_x)*scale_x)*(($2-offset_x)*scale_x) + (($3-offset_y)*scale_y)*(($3-offset_y)*scale_y) + (($4-offset_z)*scale_z)*(($4-offset_z)*scale_z));fflush();'} |tee >(awk '{print $1,$2,$3,$4,$5;fflush();}'|feedgnuplot --lines --stream 0.01 --ylabel 'utesla' --xlabel 'time' --domain --xlen 50 --legend 0 "x" --legend 1 "y" --legend 2 "z" --legend 3 "m" --legend 4 "$device") > $file
# ~/contiki-magnetometer-nicholasinatel/contiki/tools/sky/serialdump-linux -b115200 $device|ts "%.s"|awk -v offset_x="$offset_x" -v offset_y="$offset_y" -v offset_z="$offset_z" -v scale_x="$scale_x" -v scale_y="$scale_y" -v scale_z="$scale_z" {'print $1%10000,($2-offset_x)*scale_x,($3-offset_y)*scale_y,($4-offset_z)*scale_z,sqrt((($2-offset_x)*scale_x)*(($2-offset_x)*scale_x) + (($3-offset_y)*scale_y)*(($3-offset_y)*scale_y) + (($4-offset_z)*scale_z)*(($4-offset_z)*scale_z));fflush();'} |tee >(awk '{print $1,$3,$5;fflush();}'|feedgnuplot --lines --stream 0.01 --ylabel 'utesla' --xlabel 'time' --domain --xlen 50 --legend 0 "y" --legend 1 "m" ) > test.cal
