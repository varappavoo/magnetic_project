#!/bin/bash
experiment=$1
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.11__Embed_with_CMSIS-DAP_L1498-if00 "s0_1_0_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.11__Embed_with_CMSIS-DAP_L1188-if00 "s0_1_1_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.12__Embed_with_CMSIS-DAP_L237-if00 "s0_2_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.12__Embed_with_CMSIS-DAP_L3002818-if00 "s0_3_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.12__Embed_with_CMSIS-DAP_L37-if00 "s0_4_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.12__Embed_with_CMSIS-DAP_L415-if00 "s0_5_$experiment.dat" 2>/dev/null &
./run.sh /dev/serial/by-id/usb-Texas_Instruments_XDS110__02.03.00.12__Embed_with_CMSIS-DAP_L249-if00 "s1_$experiment.dat" 2>/dev/null &
# ./run.sh /dev/ttyACM13 "s2_$experiment.dat" 2>/dev/null &
# ./run_phone.sh "s1_$experiment.dat" 2>/dev/null &
#watch -n 1 'ls *dat -l'
