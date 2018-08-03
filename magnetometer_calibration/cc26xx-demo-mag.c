/*
 * Copyright (c) 2014, Texas Instruments Incorporated - http://www.ti.com/
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*---------------------------------------------------------------------------*/
/**
 * \addtogroup cc26xx-platforms
 * @{
 *
 * \defgroup cc26xx-examples CC26xx Example Projects
 *
 * Example projects for CC26xx-based platforms.
 * @{
 *
 * \defgroup cc26xx-demo CC26xx Demo Project
 *
 *   Example project demonstrating the CC13xx/CC26xx platforms
 *
 *   This example will work for the following boards:
 *   - srf06-cc26xx: SmartRF06EB + CC13xx/CC26xx EM
 *   - CC2650 and CC1350 SensorTag
 *   - CC1310, CC1350, CC2650 LaunchPads
 *
 *   This is an IPv6/RPL-enabled example. Thus, if you have a border router in
 *   your installation (same RDC layer, same PAN ID and RF channel), you should
 *   be able to ping6 this demo node.
 *
 *   This example also demonstrates CC26xx BLE operation. The process starts
 *   the BLE beacon daemon (implemented in the RF driver). The daemon will
 *   send out a BLE beacon periodically. Use any BLE-enabled application (e.g.
 *   LightBlue on OS X or the TI BLE Multitool smartphone app) and after a few
 *   seconds the cc26xx device will be discovered.
 *
 * - etimer/clock : Every CC26XX_DEMO_LOOP_INTERVAL clock ticks the LED defined
 *                  as CC26XX_DEMO_LEDS_PERIODIC will toggle and the device
 *                  will print out readings from some supported sensors
 * - sensors      : Some sensortag sensors are read asynchronously (see sensor
 *                  documentation). For those, this example will print out
 *                  readings in a staggered fashion at a random interval
 * - Buttons      : CC26XX_DEMO_SENSOR_1 button will toggle CC26XX_DEMO_LEDS_BUTTON
 *                - CC26XX_DEMO_SENSOR_2 turns on LEDS_REBOOT and causes a
 *                  watchdog reboot
 *                - The remaining buttons will just print something
 *                - The example also shows how to retrieve the duration of a
 *                  button press (in ticks). The driver will generate a
 *                  sensors_changed event upon button release
 * - Reed Relay   : Will toggle the sensortag buzzer on/off
 *
 * @{
 *
 * \file
 *     Example demonstrating the cc26xx platforms
 */
#include "contiki.h"
#include "sys/etimer.h"
#include "sys/ctimer.h"
// #include "dev/leds.h"
#include "dev/watchdog.h"
#include "random.h"
// #include "button-sensor.h"
// #include "batmon-sensor.h"
#include "board-peripherals.h"
#include "net/netstack.h"
// #include "rf-core/rf-ble.h"

#include "ti-lib.h"
#include "sys/rtimer.h"

#include <stdio.h>
#include <stdint.h>

#include "contiki-conf.h"
#include "lib/sensors.h"
#include "mpu-9250-sensor.h"
#include "sys/rtimer.h"
#include "sensor-common.h"
#include "board-i2c.h"

#include "ti-lib.h"
#include "net/rime/rime.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*---------------------------------------------------------------------------*/
#define CC26XX_DEMO_LOOP_INTERVAL       (CLOCK_SECOND * 0.001) // MAX FOR MAGNETOMETER 120Hz
#define CC26XX_DEMO_LEDS_PERIODIC       LEDS_YELLOW
#define CC26XX_DEMO_LEDS_BUTTON         LEDS_RED
#define CC26XX_DEMO_LEDS_REBOOT         LEDS_ALL

#define delay_ms(i) (ti_lib_cpu_delay(8000 * (i)))

/*---------------------------------------------------------------------------*/
static struct etimer et;
/*---------------------------------------------------------------------------*/
PROCESS(cc26xx_demo_process, "cc26xx demo process");
AUTOSTART_PROCESSES(&cc26xx_demo_process);

PROCESS_THREAD(cc26xx_demo_process, ev, data)
{

  PROCESS_BEGIN();
  static int *m;
  printf("CC26XX demo\n");

  etimer_set(&et, CC26XX_DEMO_LOOP_INTERVAL);
  NETSTACK_RDC.init();
  NETSTACK_MAC.init();

  watchdog_init(); 
  watchdog_stop();

  NETSTACK_MAC.off(0);
  mpu_9250_sensor.configure(SENSORS_ACTIVE, MPU_9250_SENSOR_TYPE_MAG); //  option MAG implies ACC as default
  // printf("mpu 9250 configured!\n");
  static int count = 0;
  // int mag_values[5];
  while(1) {

    PROCESS_WAIT_EVENT();
    if(ev == PROCESS_EVENT_TIMER) {
      if(data == &et) {

        etimer_set(&et, CC26XX_DEMO_LOOP_INTERVAL);
        // mpu_9250_sensor.configure(SENSORS_ACTIVE, MPU_9250_SENSOR_TYPE_ACC); //  option MAG implies ACC as default
        // delay_ms(20);
        m = mpu_9250_sensor.value(MPU_9250_SENSOR_TYPE_MAG_ALL);
        if(*m != CC26XX_SENSOR_READING_ERROR){
            // count++;
            // if(count%10 == 0){
            //   count=0;
              // printf("\n");
              // mag_values[count] = 
              // printf("%d\n", count);

            // }
            printf("%d %d %d\n",*m, *(m+1), *(m+2));
            // printf("%d %d %d %d\n",*m, *(m+1), *(m+2), *(m+3));
        }else{
          printf("error,%d\n",*m);
        }               
        // SENSORS_DEACTIVATE(mpu_9250_sensor);
      }
    } 
  }
  // deactivate version - lower power
  /*
    while(1) {
    PROCESS_YIELD();
    if(ev == PROCESS_EVENT_TIMER) {
      if(data == &et) {

       etimer_set(&et, CC26XX_DEMO_LOOP_INTERVAL);
       mpu_9250_sensor.configure(SENSORS_ACTIVE, MPU_9250_SENSOR_TYPE_MAG);
     }
   }
    else if(ev == sensors_event) {
        if(ev == sensors_event && data == &mpu_9250_sensor) {
          m = mpu_9250_sensor.value(MPU_9250_SENSOR_TYPE_MAG_ALL);
                 if(m != CC26XX_SENSOR_READING_ERROR){
                    printf("%d\n",m);
                 }else{
                   printf("error,%d\n",m);
                 }          
       
       SENSORS_DEACTIVATE(mpu_9250_sensor);
      }
    } 
  }*/
  PROCESS_END();
}
