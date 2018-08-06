#!/bin/bash
rm calibration_data.pickle
ls *_.dat|xargs -n 1 ./calibrate.py -f