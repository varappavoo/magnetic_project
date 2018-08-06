#!/usr/bin/python3
import numpy as np
from argparse import ArgumentParser
import sys
import math
import pickle



# calibration_data = {}
# calibration_data['emtpy'] = None
record={}
# with open('calibration_data.pickle', 'wb') as f:
# 	pickle.dump(calibration_data,f)


parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f", "--file", dest="file", help="", required=True)
# parser.add_argument("-f2", "--filecal", dest="file_cal", help="", required=True)
args = parser.parse_args()

data = np.loadtxt(args.file, delimiter=' ')
print(args.file)

t = data[:,0]
x = data[:,1]
y = data[:,2]
z = data[:,3]


#############################################
## HARD IRON BIAS
## "interference can be caused by ferromagnetic material or equipment in the magnetometers vicinity"
## src file:///home/vara/Zotero/storage/G9BXI88X/calibrate-magnetometer.html ????
###
#############################################
offset_x = (np.max(x) + np.min(x))/2
offset_y = (np.max(y) + np.min(y))/2
offset_z = (np.max(z) + np.min(z))/2

print("offset_x=" + str(offset_x))
print("offset_y=" + str(offset_y))
print("offset_z=" + str(offset_z))
# data[:,1] = x - offset_x
# data[:,2] = y - offset_y
# data[:,3] = z - offset_z  
record["offset_x"]= offset_x
record["offset_y"]= offset_y
record["offset_z"]= offset_z


###########################################
## SOFT IRON BIAS
## Soft iron distortion is the result of material that distorts a magnetic field but does not necessarily generate a magnetic field itself. For example iron (the metal) will generate a distortion but this distorion is dependent upon the orientation of the material relative to the magnetometer.
## Unlike hard iron distortion, soft iron distortion cannot be removed by simply removing the constant offset. Correcting soft iron distortion is usually more computation expensive and involves 3x3 transformation matrix.
## There is also a computatively cheaper way by using scale biases as explained by Kris Winer. This method should also give reasonably good results. Example pseudocode below includes also the hard iron offset from the previous step.
## src file:///home/vara/Zotero/storage/G9BXI88X/calibrate-magnetometer.html ????
###########################################
avg_delta_x = (np.max(x) - np.min(x))/2
avg_delta_y = (np.max(y) - np.min(y))/2
avg_delta_z = (np.max(z) - np.min(z))/2

avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3

scale_x = avg_delta / avg_delta_x
scale_y = avg_delta / avg_delta_y
scale_z = avg_delta / avg_delta_z

print("scale_x=" + str(scale_x))
print("scale_y=" + str(scale_y))
print("scale_z=" + str(scale_z))

record["scale_x"]= scale_x
record["scale_y"]= scale_y
record["scale_z"]= scale_z

data[:,1] = (x - offset_x) * scale_x
data[:,2] = (y - offset_y) * scale_y
data[:,3] = (z - offset_z) * scale_z

data[:,4] = np.sqrt((data[:,1]*data[:,1]) + (data[:,2]*data[:,2]) + (data[:,3]*data[:,3]))

# np.savetxt(args.file +'.cal', data, delimiter=' ', fmt="%.2f %.2f %.2f %.2f %.2f",newline='\n')

try:
	with open('calibration_data.pickle', 'rb') as f:
		calibration_data = pickle.load(f)
		print(calibration_data)
except:
	print("calibration_data.pickle does not exist.")
	calibration_data = {}
	
calibration_data[args.file[:len(args.file)-5]]=record
print(calibration_data)
print(calibration_data.keys())

with open('calibration_data.pickle', 'wb') as f:
	pickle.dump(calibration_data,f)