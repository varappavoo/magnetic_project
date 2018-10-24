#!/usr/bin/python3

file = "beacons_validity.txt"

sensor_valid_beacons = {}
# cluster_distance_ideal_center = 400
# cluster_distance_bound = 10

with open(file, "r") as f:
	line = f.readline()
	while(line != ""):
		line = line.split(",")
		print(line)
		if(sensor_valid_beacons.get(line[1]) == None):
			sensor_valid_beacons[line[1]] = []
		if(float(line[2]) >= 370 and float(line[2]) <= 430) :
			sensor_valid_beacons[line[1]].append(line[0])
		line = f.readline()

for sensor in sensor_valid_beacons:
	print(sensor + "," + str(sensor_valid_beacons[sensor]).replace("[","\"").replace("]","\""))
# print(sensor_valid_beacons)