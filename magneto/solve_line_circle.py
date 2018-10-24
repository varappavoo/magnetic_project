#!/usr/bin/python3
import math 

from sympy import Symbol
from sympy import *
import numpy as np
from pylab import *
import matplotlib.patches as patches
import scipy
from scipy import cluster

from scipy.spatial import distance_matrix


from argparse import ArgumentParser
import traceback
from equation_of_line_given_beacon_degree import equation_of_line_given_beacon_degree

from sympy.solvers import solve

parser = ArgumentParser(description='solve...?', epilog="?")
parser.add_argument("-b", "--beacon", dest="b", required=True, action='append')
parser.add_argument("-s", "--sensor", dest="s", required=True)
parser.add_argument("-f", "--file", dest="f", required=True)
parser.add_argument("-d", "--Bb3distance", dest="d", required=False, help="B3 distance, in cm, from initial delimited area")
# m = 1
# # equation of line y = mx
# beacon_x = 0
# beacon_y = 0

# radius = 100

# r**2 = (x1 - xc)**2 + (y1 - yc)**2

# intersection of line = math.sqrt((x - beacon_x)**2 + (y - beacon_y)**2)
# 
# b=[]
# for i in range(1,5):
# 	b.append('b' + str(i))


args = parser.parse_args()

b = args.b
s = args.s
myfile = args.f
b3distance = int(args.d) if args.d != None else 0

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(myfile)

x = linspace(-300, 300, 600)
limit=300
ax.set_xlim(-limit,limit)
ax.set_ylim(-limit,limit)
ax.set_xlabel('x')
ax.set_ylabel('y')

rect = patches.Rectangle((-250,-250),500,500,linewidth=1,edgecolor='black',facecolor='none',linestyle=":")
ax.add_patch(rect)
ax.set_aspect('equal')
# ax.axis('equal')
# su=[]

colors={}
colors['b1']='tomato'
colors['b2']='blue'
colors['b3']='olive'
colors['b4']='purple'

EXCLUDE_POINTS_OUTSIDE_AREA = 1
MAX_DISTANCE = 250
bx=[-MAX_DISTANCE	,0		,MAX_DISTANCE-b3distance	,0]
by=[0		,-MAX_DISTANCE	,0		,MAX_DISTANCE]

def line_points_intercepting_circle( gradient, beacon_x=0, beacon_y=0, radius=100):
	x = Symbol('x')
	x_solved = solve( (x-beacon_x)**2 + (gradient*x-beacon_y)**2 - radius**2 , Real=True)
	x_a = x_solved[0].evalf()
	x_b = x_solved[1].evalf()
	return x_a, gradient*x_a, x_b, gradient*x_b

def adjust_point(x, y, beacon):
	beacon_index = int(beacon[1:]) - 1
	x = x + bx[beacon_index]
	y = y + by[beacon_index]
	return x,y

def main(plot_it=0):
	# myfile = "experiment_20181013/RESULTS/15SEC/mydata_15s_0_225.csv"
	intersection_points_x = []
	intersection_points_y = []
	intersection_points = []
	mydata = np.genfromtxt(myfile, delimiter=',', dtype=None, names=('beacon', 'u_sensor', 'b_sensor', 'tstart', 'tend', 'degree'))
	for i in range(0,len(mydata)):
		beacon = str(mydata[i]['beacon'])
		sensor = str(mydata[i]['u_sensor'])
		beacon = beacon[2:4]
		sensor = sensor[2:sensor.rfind("'")]

		# print(beacon,sensor, mydata[i])
		# if(beacon!=''):
			
		if(beacon in b and sensor==s):
			# print(mydata[i])
			m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'], b3distance)
			# print(line_points_intercepting_circle(bx[beacon_index], by[beacon_index], m, radius=100))
			x1,y1, x2,y2 = line_points_intercepting_circle(m, radius=200)
			# print(line_points_intercepting_circle(m, radius=200))
			# print((bx[beacon_index], by[beacon_index], m))

			# M[beacon].append(m)
			# D[beacon].append(math.degrees(math.atan(m)))

			# if(plot_it):
			ax.plot(x, (m*x)+c, alpha=0.4, color=colors[beacon])#, label="y = x**2")
			x1,y1 = adjust_point(x1, y1, beacon)
			x2,y2 = adjust_point(x2, y2, beacon)
			# print((x1,y1),(x2,y2))
			ax.scatter(x1, y1,marker='x', color='blue')
			ax.scatter(x2, y2,marker='x', color='blue')

			# if(abs(x1) <= 250 and abs(y1) <= 250):
			intersection_points_x.append(x1)
			intersection_points_y.append(y1)
			intersection_points.append([x1,y1])

			# if(abs(x2) <= 250 and abs(y2) <= 250):
			intersection_points_x.append(x2)
			intersection_points_y.append(y2)
			intersection_points.append([x2,y2])

			# # A.append([1, -m])
			# # B.append(c)

			# if(m!=0):
			# 	A.append([1, -1/m])
			# 	B.append(-c/m)
			# else:
			# 	A.append([1,0])
			# 	B.append(0)
			# count_beacons[beacon] = count_beacons[beacon] + 1


	intersection_points_x = np.array(intersection_points_x, dtype=np.float64)
	intersection_points_y = np.array(intersection_points_y, dtype=np.float64)
	intersection_points = np.array(intersection_points, dtype=np.float64)

	# print(intersection_points)
	
	cluster_data = cluster.vq.kmeans2(intersection_points, 2)
	# print(cluster_data)
	# print(cluster_data[1])
	# print(cluster_data[0][0])
	cluster_distance = math.sqrt((cluster_data[0][0][0] - cluster_data[0][1][0])**2 + (cluster_data[0][0][1] - cluster_data[0][1][1])**2)


	cluster0_x=[]
	cluster0_y=[]
	cluster1_x=[]
	cluster1_y=[]

	for i in range(len(cluster_data[1])):
		if cluster_data[1][i] == 0:
			cluster0_x.append(intersection_points_x[i])
			cluster0_y.append(intersection_points_y[i])
		else:
			cluster1_x.append(intersection_points_x[i])
			cluster1_y.append(intersection_points_y[i])


	# print("x", intersection_points_x)
	cluster0_x = np.array(cluster0_x)
	cluster0_y = np.array(cluster0_y)
	cluster1_x = np.array(cluster1_x)
	cluster1_y = np.array(cluster1_y)

	print(str(b[0]) + "," + str(s) + "," + str(round(cluster_distance,2)) + "," +str(round(np.std(cluster0_x),2)) + "," + str(round(np.std(cluster1_x) ,2)) + "," + str(round(np.std(cluster0_y),2)) + "," + str(round(np.std(cluster1_y),2)))

	plt.show()

main()