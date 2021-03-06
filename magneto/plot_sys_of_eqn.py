#!/usr/bin/python3

import numpy as np
import matplotlib.patches as patches
# import matplotlib.pyplot as plt
from pylab import *
# from math import *
from argparse import ArgumentParser

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-b", "--beacon", dest="b", required=True, action='append')
parser.add_argument("-s", "--sensor", dest="s", required=True)
parser.add_argument("-f", "--file", dest="f", required=True)
parser.add_argument("-d", "--B3distance", dest="d", required=False, help="B3 distance, in cm, from initial delimited area")

parser.add_argument("-p", "--plot", dest="p", required=False, help="plot, 1, or not, 0")

# parser.add_argument("-f2", "--file2", dest="file2", help="use filename of stag that need to be localized, (for eg. s1_test_1.dat", required=True)

# parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
# parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

args = parser.parse_args()
# print(args)
# file1 = args.experiment
b = args.b
s = args.s
myfile = args.f
b3distace = int(args.d) if args.d != None else 0
plot_it = int(args.p) if args.p != None else 1


def solve_sys_of_eqn(A, B):
	x = np.linalg.lstsq(A, B, rcond=-1)
	u_y = x[0][0]
	u_x = x[0][1]
	return u_x, u_y

def equation_of_line_given_beacon_degree(beacon, theta):
	if beacon=='b1':
		# m=math.tan(math.radians(theta))
		m=math.tan(math.radians((90 - theta) % 360 ))
		# m=math.tan(math.radians(270 - theta))
		# c=-math.tan(math.radians(theta)) * -250
		# c=250
		c=m*250
	elif beacon=='b2':
		# m=math.tan(math.radians(270 + theta)) 
		m=math.tan(math.radians(180-theta)) 
		c=-250
	elif beacon=='b3':
		m=math.tan(math.radians((270 - theta)))
		c=m*(-250+b3distace)
		# m=math.tan(math.radians(180 + theta))
		# c=-math.tan(math.radians(180 + theta)) * 250
	elif beacon=='b4':
		m=math.tan(math.radians(180 - theta))
		c=250
	else:
		m=0
		c=0
	# print(m,c)
	return m,c

fig = plt.figure()
ax = fig.add_subplot(111)
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

sx=[0,-50, 100, 0,   150, 100,  50, 200, -200,-200]
sy=[0, 50, 50,  150, 150,-100,-150, -200,-100, 250]
colors={}
colors['b1']='tomato'
colors['b2']='blue'
colors['b3']='olive'
colors['b4']='purple'

bx=[-250	,0		,250-b3distace	,0]
by=[0		,-250	,0		,250]

A=[]
B=[]


count_beacons={}

for i in range(4):
	ax.text(bx[i], by[i], "B"+str(i+1))
	count_beacons['b'+str(i+1)]=0

mydata = np.genfromtxt(myfile, delimiter=',', dtype=None, names=('beacon', 'u_sensor', 'b_sensor', 'tstart', 'tend', 'degree'))
for i in range(0,len(mydata)):
	beacon = str(mydata[i]['beacon'])
	sensor = str(mydata[i]['u_sensor'])
	beacon = beacon[2:4]
	sensor = sensor[2:sensor.rfind("'")]

	# print(beacon,sensor, mydata[i])
	# if(beacon!=''):
		

	if(beacon in b and sensor==s):
		m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'])
		ax.plot(x, (m*x)+c, alpha=0.4, color=colors[beacon])#, label="y = x**2")
		A.append([1, -m])
		B.append(c)
		count_beacons[beacon] = count_beacons[beacon] + 1

	# elif(b=="all" and sensor==s):
	# 	if(beacon in ['b1','b2','b3','b4']):
	# 		m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'])
	# 		ax.plot(x, (m*x)+c, alpha=0.4, color=colors[beacon])#, label="y = x**2")

	# 		A.append([1, -m])
	# 		B.append(c)
	# 		count_beacons[beacon] = count_beacons[beacon] + 1

print("A\n",A,"\nB\n",B)
# if(b=="all" and sensor==s):
# x = np.linalg.lstsq(A, B, rcond=-1)
# u_y = x[0][0]
# u_x = x[0][1]
# # print(x)
# print(x,x[0])

u_x, u_y = solve_sys_of_eqn(A, B)

ax.scatter(sx,sy,marker='x', color='blue')
ax.scatter(bx,by,marker='o', color='green')

for i in range(10):
	ax.text(sx[i]+10, sy[i]-10, "s"+str(i+1))




ax.scatter(np.array([u_x]),np.array([u_y]),marker='*', color='red')#,c=[100])
ax.text(np.array([u_x])+10,np.array([u_y])+10, s, color='red')#,c=[100])

print(count_beacons)
sensorid = int(s[1:]) - 1
print(sensorid)
actual_x = sx[sensorid]
actual_y = sy[sensorid]
print("RESULTS,%s,%s,actual,(%d %d),computed,(%d %d),%d"% (s,str(b),actual_x, actual_y, u_x, u_y, math.sqrt((actual_y - u_y)**2 + (actual_x - u_x)**2)))

if plot_it:
	plt.show()

