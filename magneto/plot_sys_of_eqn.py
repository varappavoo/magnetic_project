#!/usr/bin/python3

import numpy as np
# import matplotlib.pyplot as plt
from pylab import *
# from math import *
from argparse import ArgumentParser

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-b", "--beacon", dest="b", required=True)
parser.add_argument("-s", "--sensor", dest="s", required=True)

# parser.add_argument("-f2", "--file2", dest="file2", help="use filename of stag that need to be localized, (for eg. s1_test_1.dat", required=True)

# parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
# parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

args = parser.parse_args()
# file1 = args.experiment
b = args.b
s = args.s


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
		m=math.tan(math.radians((270 - theta) % 360 ))
		c=m*-250
		# m=math.tan(math.radians(180 + theta))
		# c=-math.tan(math.radians(180 + theta)) * 250
	elif beacon=='b4':
		m=math.tan(math.radians(180 - theta))
		c=250
	else:
		m=0
		c=0
	print(m,c)
	return m,c

fig = plt.figure()
ax = fig.add_subplot(111)
x = linspace(-250, 250, 500)
ax.set_ylim(-250,250)
ax.set_xlabel('x')
ax.set_ylabel('y')
sx=[0,-50, 100, 0,   150, 100,  50, 200, -200,-200]
sy=[0, 50, 50,  150, 150,-100,-150, -200,-100, 250]



mydata = np.genfromtxt('experiment_201810_0405/RESULTS/mydata.csv', delimiter=',', dtype=None, names=('beacon', 'u_sensor', 'b_sensor', 'tstart', 'tend', 'degree'))
for i in range(0,len(mydata)):
	beacon = str(mydata[i]['beacon'])
	sensor = str(mydata[i]['u_sensor'])
	beacon = beacon[2:4]
	sensor = sensor[2:sensor.rfind("'")]
	if(beacon==b and sensor==s):
		print(mydata[i])
		m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'])
		ax.plot(x, (m*x)+c, alpha=0.4)#, label="y = x**2")
		# print(str(mydata[i]['beacon'])[2:4])
		# print(str(mydata[i]['beacon'])[2:4]=='b1')
	elif(b=="all" and sensor==s):
		print(mydata[i])
		m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'])
		ax.plot(x, (m*x)+c, alpha=0.4)#, label="y = x**2")

ax.scatter(sx,sy)
plt.show()



# ax.plot(x, x**2, label="y = x**2")
# ax.plot(x, x**3, label="y = x**3")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('title')
# ax.legend(loc=2); # upper left corner