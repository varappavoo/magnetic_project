#!/usr/bin/python3

import numpy as np
import matplotlib.patches as patches
import scipy
# import scipy.linalg as linalg 
import numpy.linalg as linalg
# import matplotlib.pyplot as plt
from pylab import *
# from math import *
from argparse import ArgumentParser
import traceback
from equation_of_line_given_beacon_degree import equation_of_line_given_beacon_degree, find_intercept_given_beacon_and_gradient

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-b", "--beacon", dest="b", required=True, action='append')
parser.add_argument("-s", "--sensor", dest="s", required=True)
parser.add_argument("-f", "--file", dest="f", required=True)
parser.add_argument("-g", "--groundtruthfile", dest="g", required=True, help="ground truth file for positions of sensors")
parser.add_argument("-d", "--Bb3distance", dest="d", required=False, help="B3 distance, in cm, from initial delimited area")

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
ground_truth_file = args.g
b3distance = int(args.d) if args.d != None else 0
plot_it = int(args.p) if args.p != None else 1

# print("b",b)

sx=[]#[0,-50, 100, 0,   150, 100,  50, 200, -200,-200]
sy=[]#[0, 50, 50,  150, 150,-100,-150, -200,-100, 250]
count_beacons={}

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(myfile)

x = linspace(-400, 400, 800)
limit=400
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
MARGIN_OUTSIDE_AREA = 50
bx=[-MAX_DISTANCE	,0		,MAX_DISTANCE-b3distance	,0]
by=[0		,-MAX_DISTANCE	,0		,MAX_DISTANCE]

cluster_by_beacon_pairs = {}

def check_beacons_pair(pair_cluster_of_intersected_points):
	pair_cluster_of_intersected_points = np.array(pair_cluster_of_intersected_points)
	# print("check beacon pair")
	# print("std x: ",np.std(pair_cluster_of_intersected_points[:,0]))
	# print("std y: ", np.std(pair_cluster_of_intersected_points[:,1]))
	# print("std dev,std x,", np.std(pair_cluster_of_intersected_points[:,0]),"std y,",np.std(pair_cluster_of_intersected_points[:,1]))
	return np.std(pair_cluster_of_intersected_points[:,0]), np.std(pair_cluster_of_intersected_points[:,1])


def solve_sys_of_eqn(A, B, lstsq=1):
	# x = np.linalg.lstsq(A, B, rcond=-1)
	# print(A)
	# print(B)

	# print(x)
	if(lstsq):
		x = linalg.lstsq(A, B, rcond=-1)
		# x = linalg.tensorsolve(A, B)
		# LU = scipy_linalg.lu_factor(A) 
		# x = scipy_linalg.lu_solve(LU, B) 
		# print(x)
		u_x = x[0][0]
		u_y = x[0][1]
	else:
		x = np.linalg.solve(A, B)
		u_x = x[0]
		u_y = x[1]
		# x = scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), B)
		# print("######################")
	# print(x)

	return u_x, u_y

# def equation_of_line_given_beacon_degree(beacon, theta):
# 	if beacon=='b1':
# 		# m=math.tan(math.radians(theta))
# 		m=math.tan(math.radians((90 - theta) % 360 ))
# 		# m=math.tan(math.radians(270 - theta))
# 		# c=-math.tan(math.radians(theta)) * -250
# 		# c=250
# 		c=m*250
# 	elif beacon=='b2':
# 		# m=math.tan(math.radians(270 + theta)) 
# 		m=math.tan(math.radians(180-theta)) 
# 		c=-250
# 	elif beacon=='b3':
# 		m=math.tan(math.radians((270 - theta)))
# 		c=m*(-250+b3distance)
# 		# m=math.tan(math.radians(180 + theta))
# 		# c=-math.tan(math.radians(180 + theta)) * 250
# 	elif beacon=='b4':
# 		m=math.tan(math.radians(180 - theta))
# 		c=250
# 	else:
# 		m=0
# 		c=0
# 	# print("y=%.2fx+%.2f"%(m,c))
# 	return m,c



def compute_sensor_location(b ,s, myfile, b3distance=0, plot_it=0, lstsq=0, gt=0, iterate_through_all_combinations=0):
	A=[]
	B=[]
	M={} # gradients
	D={} # degrees

	for beacon in b:
		M[beacon] = []
		D[beacon] = []

	for i in range(4):
		ax.text(bx[i], by[i], "B"+str(i+1))
		count_beacons['b'+str(i+1)]=0

	mydata = np.genfromtxt(myfile, delimiter=',', encoding='bytes', dtype=None, names=('beacon', 'u_sensor', 'b_sensor', 'tstart', 'tend', 'degree'))
	print(mydata)
	for i in range(0,len(mydata)):
		beacon = str(mydata[i]['beacon'])
		sensor = str(mydata[i]['u_sensor'])
		beacon = beacon[2:4]
		sensor = sensor[2:sensor.rfind("'")]

		if(beacon in b and sensor==s):

			m,c = equation_of_line_given_beacon_degree(beacon, mydata[i]['degree'], b3distance)
			M[beacon].append(m)
			D[beacon].append(math.degrees(math.atan(m)))

			if(plot_it):
				ax.plot(x, (m*x)+c, alpha=0.4, color=colors[beacon])#, label="y = x**2")

			count_beacons[beacon] = count_beacons[beacon] + 1

	# # print("b",b)
	if gt:
		for beacon in b:
			if M[beacon] != []:
				# ONLY ONE ITEM NOW; MISLEADING MEDIAN IS NOT NECESSAY
				# print("###########GT M[beacon]:", M[beacon])
				m = np.median(M[beacon])
				# m=M[beacon][int(len(M[beacon])/2)]
				c = find_intercept_given_beacon_and_gradient(beacon, m, b3distance)
	
				if(plot_it):
					ax.plot(x, (m*x)+c, alpha=1, linewidth=3, linestyle=":", color=colors[beacon])#, label="y = x**2")
	
				# if(m!=0):
				# 	A.append([1, -1/m])
				# 	B.append(-c/m)
				# else:
				# 	A.append([1,0])
				# 	B.append(0)

				A.append([-m,1])
				B.append(c)
		# if(m == 0):
		# 	u_y = c
		# 	u_x = 
		u_x, u_y = solve_sys_of_eqn(np.array(A), np.array(B), lstsq)
		
	elif iterate_through_all_combinations:
		pass
	else:

		cluster_of_intersected_points = []
		

		processed_beacons = []
		for beacon1 in b:
			processed_beacons.append(beacon1)
			for beacon2 in b:
				if beacon2 not in processed_beacons:
					if cluster_by_beacon_pairs.get(beacon1) == None:
						cluster_by_beacon_pairs[beacon1]={}
					cluster_by_beacon_pairs[beacon1][beacon2]=[]

		# print(cluster_by_beacon_pairs)


		processed_beacons = []
		for beacon1 in b:
			processed_beacons.append(beacon1)
			for beacon2 in b:
				if beacon1 != beacon2 and M[beacon1] != [] and M[beacon2] != [] and beacon2 not in processed_beacons:
					for m1 in M[beacon1]:
						for m2 in M[beacon2]:
							# print(beacon1, beacon2, m1, m2)
							A=[]
							B=[]
							c1 = find_intercept_given_beacon_and_gradient(beacon1, m1, b3distance)
							c2 = find_intercept_given_beacon_and_gradient(beacon2, m2, b3distance)
							if(m1!=0):
								A.append([1, -1/m1]) # divide by gradient; format x:y
								B.append(-c1/m1)
							else:
								A.append([0,1]) # m==0, x=0,y=1  
								B.append(0)
							if(m2!=0):
								A.append([1, -1/m2])
								B.append(-c2/m2)
							else:
								A.append([0,1])
								B.append(0)

							print("A\n",A,"\nB\n",B)
							try:
								u_x, u_y = solve_sys_of_eqn(np.array(A), np.array(B), lstsq=0) # neither over- or under- determined
								print("Solution: ",u_x,",",u_y)

								if( ((abs(u_x) <= MAX_DISTANCE + MARGIN_OUTSIDE_AREA and abs(u_y) <= MAX_DISTANCE + MARGIN_OUTSIDE_AREA) and EXCLUDE_POINTS_OUTSIDE_AREA) or not EXCLUDE_POINTS_OUTSIDE_AREA):
									cluster_by_beacon_pairs[beacon1][beacon2].append([u_x, u_y])
									cluster_of_intersected_points.append([u_x,u_y])
									ax.scatter(u_x,u_y,marker='+', color='green')

							except:
								# pass
								print(traceback.print_exc())
								# print("ERROR SOLVING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
							# print("A\n",A,"\nB\n",B)

		cluster_of_intersected_points = np.array(cluster_of_intersected_points)

		# print("CLUSTER#############################")
		# print(cluster_of_intersected_points[:,0])
		# print(cluster_of_intersected_points[:,1])
		# m_x = np.mean(cluster_of_intersected_points[:,0])
		# m_y	= np.mean(cluster_of_intersected_points[:,1])			

		print("cluster_of_intersected_points\n",	len(cluster_of_intersected_points[:,0]), cluster_of_intersected_points)
		m_x = np.median(cluster_of_intersected_points[:,0])
		m_y	= np.median(cluster_of_intersected_points[:,1])			


		ax.scatter(m_x,m_y,marker='o', color='yellow',s=400)
		u_x = m_x
		u_y = m_y
			# scipy.cluster.vq.kmeans2()
			# m = np.median(M[beacon])
			# # m=M[beacon][int(len(M[beacon])/2)]
			# c = find_intercept_given_beacon_and_gradient(beacon, m)
			# if(not gt):
			# 	print(beacon,",",M[beacon])
			# 	print(s,",",beacon, "min:", np.min(M[beacon]))
			# 	print(s,",",beacon, "max:", np.max(M[beacon]))
			# 	print(s,",",beacon, "std dev:", np.std(M[beacon]))
			# 	print(s,",",beacon, "variance:", np.var(M[beacon]))
			# # print("m",m,"c",c)
			# # if(m != 0):
			# # 	m = m/c
			# # 	c = 1
			# if(plot_it):
			# 	ax.plot(x, (m*x)+c, alpha=1, linewidth=3, linestyle=":", color=colors[beacon])#, label="y = x**2")
			# # A.append([1, -m])
			# # B.append(c)
			# if(m!=0):
			# 	A.append([1, -1/m])
			# 	B.append(-c/m)
			# else:
			# 	A.append([1,0])
			# 	B.append(0)

	# print("A\n",A,"\nB\n",B)
	# if(b=="all" and sensor==s):
	# x = np.linalg.lstsq(A, B, rcond=-1)
	# u_y = x[0][0]
	# u_x = x[0][1]
	# # print(x)
	# print(x,x[0])
	# print("######################")
	# print(linalg.cond(A))

	# u_x, u_y = solve_sys_of_eqn(np.array(A), np.array(B), lstsq)
	# print("\n\nM\n",M,"\n")
	# print("\n\nD\n",D,"\n")
	return u_x, u_y

def plot(b, s, ax, sx, sy, u_x, u_y):


	for i in range(10):
		ax.text(sx[i]+10, sy[i]-10, "s"+str(i+1))
		if(s == 's'+str(i+1)):
			ax.scatter(sx[i],sy[i],marker='o', color='pink',s=400, alpha=1)


	ax.scatter(sx,sy,marker='x', color='blue')
	ax.scatter(bx,by,marker='o', color='green')


	ax.scatter(np.array([u_x]),np.array([u_y]),marker='*', color='red')#,c=[100])
	ax.text(np.array([u_x])+10,np.array([u_y])+10, s, color='red')#,c=[100])

	# print(count_beacons)
	sensorid = int(s[1:]) - 1
	# print(sensorid)
	actual_x = sx[sensorid]
	actual_y = sy[sensorid]
	print("RESULTS,%s,%s,actual,(%d %d),computed,(%d %d),%d"% (s,str(b),actual_x, actual_y, u_x, u_y, math.sqrt((actual_y - u_y)**2 + (actual_x - u_x)**2)))




def compute_ground_truth(ground_truth_file):
	for i in range(10):
		s = "s" + str(i+1)
		# print("###################",s)
		u_x, u_y = compute_sensor_location(['b1','b2','b3','b4'] ,s, ground_truth_file, b3distance=b3distance, plot_it=0, gt=1, lstsq=0)
		sx.append(u_x)
		sy.append(u_y)

	b_dis_all = []
	for i in range(10):
		b_dis = []
		s = "s" + str(i+1)
		# print("DISTANCE,",s,end="")
		for j in [1,2,3,4]:#range(4):
			distance_b_s = math.sqrt((sx[i] - bx[j-1])**2 + (sy[i] - by[j-1])**2)
			# print(",b"+ str(j),",",round(distance_b_s),end="")
			b_dis.append(distance_b_s)
		# print(",",sort(b_dis))
		b_dis_all.append(sort(b_dis))
	# print("DISTANCE ALL:",b_dis_all,"\n\n")			
	b_dis_all = np.array(b_dis_all)
	# print(b_dis_all[:,0])
	# print(b_dis_all[:,1])
	np.savetxt("b_distance.csv", b_dis_all, delimiter=",")


compute_ground_truth(ground_truth_file)
# if plot_it:
# 	plt.show()
# 	plt.clf()
# 	plt.cla()
# 	plt.close()
# ax.clear()
# plt.gcf().clear()
# plt.sh
# print("b",b)
u_x, u_y = compute_sensor_location(b ,s, myfile, b3distance, plot_it=1, lstsq=0)

# print(sx)
# print(sy)

processed_beacons = []
for beacon1 in b:
	processed_beacons.append(beacon1)
	for beacon2 in b:
		if beacon2 not in processed_beacons:
			std_dev_x, std_dev_y = check_beacons_pair(cluster_by_beacon_pairs[beacon1][beacon2])
			print("std dev,",beacon1,",", beacon2, "," , std_dev_x, ",", std_dev_y)

plot(b, s, ax, sx, sy, u_x, u_y)
if plot_it:
	plt.show()
