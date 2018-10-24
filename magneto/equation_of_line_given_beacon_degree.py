import math

def equation_of_line_given_beacon_degree(beacon, theta, b3distance=0):
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
		# m=math.tan(math.radians((270 - theta)))
		# c=m*(-250+b3distance)
		# m=math.tan(math.radians(180 + theta))
		# c=-math.tan(math.radians(180 + theta)) * 250

		# new layout
		m=math.tan(math.radians((90 - theta) % 360 ))
		# c=m*250
		c=m*(-250+b3distance)
	elif beacon=='b4':
		m=math.tan(math.radians(180 - theta))
		c=250
	else:
		m=0
		c=0
	# print("y=%.2fx+%.2f"%(m,c))
	return m,c

def find_intercept_given_beacon_and_gradient(beacon, m, b3distance=0):
	if beacon=='b1':
		c=m*250
	elif beacon=='b2': 
		c=-250
	elif beacon=='b3':
		# c=m*(-250+b3distance)
		# new layout
		c=m*(-250+b3distance)
	elif beacon=='b4':
		c=250
	# else:
	# 	#m==0
	# 	c=0
	# print("y=%.2fx+%.2f"%(m,c))
	return c