#!/usr/bin/python3

import os
import itertools

from argparse import ArgumentParser
import subprocess

parser = ArgumentParser(description='check all...', epilog="?")
# parser.add_argument("-b", "--beacon", dest="b", required=True, action='append')
# parser.add_argument("-s", "--sensor", dest="s", required=True)
parser.add_argument("-f", "--file", dest="f", required=True)
parser.add_argument("-g", "--groundtruthfile", dest="g", required=True, help="ground truth file for positions of sensors")
parser.add_argument("-d", "--B3distance", dest="d", required=False, help="B3 distance, in cm, from initial delimited area")

args = parser.parse_args()
# print(args)

# b = args.b
# s = args.s
myfile = args.f
ground_truth_file = args.g

b3distace = int(args.d) if args.d != None else 0

b = []
for i in [1,2,3,4]:
	b.append('b' + str(i))

for i in range(10):
	s="s"+str(i+1)
	# print(s)
	for num_combination_elements in range(2,5):
		for combination in itertools.combinations(b, num_combination_elements):
			b_params = ""
			# print(len(combination), combination)
			for i in range(len(combination)):
				b_params += " -b " + combination[i]
			# print(b_params)
			# cmd = "./plot_sys_of_eqn.py " + b_params + " -f " + myfile + " -d " + str(b3distace) + " -s " + s + " -p 0"
			# ./plot_sys_of_eqn_dynamic_config.py -b b1 -b b2 -b b3 -b b4  -f experiment_20181013/RESULTS/20SEC/mydata.csv -g ground_truth_20181013.csv -p 1 -s s9
			cmd = "./plot_sys_of_eqn_dynamic_config.py " + b_params + " -f " + myfile + " -s " + s + " -g " + ground_truth_file + " -p 0"
			# cmd = "./plot_sys_of_eqn_dynamic_config_cluster_of_points.py " + b_params + " -f " + myfile + " -s " + s + " -g " + ground_truth_file + " -p 0"
			print(cmd)
			# print(cmd)
			# os.system(cmd)
			# cmd = ['awk', 'length($0) > 5']
			# input = 'foo\nfoofoo\n'.encode('utf-8')
			result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
			# result.stdout.decode('utf-8')
			# result = subprocess.check_output([cmd, inputs])
			print(result.stdout)




# cmd = "./plot_sys_of_eqn.py -b b4 -b b3 -f experiment_201810_040509_b3_plus45/RESULTS/20SEC/mydata.csv  -d 45 -s s3"
# os.system(cmd)
