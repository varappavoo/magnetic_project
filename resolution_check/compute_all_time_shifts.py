#!/usr/bin/python3
from prettytable import PrettyTable
from argparse import ArgumentParser
from termcolor import colored
import numpy as np

from compute_time_shift import compute_time_shift

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f", "--file1", dest="experiment", help="use filename of stag that need to be localized, (for eg. s1_test_1.dat", required=True)
# parser.add_argument("-f2", "--file2", dest="file2", help="use filename of stag that need to be localized, (for eg. s1_test_1.dat", required=True)
parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

args = parser.parse_args()
# file1 = args.experiment
truncate_start = int(args.truncate_start)
truncate_end =  int(args.truncate_end)

# file1 = args.file1
# file2 = args.file2

# print(colored(compute_time_shift(file1,file2,truncate_start,truncate_end, show_plot= False),'yellow'))

file_extension = args.experiment[2:]
unknown_stags = [args.experiment[:2]]
# unknown_stags =['s0_1_1']

truncate_start = 0 if(args.truncate_start == None) else  int(args.truncate_start)
truncate_end = 0 if(args.truncate_end == None) else  int(args.truncate_end)

# CW
ground_truth_stags = ['s0_1','s0_2','s0_3','s0_4','s0_5']

ground_truth_diff = np.array([	[0,		45, 	90,		135,	180],\
								[-45,	0,		45,		90,		135],\
								[-90,	-45,	0,		45,		90],\
								[-135,	-90,	-45,	0,		45],\
								[-179.9, -135,	-90,	-45,	0],\
							])


ground_truth_stags_data = {}
unknown_stags_data = {}
time_shift_with_unknown = {}
center = 's0_3'
max_degree_possible_from_center_abs = 90 + 15 # from 1 to 3 and 3 to 5, there's a possibility of 90, including a margin of 10
degree_from_center_for_all = []
time_diff_between_ground_truth_tags = {}

for tag in ground_truth_stags:
	ground_truth_stags_data[tag] = tag + file_extension
	# ground_truth_stags_data[tag][:,1] = ground_truth_stags_data[tag][:,DATA_COL_S0]



for tag in unknown_stags:
	unknown_stags_data[tag] = tag + file_extension#, delimiter=" ")
	# unknown_stags_data[tag][:,1] = unknown_stags_data[tag][:,DATA_COL_S1]

t1 = PrettyTable(["Stag x", "Stag y", "tdiff (ms)"])#, "s0 period (ms)", "ddiff"])
print(t1)
for ground_truth_stag in ground_truth_stags:
		for unknown_stag in unknown_stags:
			# print("s1 shape",np.shape(ground_truth_stags_data[ground_truth_stag]))
			# time_shift, period_s0 = process_data(ground_truth_stags_data[ground_truth_stag], unknown_stags_data[unknown_stag], DATA_COL_S0, DATA_COL_S1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ_W_S1, SMOOTH_CUT_OFF_FREQ_W_S0,ground_truth_stag, unknown_stag, False, True)
			# print("-----------------------------------------------------------------")
			# print(colored((ground_truth_stag, unkn
			time_shift = compute_time_shift(ground_truth_stags_data[ground_truth_stag], unknown_stags_data[unknown_stag], truncate_start,truncate_end, show_plot= False)
			t1.add_row([ground_truth_stag, unknown_stag,  time_shift])#, period_s0, ddiff])
			# print(time_shift)
			print( "\n".join(t1.get_string().splitlines()[-2:]) )
			# print(time_shift)

			time_shift_with_unknown[ground_truth_stag] = time_shift
			# print(time_shift)
			ddiff = None#round((time_shift/period_s0) * 360,1)

t_s0_diff = PrettyTable(["Stag x", "Stag y", "tdiff (ms)"])
print(t_s0_diff)
for i in range(len(ground_truth_stags)-1):
			ground_truth_stag_1 = ground_truth_stags[i]
			ground_truth_stag_2 = ground_truth_stags[i+1]
			
			# time_shift, period_s0 = process_data(DATA_COL_S0, DATA_COL_S0, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ_W_S0, SMOOTH_CUT_OFF_FREQ_W_S0, ground_truth_stag_1, ground_truth_stag_2, False, False)
			time_shift = compute_time_shift(ground_truth_stags_data[ground_truth_stag_1], ground_truth_stags_data[ground_truth_stag_2], truncate_start,truncate_end, show_plot= False)
			# print(time_shift)
			t_s0_diff.add_row([ground_truth_stag_1, ground_truth_stag_2, time_shift])
			print( "\n".join(t_s0_diff.get_string().splitlines()[-2:]) )

			if time_diff_between_ground_truth_tags.get(ground_truth_stag_1) == None: time_diff_between_ground_truth_tags[ground_truth_stag_1] = {}
			if time_diff_between_ground_truth_tags.get(ground_truth_stag_2) == None: time_diff_between_ground_truth_tags[ground_truth_stag_2] = {}
			time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2] = time_shift
			time_diff_between_ground_truth_tags[ground_truth_stag_2][ground_truth_stag_1] = -time_shift

			ddiff = ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]


t2 = PrettyTable(["Stag x", "tdiff x->s_a", "ddiff x->s_a","ddiff s0_3->x","from " + center])
print(t2)

closest_s0_time_shift = 100000000
for ground_truth_stag in ground_truth_stags:

	ddiff = 0
	time_shift_ori = time_shift = time_shift_with_unknown[ground_truth_stag]
	ground_truth_stag_idx = ground_truth_stags.index(ground_truth_stag)


	if(time_shift > 0):
		# move to the right

		while (True):
			if (ground_truth_stag_idx + 1) <= (len(ground_truth_stags) - 1):
				ground_truth_stag_1 = ground_truth_stags[ground_truth_stag_idx]
				ground_truth_stag_2 = ground_truth_stags[ground_truth_stag_idx+1]
				ground_truth_stag_idx += 1
				if(time_shift > time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2] ):
					time_shift -= time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2]
					ddiff +=  ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]
				else:
					ddiff += (time_shift/(time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2])) * (ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)])
					# print(ddiff)
					break

			else:
				# print(ddiff)
				break					
		pass
	else:
		# move to the left
		# time_shift = -time_shift # set to positive
		while (True):
			if ((ground_truth_stag_idx - 1) >= 0):
				ground_truth_stag_1 = ground_truth_stags[ground_truth_stag_idx]
				ground_truth_stag_2 = ground_truth_stags[ground_truth_stag_idx-1]
				ground_truth_stag_idx -= 1
				if(time_shift < time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2] ):
					time_shift -= time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2]
					ddiff +=  ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]
				else:
					ddiff += (time_shift/(time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2])) * (ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)])
					# print(ddiff)
					break

			else:
				# print(ddiff)
				break	
		pass

	ddiff = round(ddiff,1)
	# degree_from_center = '?'

	ground_truth_diff_between_tags = ground_truth_diff[ground_truth_stags.index(center)][ground_truth_stags.index(ground_truth_stag)]
	degree_from_center = round((ddiff + ground_truth_diff_between_tags),1)

	degree_from_center_for_all.append(degree_from_center)

	t2.add_row([ground_truth_stag,
			time_shift_ori,\
			ddiff,\
			ground_truth_diff_between_tags,\
			degree_from_center])
	print( "\n".join(t2.get_string().splitlines()[-2:]) )

	color = "yellow"
	if(abs(closest_s0_time_shift) > abs(time_shift_ori)):
		closest_s0 = [colored(ground_truth_stag, color),
			colored(time_shift_ori, color),\
			colored(ddiff, color),\
			colored(ground_truth_diff_between_tags, color),\
			colored(degree_from_center, color)]
		closest_s0_time_shift = time_shift_ori

# print( "\n".join(t2.get_string().splitlines()[-2:]) )
t2.add_row(closest_s0)
print( "\n".join(t2.get_string().splitlines()[-2:]) )