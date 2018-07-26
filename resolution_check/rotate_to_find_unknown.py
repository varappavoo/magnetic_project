#!/usr/bin/python3
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate, decimate, filtfilt, wiener, medfilt, fftconvolve
from scipy.interpolate import spline
from sklearn.preprocessing import normalize
from scipy import interpolate, stats, cluster
from scipy.signal import spectral

from peakdetect import peakdetect
from termcolor import colored
from pandas import Series
import datetime
from prettytable import PrettyTable
from argparse import ArgumentParser
from random import randint
from pywt import wavedec
import pywt
import sys

from numpy.fft import fft, ifft

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f", "--files1", dest="experiment", help="use filename of stag that need to be localized, (for eg. s1_test_1.dat", required=True)
parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

CORRELATION_CIRCULAR = 0
X_CORR_BY_WINDOW = 0
TIMESTAMP_COL = 0
DATA_COL = 1
SMOOTH_WO_SHIFT = 1	
NORMALIZE = 1
SHOW_PLOT = 0
SHOW_PLOT_XCORR = 0

TRIM_START_END = 1
TRIM_LENGTH = 2000

peak_width = 1000
mean_samples = 1
SMOOTH_CUT_OFF_FREQ_W_S0 = 0.05#0.2
SMOOTH_CUT_OFF_FREQ_W_S1 = 0.05#0.08 # 0.08 seems fine for a period of approx 30sec with frame of size 60secs
WAVELET = 0
PERIOD_APPROX = 20000
# NUM_CLUSTERS_kmeans_2d = 5
# cluster_seed = [-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]
cluster_seed = np.arange(-90,100,10)
# cluster_seed = [-90,-45,0,45,90]
kmeans_iterations  = 10

DEBUG = 2

def align(s1,s2):
	global DATA_COL
	# if(DEBUG > 10): print("aligning...")

	# return s1,s2

	s1_series = Series(s1[:,DATA_COL],index=[datetime.datetime.fromtimestamp(ts) for ts in s1[:,0]])
	s2_series = Series(s2[:,DATA_COL],index=[datetime.datetime.fromtimestamp(ts) for ts in s2[:,0]])

	DATA_COL = 1

	'''
	S       secondly frequency
	L       milliseonds
	U       microseconds
	N       nanoseconds
	'''
	s1_series = s1_series.resample('1L').mean() # RESAMPLE TO 1ms BIN
	s1_series = s1_series.interpolate()

	s2_series = s2_series.resample('1L').mean() # 
	s2_series = s2_series.interpolate()

	# s1_series.plot()

	s1 = np.stack( (s1_series.index.astype(np.int64)/10**9, s1_series.values), axis = -1)
	s2 = np.stack( (s2_series.index.astype(np.int64)/10**9, s2_series.values), axis = -1)
	# , s2 = s1_series.as_matrix(),s2_series.as_matrix()

	# REMOVE ROW WITH nan VALUES
	# s1 = s1[~np.isnan(s1).any(axis=1)]
	# s2 = s2[~np.isnan(s2).any(axis=1)]

	min_t_possible = np.min(s2[:,0]) if  np.min(s2[:,0]) > np.min(s1[:,0]) else np.min(s1[:,0])
	max_t_possible = np.max(s1[:,0]) if  np.max(s2[:,0]) > np.max(s1[:,0]) else np.max(s2[:,0])

	s1 = s1[np.where((s1[:,0] >= min_t_possible) & (s1[:,0] <= max_t_possible))]
	s2 = s2[np.where((s2[:,0] >= min_t_possible) & (s2[:,0] <= max_t_possible))]

	if(DEBUG > 10): print(min_t_possible, max_t_possible)
	s1 = s1[np.where((s1[:,0] >= min_t_possible) & (s1[:,0] <= max_t_possible))]
	s2 = s2[np.where((s2[:,0] >= min_t_possible) & (s2[:,0] <= max_t_possible))]
	
	return s1, s2


def smoothen_without_shift(sig, smooth_cut_off_freq):#, impulse_length):
	fs = 1000
	nyq = 0.5 * fs
	low = smooth_cut_off_freq/nyq
	# low = 0.001/nyq 
	# high = 0.1/nyq
	b, a = signal.butter(1, low, 'lowpass') # a lowpass Butterworth filter with a cutoff of 0.125 times the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
	# b, a = signal.butter(1, high, 'highpass')
	# b, a = signal.butter(1, [low, high], 'bandpass', analog=True) # a lowpass Butterworth filter with a cutoff of 0.125 times the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
	# b, a = signal.butter(4, 100, 'low', analog=True)
	fgust = signal.filtfilt(b, a, sig, method="gust")#, irlen = impulse_length)
	return fgust
	# fgust -= fgust.mean(); fgust /= (3*fgust.std())
	# plt.plot(s1_t,fgust, label='gust', linestyle="--")

########################################
# REGULARIZE DATASETS
# https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
########################################
def normalize_regularize(sig):
	sig -= sig.mean(); sig /= (sig.std())
	# sig = normalize([sig], norm='l2')[0] # sklearn.preprocessing
	return sig

def normalize_zee_style(sig):
	pass



def periodic_corr_np(x, y):
	#
	# src: https://stackoverflow.com/questions/28284257/circular-cross-correlation-python
	# circular cross correlation python
	#
	#
	"""Periodic correlation, implemented using np.correlate.

	x and y must be real sequences with the same length.
	"""
	return np.correlate(x, np.hstack((y[1:], y)), mode='valid')

	# return ifft(fft(x) * fft(y).conj()).real


def process_data(s1,s2, truncate_start, truncate_end, smooth_cut_off_freq=None, smooth_cut_off_freq_w_s0=None, s1_name=None, s2_name=None, compute_periodogram = False):

	# global SMOOTH_CUT_OFF_FREQ
	# SMOOTH_CUT_OFF_FREQ = smooth_cut_off_freq
	########################################################
	### ALIGN S1 AND S2 BEFORE XCORR
	########################################################
	# s1,s2 = align(s1,s2)
	s1,s2 = align(s1,s2)

	s1_t = s1[:,0]
	s2_t = s2[:,0]

	s1_data = s1[:,DATA_COL]
	s2_data = s2[:,DATA_COL]

	###############################################
	## TRUNCATE
	###############################################
	if truncate_end != 0:
		TRUNCATE_END = truncate_end
		s1_t = s1_t[:TRUNCATE_END]
		s2_t = s2_t[:TRUNCATE_END]
		s1_data = s1_data[:TRUNCATE_END]
		s2_data = s2_data[:TRUNCATE_END]

	if truncate_start != 0:
		TRUNCATE_START = truncate_start
		s1_t = s1_t[TRUNCATE_START:]
		s2_t = s2_t[TRUNCATE_START:]
		s1_data = s1_data[TRUNCATE_START:]
		s2_data = s2_data[TRUNCATE_START:]

	# s1_data = smoothen_without_shift(s1_data)
	# s2_data = smoothen_without_shift(s2_data)
	# impulse_length = 5000
	if(SMOOTH_WO_SHIFT):
		s1_data = smoothen_without_shift(s1_data, smooth_cut_off_freq_w_s0)#, impulse_length)
		s2_data = smoothen_without_shift(s2_data, smooth_cut_off_freq)#, impulse_length)



	if(NORMALIZE):
		# pass
		s1_data = normalize_regularize(s1_data)
		s2_data = normalize_regularize(s2_data)

	if(TRIM_START_END):
		s1_t = s1_t[TRIM_LENGTH:len(s1_t)-TRIM_LENGTH]
		s2_t = s2_t[TRIM_LENGTH:len(s2_t)-TRIM_LENGTH]
		s1_data = s1_data[TRIM_LENGTH:len(s1_data)-TRIM_LENGTH]
		s2_data = s2_data[TRIM_LENGTH:len(s2_data)-TRIM_LENGTH]
		# print(len(s1_data),len(s2_data))

	if(DEBUG > 10): print("len " + s1_name, len(s1_t), len(s1_data))
	if(DEBUG > 10): print("len " + s2_name, len(s2_t), len(s2_data))

	# print(find_peaks(s1_data,height=1,width=1000))
	# print(find_peaks(s2_data,height=1,width=1000))

	if(not CORRELATION_CIRCULAR):
		# METHOD 1: CORRELATION
		# numpy default correlation, 'zero padding?'
		xcorr = correlate(s1_data, s2_data)
		# xcorr = fftconvolve(s1_data, s2_data)
		time_shift = (len(xcorr)/2 - xcorr.argmax())
	else:
		# METHOD 2: CORRELATION
		# periodic/circular correlation
		xcorr = periodic_corr_np(s2_data, s1_data)
		xcorr_max = xcorr.argmax()
		time_shift = xcorr_max
		# if xcorr_max <= len(s1_data)/2:
		# 	time_shift = xcorr_max
		# else:
		# 	time_shift = xcorr_max - len(s1_data)

	if(SHOW_PLOT):
		plt.plot(s1_data, label = s1_name)
		plt.plot(s2_data, label = s2_name)
		# plt.plot(wiener(s2_data,1000), label = 'wiener')
		# plt.plot(medfilt(s2_data,5), label = 'medfilt')
		# plt.plot(normalize_regularize(fftconvolve(s1_data, s2_data)), label = 'fftconvolve')
		if(SHOW_PLOT_XCORR):
			plt.plot(normalize_regularize(xcorr), label = "xcorr")
		plt.legend()
		plt.show()


	if(DEBUG > 2): print(colored("Time shift:" + str(time_shift) + " ms", "yellow"))

	# generates 1000 frequencies between 0.01 and 1
	freqs = np.linspace(0.01, 1, 100)

	# computes the Lomb Scargle Periodogram of the time and scaled magnitudes using each frequency as a guess
	if(compute_periodogram):
		periodogram_s0 = spectral.lombscargle(s1_t, s1_data, freqs)
		angular_freq_s0 = 1/freqs[np.argmax(periodogram_s0)]
		period_s0 = round(((2*math.pi)/(1/angular_freq_s0)) * 1000,2)

		# negative greater than half of period		
		if(math.fabs(time_shift*2) > period_s0): time_shift = round(time_shift % period_s0,1)
		# positive greater than half of period
		if((time_shift*2) > period_s0): time_shift = round(time_shift - period_s0,1)
	else:
		period_s0 = 1

	return time_shift, period_s0

def check_monotonicity(L):
	# https://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
    return all(x>y for x, y in zip(L, L[1:])) or all(x<y for x, y in zip(L, L[1:]))

def loop(truncate_start, truncate_end):
	global ground_truth_stags_data, unknown_stags_data, time_shift_with_unknown, center, degree_from_center_for_all, max_degree_possible_from_center_abs
	monotonic = []
	t1 = PrettyTable(["Stag x", "Stag y", "tdiff (ms)", "s0 period (ms)", "ddiff"])
	print(t1)
	for ground_truth_stag in ground_truth_stags:
		for unknown_stag in unknown_stags:
			time_shift, period_s0 = process_data(ground_truth_stags_data[ground_truth_stag], unknown_stags_data[unknown_stag], truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ_W_S1, SMOOTH_CUT_OFF_FREQ_W_S0,ground_truth_stag, unknown_stag, True)
			# print("-----------------------------------------------------------------")
			# print(colored((ground_truth_stag, unknown_stag, time_shift),"cyan"))
			# print("-----------------------------------------------------------------")
			time_shift_with_unknown[ground_truth_stag] = time_shift
			ddiff = round((time_shift/period_s0) * 360,1)
			# monotonic.append(ddiff)
			monotonic.append(time_shift)
			t1.add_row([ground_truth_stag, unknown_stag,  time_shift, period_s0, ddiff])
			print( "\n".join(t1.get_string().splitlines()[-2:]) )
			# t1.add_row(["-","-","-"])
			# print( "\n".join(t1.get_string().splitlines()[-2:]) )
	monotonically_decreasing = check_monotonicity(monotonic)
	print("MONOTONICITY (tdiff): ", monotonically_decreasing)

	monotonically_decreasing = True
	if(monotonically_decreasing):
		print()
		t_s0_diff = PrettyTable(["Stag x", "Stag y", "tdiff"])
		print(t_s0_diff)
# processed_tags = []
# for ground_truth_stag_1 in ground_truth_stags:
# 	processed_tags.append(ground_truth_stag_1)
# 	for ground_truth_stag_2 in ground_truth_stags:
# 		if(ground_truth_stag_2 not in processed_tags):
				# if(ground_truth_stag_1 != ground_truth_stag_2):
		time_diff_between_ground_truth_tags = {}
		for i in range(len(ground_truth_stags)-1):
			ground_truth_stag_1 = ground_truth_stags[i]
			ground_truth_stag_2 = ground_truth_stags[i+1]
			
			time_shift, period_s0 = process_data(ground_truth_stags_data[ground_truth_stag_1], ground_truth_stags_data[ground_truth_stag_2], truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ_W_S0, SMOOTH_CUT_OFF_FREQ_W_S0, ground_truth_stag_1, ground_truth_stag_2)

			# print("-----------------------------------------------------------------")
			# print(colored((ground_truth_stag_1, ground_truth_stag_2, time_shift),"cyan"))
			# print("-----------------------------------------------------------------")
			t_s0_diff.add_row([ground_truth_stag_1, ground_truth_stag_2, time_shift])
			print( "\n".join(t_s0_diff.get_string().splitlines()[-2:]) )

			if time_diff_between_ground_truth_tags.get(ground_truth_stag_1) == None: time_diff_between_ground_truth_tags[ground_truth_stag_1] = {}
			if time_diff_between_ground_truth_tags.get(ground_truth_stag_2) == None: time_diff_between_ground_truth_tags[ground_truth_stag_2] = {}
			time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2] = time_shift
			time_diff_between_ground_truth_tags[ground_truth_stag_2][ground_truth_stag_1] = -time_shift

			ddiff = ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]

		print()

		t2 = PrettyTable(["Stag x", "tdiff x->s_a", "ddiff x->s_a","ddiff s0_3->x","from s0_3"])
		print(t2)
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
			# degree_from_center = (ddiff + ground_truth_diff_between_tags)

			degree_from_center_for_all.append(degree_from_center)
			# if(math.fabs(degree_from_center) <= max_degree_possible_from_center_abs ):
			# 	# degree_from_center_for_all.append(degree_from_center)
			# 	degree_from_center_str = str(degree_from_center) if(math.fabs(degree_from_center) <= max_degree_possible_from_center_abs ) else colored(str(degree_from_center),"red")
			t2.add_row([ground_truth_stag,
					time_shift_ori,\
					ddiff,\
					ground_truth_diff_between_tags,\
					degree_from_center])
			print( "\n".join(t2.get_string().splitlines()[-2:]) )

	
	# i = 0
	# # ground_truth_stag_1 = ground_truth_stags[i]
	# # ground_truth_stag_2 = ground_truth_stags[i+1]
	# for ground_truth_stag in ground_truth_stags:
	# 	time_shift_with_unknown = time_shift_with_unknown[ground_truth_stag]
	# 	while(True):
	# 		ground_truth_stag_1 = ground_truth_stags[i]
	# 		ground_truth_stag_2 = ground_truth_stags[i+1]
	# 		if(time_shift_with_unknown > time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2]):
	# 			time_shift_with_unknown -= time_diff_between_ground_truth_tags[ground_truth_stag_1][ground_truth_stag_2]
	# 			i += 1
	# 		else:
	# 			break



		# ddiff = ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]
		
		# # print(ground_truth_diff_between_tags)
		# if(ddiff == 180): time_shift = math.fabs(time_shift) 

		
		# # ddiff_to_unknown = round(((ddiff/time_shift)*time_shift_with_unknown[ground_truth_stag_1])%360,1)
		# # degree_from_center = round((ddiff_to_unknown + ground_truth_diff[ground_truth_stags.index(center)][ground_truth_stags.index(ground_truth_stag_1)])%360,1)
		# ddiff_to_unknown = round(((ddiff/time_shift)*time_shift_with_unknown[ground_truth_stag_1]),1)
		# if(math.fabs(ddiff_to_unknown) > 180): ddiff_to_unknown = ddiff_to_unknown%180 
		# ground_truth_diff_between_tags = ground_truth_diff[ground_truth_stags.index(center)][ground_truth_stags.index(ground_truth_stag_1)]
		# degree_from_center = round((ddiff_to_unknown + ground_truth_diff_between_tags),1)
		# if(math.fabs(degree_from_center) <= max_degree_possible_from_center_abs ):
		# 	degree_from_center_for_all.append(degree_from_center)
		# degree_from_center_str = str(degree_from_center) if(math.fabs(degree_from_center) <= max_degree_possible_from_center_abs ) else colored(str(degree_from_center),"red")
		# t2.add_row([ground_truth_stag_1, ground_truth_stag_2, ddiff,\
		# 			period_s0,\
		# 			time_shift,\
		# 			time_shift_with_unknown[ground_truth_stag_1],\
		# 			ddiff_to_unknown ,\
		# 			degree_from_center_str])
		# print( "\n".join(t2.get_string().splitlines()[-2:]) )
		# # processed_tags.append(ground_truth_stag_2)

	# 	# degree_from_center_for_all = np.array(degree_from_center_for_all)
	# 	# print(degree_from_center_for_all)
	# 	# print(len(degree_from_center_for_all))
		print("MEDIAN:\t\t\t\t", colored(np.median(np.array(degree_from_center_for_all)),"yellow"))
		centroids, distortion = cluster.vq.kmeans(degree_from_center_for_all,cluster_seed, kmeans_iterations, 1e-05,'matrix');
		idx,_ = cluster.vq.vq(degree_from_center_for_all,centroids)
		bincount = np.bincount(idx)
		print("labels:",idx)
		print("bin count:",bincount)
		print("clusters with highest bin counts:", centroids[np.argsort(bincount)[len(bincount)-2:]])
		print("CENTROID OF LARGEST CLUSTER:\t", colored(round(centroids[np.argmax(np.bincount(idx))],1),"cyan"))
		c1 = cluster.vq.kmeans2(degree_from_center_for_all,cluster_seed, kmeans_iterations, 1e-05,'matrix');
		print("KMEANS2 CLUSTER MODE:\t\t", colored(round(c1[0][stats.mode(c1[1])[0][0]],1), "green"),"clusters:", c1[0], c1[1])
		# print("CLUSTER HIERARCHY CENTROID:\t", colored(cluster.hierarchy.centroid(degree_from_center_for_all), "green"))
		# print("CLUSTER HIERARCHY MEDIAN:\t", colored(cluster.hierarchy.median(degree_from_center_for_all), "green"))
	else:
		print(colored("NOT MONOTICALLY DECREASING","red"))

args = parser.parse_args()
file_extension = args.experiment[2:]
unknown_stags = [args.experiment[:2]]

truncate_start = 0 if(args.truncate_start == None) else  int(args.truncate_start)
truncate_end = 0 if(args.truncate_end == None) else  int(args.truncate_end)

# CW
ground_truth_stags = ['s0_1','s0_2','s0_3','s0_4','s0_5']
# ground_truth_diff_ccw
# ground_truth_stags = ['s0_5','s0_4','s0_3','s0_2','s0_1']


ground_truth_diff = np.array([	[0,		45, 	90,		135,	180],\
								[-45,	0,		45,		90,		135],\
								[-90,	-45,	0,		45,		90],\
								[-135,	-90,	-45,	0,		45],\
								[-179.9, -135,	-90,	-45,	0],\
							])

# ground_truth_diff_ccw = np.array([	[0,		45, 	90,		135,	180],\
# 								[-45,	0,		45,		90,		135],\
# 								[-90,	-45,	0,		45,		90],\
# 								[-135,	-90,	-45,	0,		45],\
# 								[-179.9, -135,	-90,	-45,	0],\
# 							])
# end = 0
# start = 0
# period_avg = 0

# while(end < truncate_end):
ground_truth_stags_data = {}
unknown_stags_data = {}
time_shift_with_unknown = {}
center = 's0_3'
max_degree_possible_from_center_abs = 90 + 15 # from 1 to 3 and 3 to 5, there's a possibility of 90, including a margin of 10
degree_from_center_for_all = []




# print( "\n".join(t1.get_string().splitlines()[-2:]) )

for tag in ground_truth_stags:
	ground_truth_stags_data[tag] = np.loadtxt(tag + file_extension, delimiter=" ")

for tag in unknown_stags:
	unknown_stags_data[tag] = np.loadtxt(tag + file_extension, delimiter=" ")

if(X_CORR_BY_WINDOW):
	for i in range(truncate_start, truncate_end, int(PERIOD_APPROX/3)):#int(PERIOD_APPROX/4)):
	# for i in range(int(truncate_end/PERIOD_APPROX)):#int(PERIOD_APPROX/4)):
		truncate_start = i
		truncate_end_tmp = truncate_start + int(PERIOD_APPROX * 1.1) #int(PERIOD_APPROX * 1.5)
		print("start:",truncate_start,"\tend:",truncate_end_tmp)
		loop(truncate_start, truncate_end_tmp)
		truncate_start = truncate_end_tmp

	# for i in range(truncate_start, truncate_end, 1000):#int(PERIOD_APPROX/4)):
	# 	truncate_start = i
	# 	truncate_end = i + int(PERIOD_APPROX * 2) #int(PERIOD_APPROX * 1.5)
	# 	print("start:",truncate_start,"\tend:",truncate_end)
	# 	loop(truncate_start, truncate_end)
else:
	loop(truncate_start, truncate_end)

# print(degree_from_center_for_all)
# print(len(degree_from_center_for_all))
# print("MEDIAN:\t\t\t\t", colored(np.median(np.array(degree_from_center_for_all)),"yellow"))
# # centroids, distortion = cluster.vq.kmeans(degree_from_center_for_all,NUM_CLUSTERS)
# # centroids, distortion = cluster.vq.kmeans(b,[0,45,90,135,180],10,1e-05,'matrix');
# centroids, distortion = cluster.vq.kmeans(degree_from_center_for_all,cluster_seed, kmeans_iterations, 1e-05,'matrix');
# idx,_ = cluster.vq.vq(degree_from_center_for_all,centroids)
# bincount = np.bincount(idx)
# print("bin count:",bincount)
# # print("bin count:",np.bincount(idx))
# print("clusters with highest bin counts:", centroids[np.argsort(bincount)[len(bincount)-2:]])
# print("CENTROID OF LARGEST CLUSTER:\t", colored(round(centroids[np.argmax(np.bincount(idx))],1),"cyan"))
# c1 = cluster.vq.kmeans2(degree_from_center_for_all,cluster_seed, kmeans_iterations, 1e-05,'matrix');
# print("KMEANS2 CLUSTER MODE:\t\t", colored(round(c1[0][stats.mode(c1[1])[0][0]],1), "green"),"clusters:", c1[0])
# print("CLUSTER SEED:", cluster_seed)
