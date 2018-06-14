#!/usr/bin/python3
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate, decimate, filtfilt
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

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-exp", "--experiment", dest="experiment", help="experiment name", required=True)
parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

CORRELATION_CIRCULAR = 0
X_CORR_BY_WINDOW = 1
TIMESTAMP_COL = 0
DATA_COL = 1
SMOOTH_WO_SHIFT = 1
NORMALIZE = 1

peak_width = 1000
mean_samples = 1
SMOOTH_CUT_OFF_FREQ = 1
WAVELET = 0

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
	s1 = s1[~np.isnan(s1).any(axis=1)]
	s2 = s2[~np.isnan(s2).any(axis=1)]

	min_t_possible = np.min(s2[:,0]) if  np.min(s2[:,0]) > np.min(s1[:,0]) else np.min(s1[:,0])
	max_t_possible = np.max(s1[:,0]) if  np.max(s2[:,0]) > np.max(s1[:,0]) else np.max(s2[:,0])

	s1 = s1[np.where((s1[:,0] >= min_t_possible) & (s1[:,0] <= max_t_possible))]
	s2 = s2[np.where((s2[:,0] >= min_t_possible) & (s2[:,0] <= max_t_possible))]

	if(DEBUG > 10): print(min_t_possible, max_t_possible)
	s1 = s1[np.where((s1[:,0] >= min_t_possible) & (s1[:,0] <= max_t_possible))]
	s2 = s2[np.where((s2[:,0] >= min_t_possible) & (s2[:,0] <= max_t_possible))]
	
	return s1, s2


def smoothen_without_shift(sig):#, impulse_length):
	fs = 1000
	nyq = 0.5 * fs
	low = SMOOTH_CUT_OFF_FREQ/nyq 
	# high = 1/nyq
	b, a = signal.butter(1, low, 'lowpass') # a lowpass Butterworth filter with a cutoff of 0.125 times the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
	
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


def process_data(s1,s2, truncate_start, truncate_end, smooth_cut_off_freq=SMOOTH_CUT_OFF_FREQ, s1_name=None, s2_name=None):

	global SMOOTH_CUT_OFF_FREQ
	SMOOTH_CUT_OFF_FREQ = smooth_cut_off_freq
	########################################################
	### ALIGN S1 AND S2 BEFORE XCORR
	########################################################
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
		s1_data = smoothen_without_shift(s1_data)#, impulse_length)
		s2_data = smoothen_without_shift(s2_data)#, impulse_length)
	if(NORMALIZE):
		# pass
		s1_data = normalize_regularize(s1_data)
		s2_data = normalize_regularize(s2_data)

	if(DEBUG > 10): print("len " + s1_name, len(s1_t), len(s1_data))
	if(DEBUG > 10): print("len " + s2_name, len(s2_t), len(s2_data))

	if(not CORRELATION_CIRCULAR):
		# METHOD 1: CORRELATION
		# numpy default correlation, 'zero padding?'
		xcorr = correlate(s1_data, s2_data)
		time_shift = (len(xcorr)/2 - xcorr.argmax())
	else:
		# METHOD 2: CORRELATION
		# periodic/circular correlation
		xcorr = periodic_corr_np(s2_data, s1_data)
		time_shift = xcorr.argmax()


	if(DEBUG > 2): print(colored("Time shift:" + str(time_shift) + " ms", "yellow"))

	# generates 1000 frequencies between 0.01 and 1
	freqs = np.linspace(0.01, 1, 100)

	# computes the Lomb Scargle Periodogram of the time and scaled magnitudes using each frequency as a guess
	periodogram_s0 = spectral.lombscargle(s1_t, s1_data, freqs)
	angular_freq_s0 = 1/freqs[np.argmax(periodogram_s0)]
	period_s0 = round(((2*math.pi)/(1/angular_freq_s0)) * 1000,2)

	return time_shift, period_s0



args = parser.parse_args()
file_extension = args.experiment[2:]
unknown_stags = [args.experiment[:2]]

truncate_start = 0 if(args.truncate_start == None) else  int(args.truncate_start)
truncate_end = 0 if(args.truncate_end == None) else  int(args.truncate_end)

ground_truth_stags = ['s0_1','s0_2','s0_3','s0_4','s0_5']


ground_truth_diff = np.array([	[0,		45, 	90,		135,	180],\
								[-45,	0,		45,		90,		135],\
								[-90,	-45,	0,		45,		90],\
								[-135,	-90,	-45,	0,		45],\
								[-179.9,	-135,	-90,	-45,	0],\
							])

ground_truth_stags_data = {}
unknown_stags_data = {}
time_shift_with_unknown = {}

t1 = PrettyTable(["Stag x", "Stag y", "tdiff (ms)", "s0 period (ms)"])
print(t1)
# print( "\n".join(t1.get_string().splitlines()[-2:]) )

for tag in ground_truth_stags:
	ground_truth_stags_data[tag] = np.loadtxt(tag + file_extension, delimiter=" ")

for tag in unknown_stags:
	unknown_stags_data[tag] = np.loadtxt(tag + file_extension, delimiter=" ")

for ground_truth_stag in ground_truth_stags:
	for unknown_stag in unknown_stags:
		time_shift, periodogram_s0 = process_data(ground_truth_stags_data[ground_truth_stag], unknown_stags_data[unknown_stag], truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, ground_truth_stag, unknown_stag)
		time_shift_with_unknown[ground_truth_stag] = time_shift
		t1.add_row([ground_truth_stag, unknown_stag,  time_shift, periodogram_s0])
		print( "\n".join(t1.get_string().splitlines()[-2:]) )
		# t1.add_row(["-","-","-"])
		# print( "\n".join(t1.get_string().splitlines()[-2:]) )
print()

t2 = PrettyTable(["Stag x", "Stag y", "ddiff ", "tdiff x->y", "tdiff to s1","ddiff to s1","from s0_3", "s0 period (ms)"])
print(t2)

center = 's0_3'
degree_from_center_for_all = []

for ground_truth_stag_1 in ground_truth_stags:
	for ground_truth_stag_2 in ground_truth_stags:
		if(ground_truth_stag_1 != ground_truth_stag_2):
			time_shift, periodogram_s0 = process_data(ground_truth_stags_data[ground_truth_stag_1], ground_truth_stags_data[ground_truth_stag_2], truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, ground_truth_stag_1, ground_truth_stag_2)
			ddiff = ground_truth_diff[ground_truth_stags.index(ground_truth_stag_1)][ground_truth_stags.index(ground_truth_stag_2)]
			ddiff_to_unknown = round((time_shift_with_unknown[ground_truth_stag_1]/time_shift) * ddiff,1)
			degree_from_center = round(ddiff_to_unknown + ground_truth_diff[ground_truth_stags.index(center)][ground_truth_stags.index(ground_truth_stag_1)],1)
			degree_from_center_for_all.append(degree_from_center)
			t2.add_row([ground_truth_stag_1, ground_truth_stag_2, ddiff, time_shift,\
						time_shift_with_unknown[ground_truth_stag_1],\
						ddiff_to_unknown ,\
						degree_from_center,\
						periodogram_s0])
			print( "\n".join(t2.get_string().splitlines()[-2:]) )

degree_from_center_for_all = np.array(degree_from_center_for_all)
print("MEDIAN: ", colored(np.median(degree_from_center_for_all),"yellow"))
c1 = cluster.vq.kmeans2(degree_from_center_for_all,3)
print("clusters:", c1[0],"cluster mode: ", colored(c1[0][stats.mode(c1[1])[0][0]], "green"))