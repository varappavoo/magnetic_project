#!/usr/bin/python3
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate, decimate, filtfilt
from scipy.interpolate import spline

from scipy import interpolate

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
parser.add_argument("-f1", "--sensor0_0file", dest="s0_0_filename", help="collected data from s0_0", metavar="FILE", required=True)
parser.add_argument("-f2", "--sensor0_1file", dest="s0_1_filename", help="collected data from s0_1`", metavar="FILE", required=True)
parser.add_argument("-f3", "--sensor1file", dest="s1_filename", help="collected data from s1", metavar="FILE", required=True)
parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)


X_CORR_BY_WINDOW = 1
TIMESTAMP_COL = 0
DATA_COL = 1
DEBUG = 2
SHOW_PLOT = 0
SHOW_PEAKS_PROMINENCES = 0
SHOW_XCORR = 0
COMPUTE_ANGULAR_FREQUENCY_DEGREE_DRIFT = 1
SMOOTH_WO_SHIFT = 1
NORMALIZE = 1
CSV_OUTPUT = 0
# PRINT_TIME_SHIFT__PERIOD_S1__PERIOD_S2_ONLY = 1

peak_width = 1000
mean_samples = 1
SMOOTH_CUT_OFF_FREQ = 0.1
WAVELET = 0

lower_percentile = 25
higher_percentile = 75
histogram_bin_size = 10

time_shift_arr = []

def running_mean(x, N):
	# src: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02

def align(s1,s2):
	global DATA_COL
	if(DEBUG > 10): print("aligning...")

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
	# sig -= sig.mean(); sig /= (3*sig.std())
	return sig

def normalize_zee_style(sig):
	pass

# def normalize_max_min(sig):
# 	sig = (sig - sig[np.argmin(sig)])/(sig[np.argmax(sig)] - sig[np.argmin(sig)])
# 	# sig = (sig - np.average(sig)/(sig[np.argmax(sig)] - sig[np.argmin(sig)]))
# 	return sig

def process_data(s1,s2, truncate_start, truncate_end, smooth_cut_off_freq=SMOOTH_CUT_OFF_FREQ, filename1=None, filename2=None):
	# if(DEBUG > 10): print(np.min(s1[:,0]),np.max(s1[:,0]))
	# if(DEBUG > 10): print(np.min(s2[:,0]),np.max(s2[:,0]))
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
		s1_data = normalize_regularize(s1_data)
		s2_data = normalize_regularize(s2_data)
		# s1_data = normalize_max_min(s1_data)
		# s2_data = normalize_max_min(s2_data)


	# if(DEBUG > 10): print(np.min(s1[:,0]),np.max(s1[:,0]))
	# if(DEBUG > 10): print(np.min(s2[:,0]),np.max(s2[:,0]))

	# s1_data = s1[:,DATA_COL]-np.min(s1)
	# s2_data = s2[:,DATA_COL]-np.min(s2)
	# s1_data = s1_data[mean_samples-1:]

	# s1_data = running_mean((s1[:,DATA_COL]-np.min(s1)),mean_samples)
	# s2_data = running_mean((s2[:,DATA_COL]-np.min(s2)),mean_samples)

	# s2_data = signal.resample(s2_data, int(len(s2_data/2)))
	# s2_data = s2_data[::2] #decimate(s2_data, 10)

	########################################
	# REGULARIZE DATASETS
	# https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	########################################
	# s1_data -= s1_data.mean(); s1_data /= (3*s1_data.std())
	# s2_data -= s2_data.mean(); s2_data /= (3*s2_data.std())
		
	##########################################
	# OR NORMALIZE BETWEEN ZERO AND ONE
	# ONE EXTREME CHANGE CAN HIJACK EVERYTHING?
	##########################################
	# dx_s1 = np.max(s1_data) - np.min(s1_data)
	# s1_data = (s1_data-np.min(s1_data))/dx_s1
	# dx_s2 = np.max(s2_data) - np.min(s2_data)
	# s2_data = (s2_data-np.min(s2_data))/dx_s2

	# ########################################

	# s1_t = s1[:,0][mean_samples-1:]
	# s2_t = s2[:,0][mean_samples-1:]

	# ############################################
	# #NORMALIZE TIME
	# ############################################
	# s1_t = s1_t - s1_t[0]
	# s2_t = s2_t - s2_t[0]

	# s2_t = s2_t[::2] #decimate(s2_t, 10)
	# s2_t = signal.resample(s2_t, int(len(s2_t/2)))
	# s2_t = s2_t[:len(s2_data)]

	if(DEBUG > 10): print("len " + args.s1_filename, len(s1_t), len(s1_data))
	if(DEBUG > 10): print("len " + args.s2_filename, len(s2_t), len(s2_data))



	if(WAVELET):
	#######################################################
	# START OF: WAVELET TEST
	#######################################################
		[s1_cA, s1_cD] = wavedec(s1_data, pywt.Wavelet('dmey'),level=1)
		[s2_cA, s2_cD] = wavedec(s2_data, pywt.Wavelet('dmey'),level=1)

		# t = s1_t[0:len(s1_cD)]

		plt.plot(s1_cA, label='s1_cA')
		plt.plot(s2_cA, label='s2_cA')

		plt.plot(s1_cD, label='s1_cD')
		plt.plot(s2_cD, label='s2_cD')

		# plt.legend()
		# plt.show()

		s1_data = s1_cA
		s2_data = s2_cA

		# print(len(s1_t), len(s1_data))

		s1_t = np.linspace(0,len(s1_data)/500,num=len(s1_data))
		s2_t = np.linspace(0,len(s1_data)/500,num=len(s1_data))


		# ti = np.linspace(0, len(s1_data), 1)
		# dxdt, dydt = interpolate.splev(s1_data,der=1)
		# plt.plot(dxdt, label='dxdt_s1')
		# plt.plot(dydt, label='dydt_s1')

		# ti = np.linspace(0, len(s2_data), 1)
		# dxdt, dydt = interpolate.splev(s2_data,der=1)

		# plt.plot(dxdt, label='dxdt_s2')
		# plt.plot(dydt, label='dydt_s2')
		if(SHOW_PLOT):
			plt.legend()
			plt.show()


	# print(len(s1_t), len(s1_data))

	###########################################
	# CORRELATION
	###########################################

	if(DEBUG > 10): print("len " + args.s1_filename, len(s1_t), len(s1_data))
	if(DEBUG > 10): print("len " + args.s2_filename, len(s2_t), len(s2_data))
	xcorr = correlate(s1_data, s2_data)

	# xcorr = smoothen_without_shift(xcorr)

	time_shift = (len(xcorr)/2 - xcorr.argmax())
	# print(len(xcorr), xcorr.argmax())

	# xcorr -= xcorr.mean(); xcorr /= xcorr.std() 
	#downsampling for plotting
	# xcorr = xcorr[1:len(xcorr)]

	# if(DEBUG > 10): print(colored("XCorr: " + str(xcorr), "yellow"))
	# if(DEBUG > 10): print(colored("Time shift (APPROX - IN TERMS OF SAMPLES):" + str(time_shift), "yellow"))
	# if(DEBUG > 10): print(colored("Time shift (COULD BE APPROX):" + str(s1_t[time_shift] - s1_t[0]), "yellow"))
	if(DEBUG > 10): print(colored("Time shift:" + str(time_shift) + " ms", "yellow"))
	# if(DEBUG > 10): print(len(xcorr), len(s1_data))











	# ###########################################
	# # CORRELATION
	# ###########################################

	# xcorr = correlate(s1_cD, s2_cD)
	# time_shift = (int(len(xcorr)/2) - xcorr.argmax())*2

	# # xcorr -= xcorr.mean(); xcorr /= xcorr.std() 
	# #downsampling for plotting
	# # xcorr = xcorr[1:len(xcorr):2]

	
	# # xcorr_t = 
	# plt.plot(xcorr, label='xcorr')
	# # if(DEBUG > 10): print(colored("XCorr: " + str(xcorr), "yellow"))
	# # if(DEBUG > 10): print(colored("Time shift (APPROX - IN TERMS OF SAMPLES):" + str(time_shift), "yellow"))
	# # if(DEBUG > 10): print(colored("Time shift (COULD BE APPROX):" + str(s1_t[time_shift] - s1_t[0]), "yellow"))
	# print(colored("Time shift:" + str(time_shift) + " ms", "yellow"))

	# plt.legend()
	# plt.show()

	# sys.exit()
	# #######################################################
	# # END OF: WAVELET TEST
	# #######################################################

	# plt.figure(1)
	if(SHOW_PLOT):plt.figure(figsize=(13,9))
	# plt.subplot(211)
	# plt.plot(s1[:,0], s1[:,2]-np.min(s1),  label='s1')
	if(SHOW_PLOT):plt.plot(s1_t, s1_data,  label=filename1)
	# plt.plot(s1[:,0][mean_samples-1:], s1_data,  label='s1_avg')
	# plt.plot(s2[:,0], s2[:,2]-np.min(s2),  label='s2')
	if(SHOW_PLOT):plt.plot(s2_t, s2_data,  label=filename2)

	# plt.plot(np.arange(len(xcorr)) + s1_t[0], xcorr, label='xcorr')
	# if(len(s1_t) >= len(s2_t)):
	# 	xcorr_t = s1_t[0:len(xcorr)]
	# else:
	# 	xcorr_t = s2_t[0:len(xcorr)]
	xcorr_t = np.linspace(s1_t[0], s1_t[0] + len(xcorr)/1000, num = len(xcorr))
	xcorr = normalize_regularize(xcorr)
	if(SHOW_XCORR):plt.plot(xcorr_t, xcorr, label='xcorr')


	# #####################################################
	# # filtfilt smooth back amd forth
	# #####################################################
	# # n=100
	# # sig = np.random.randn(n)**3 + 3*np.random.randn(n).cumsum()
	# # x = np.random.randn(10)
	# # b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied.
	# b, a = signal.butter(2, 0.0001) # a lowpass Butterworth filter with a cutoff of 0.125 times the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
	# # b, a = signal.butter(4, 100, 'low', analog=True)
	# fgust = signal.filtfilt(b, a, s1_data, method="gust")
	# fgust -= fgust.mean(); fgust /= (3*fgust.std())
	# plt.plot(s1_t,fgust, label='gust', linestyle="--")

	# plt.show()

	if(DEBUG > 10): print("peak detect")
	if(DEBUG > 10): print("s1 peaks")
	# peakind = signal.find_peaks_cwt(s1_data, np.arange(1,1000), signal.ricker)
	# # peakind = signal.find_peaks_cwt(s1_data,  wavelet = signal.wavelets.daub, widths=10)
	# if(DEBUG > 10): print(peakind, s1[peakind])#, s1_data[peakind])

	peaks_sci_s1_idx, _ = find_peaks(s1_data, width=peak_width)
	prominences = peak_prominences(s1_data, peaks_sci_s1_idx)[0]
	if(DEBUG > 2): print("peaks and prominences")
	if(DEBUG > 2): print(peaks_sci_s1_idx,end="")
	if(DEBUG > 2): print(colored(np.diff(peaks_sci_s1_idx),"cyan"))	
	if(DEBUG > 2): print(colored(prominences,"yellow"),  colored(np.mean(prominences),"red"))


	peaks_peakdetect_s1 = peakdetect(s1_data, lookahead=peak_width)
	# if(DEBUG > 10): print(peaks_peakdetect_s1)
	if len(peaks_peakdetect_s1[0]) == 0: 
		if len(peaks_peakdetect_s1[1]) == 0:
			peaks_peakdetect_s1 = np.array([]) 
		else:
			peaks_peakdetect_s1 = np.array(peaks_peakdetect_s1[1])[:,0]
	else:
			if len(peaks_peakdetect_s1[1]) == 0:
		 		peaks_peakdetect_s1 = np.array(peaks_peakdetect_s1[0])[:,0]
			else:
				peaks_peakdetect_s1 = np.append(np.array(peaks_peakdetect_s1[0])[:,0],np.array(peaks_peakdetect_s1[1])[:,0]) # contains both peaks and valleys as separate lists

	peaks_peakdetect_s1.sort()
	peaks_peakdetect_s1 = peaks_peakdetect_s1.astype(int)
	# if(DEBUG > 10): print(peaks_peakdetect_s1, end="")
	# if(DEBUG > 10): print(colored(np.diff(peaks_peakdetect_s1),"cyan"))

	if(SHOW_PEAKS_PROMINENCES): plt.plot(s1_t[peaks_sci_s1_idx], s1_data[peaks_sci_s1_idx], "x")
	# plt.plot(s1[peakind], s1_data[peakind], "o")
	# plt.plot(s1_t[peaks_peakdetect_s1], s1_data[peaks_peakdetect_s1], "X")

	# if(DEBUG > 10): print(signal.peak_prominences(s1_data,peakind)[0])
	# # plt.plot(s1[:,0][mean_samples-1:][peakind], s1_data[peakind], "x")

	if(DEBUG > 10): print("s2 peaks")
	# peakind = signal.find_peaks_cwt(s2_data, np.arange(1,1000), signal.ricker)
	# if(DEBUG > 10): print(peakind, s2[peakind])

	peaks_sci_s2_idx, _ = find_peaks(s2_data, width=peak_width)
	prominences = peak_prominences(s2_data, peaks_sci_s2_idx)[0]
	if(DEBUG > 2): print("peaks and prominences")
	if(DEBUG > 2): print(peaks_sci_s2_idx, end="")
	if(DEBUG > 2): print(colored(np.diff(peaks_sci_s2_idx),"cyan"))	
	if(DEBUG > 2): print(colored(prominences,"yellow"), colored(np.mean(prominences),"red"))

	# if(DEBUG > 10): print("peak detect")
	peaks_peakdetect_s2 = peakdetect(s2_data, lookahead=peak_width)


	if len(peaks_peakdetect_s2[0]) == 0: 
		if len(peaks_peakdetect_s2[1]) == 0:
			peaks_peakdetect_s2 = np.array([]) 
		else:
			peaks_peakdetect_s2 = np.array(peaks_peakdetect_s2[1])[:,0]
	else:
		if len(peaks_peakdetect_s2[1]) == 0:
			peaks_peakdetect_s2 = np.array(peaks_peakdetect_s2[0])[:,0]
		else:
			peaks_peakdetect_s2 = np.append(np.array(peaks_peakdetect_s2[0])[:,0],np.array(peaks_peakdetect_s2[1])[:,0]) # contains both peaks and valleys as separate lists
	# peaks_peakdetect_s2 = np.append(np.array(peaks_peakdetect_s2[0])[:,0],np.array(peaks_peakdetect_s2[1])[:,0]) # contains both peaks and valleys as separate lists
	peaks_peakdetect_s2.sort()
	peaks_peakdetect_s2 = peaks_peakdetect_s2.astype(int)
	# if(DEBUG > 10): print(peaks_peakdetect_s2, end="")
	# if(DEBUG > 10): print(colored(np.diff(peaks_peakdetect_s2),"cyan"))	

	if(SHOW_PEAKS_PROMINENCES): plt.plot(s2_t[peaks_sci_s2_idx], s2_data[peaks_sci_s2_idx], "x")
	# plt.plot(s2[peakind], s2_data[peakind], "o")
	# plt.plot(s2_t[peaks_peakdetect_s2], s2_data[peaks_peakdetect_s2], "X")

	# TAKE 10 INDICES ??????????????
	# INDICES INVOLVED IN COMPUTING DIFF IS 10?
	try:
		peaks_sci_diff = (peaks_sci_s1_idx[0:6] - peaks_sci_s2_idx[0:6])
		if(DEBUG >2): print(colored(peaks_sci_diff,"red"), np.mean(peaks_sci_diff))
	except:
		if(DEBUG >2): print("ERROR TRYING TO COMPUTE PEAKS DIFF", len(peaks_sci_s1_idx), len(peaks_sci_s2_idx))
	# if(DEBUG > 10): print(colored(np.sum(peaks_sci_diff)/len(peaks_sci_diff),"green"))


	# s0_smooth = spline(s1_data[peaks_sci_s1_idx], s1_t[peaks_sci_s1_idx], s1_t)
	# plt.plot(s1_t, s0_smooth, label = "spline (????????????????)" )

	####################################################
	# CONPUTE PERIOD OF A ROTATION
	####################################################
	if(COMPUTE_ANGULAR_FREQUENCY_DEGREE_DRIFT):
		# SRC: https://stackoverflow.com/questions/13349181/using-scipy-signal-spectral-lombscargle-for-period-discovery?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
		from scipy.signal import spectral
		# generates 1000 frequencies between 0.01 and 1
		freqs = np.linspace(0.01, 1, 100)

		# computes the Lomb Scargle Periodogram of the time and scaled magnitudes using each frequency as a guess
		periodogram_s1 = spectral.lombscargle(s1_t, s1_data, freqs)
		periodogram_s2 = spectral.lombscargle(s2_t, s2_data, freqs)
		# if(DEBUG > 10): print(periodogram)
		# returns the inverse of the frequence (i.e. the period) of the largest periodogram value

		angular_freq_s1 = 1/freqs[np.argmax(periodogram_s1)]
		angular_freq_s2 = 1/freqs[np.argmax(periodogram_s2)]
		if(DEBUG > 10): print(args.s1_filename, "Angular frequency:" , angular_freq_s1, "radians per second")
		if(DEBUG > 10): print(args.s2_filename, "Angular frequency:" , angular_freq_s2, "radians per second")
		if(DEBUG > 10): print(args.s1_filename, "Period:", (2*math.pi)/(1/angular_freq_s1))
		if(DEBUG > 10): print(args.s2_filename, "Period:", (2*math.pi)/(1/angular_freq_s2))

		angular_freq_avg = ( (2*math.pi)/(1/angular_freq_s1) + (2*math.pi)/(1/angular_freq_s1) )/2
		angular_freq_max = (2*math.pi)/(1/angular_freq_s1) if (2*math.pi)/(1/angular_freq_s1) > (2*math.pi)/(1/angular_freq_s1)  else (2*math.pi)/(1/angular_freq_s2) 
		# degree_shift = (time_shift/1000)*(360/(angular_freq_avg))
		degree_shift = (time_shift/1000)*(360/(angular_freq_max))

		degree_shift = degree_shift%360 if math.fabs(degree_shift) > 180 else degree_shift
		degree_shift = round(degree_shift,1)

		# if(DEBUG > 0): print("file s1 \t\t", "file s2 \t\t", "time_shift (s)\t", "angular_freq_s1\t", "angular_freq_s2\t", "degree_shift")
		# if(DEBUG > 0): print(args.s1_filename, "\t", args.s2_filename,"\t", time_shift/1000,"\t", round((2*math.pi)/(1/angular_freq_s1),3), "\t\t",round((2*math.pi)/(1/angular_freq_s2),3), "\t",degree_shift)

		if(DEBUG > 0 and not CSV_OUTPUT):
			global t1, min_time_shift, min_time_shift_row, total_time_shift

			total_time_shift = total_time_shift + time_shift/1000
			time_shift_arr.append(time_shift/1000)
			# t1.add_row([args.s1_filename, args.s2_filename, truncate_start, truncate_end, time_shift/1000, round((2*math.pi)/(1/angular_freq_s1),5),round((2*math.pi)/(1/angular_freq_s2),5), round(angular_freq_s1,5), round(angular_freq_s2,5), colored(degree_shift, "cyan")])
			t1.add_row([filename1, filename2, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, time_shift/1000, round((2*math.pi)/(1/angular_freq_s1),5),round((2*math.pi)/(1/angular_freq_s2),5), round(degree_shift,5)])

			if(min_time_shift == None or min_time_shift > math.fabs(time_shift/1000)):
				min_time_shift = math.fabs(time_shift/1000)
				min_time_shift_row = [filename1, filename2, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, colored(time_shift/1000, "red"), round((2*math.pi)/(1/angular_freq_s1),5),round((2*math.pi)/(1/angular_freq_s2),5), colored(round(degree_shift,5), "red")]

			print( "\n".join(t1.get_string().splitlines()[-2:]) )

		elif(CSV_OUTPUT):
			print(filename1, ",", filename2, ",", truncate_start, ",", truncate_end, ",", SMOOTH_CUT_OFF_FREQ,",", time_shift/1000, "," ,  round((2*math.pi)/(1/angular_freq_s1),5), ",",round((2*math.pi)/(1/angular_freq_s2),5), ",", round(angular_freq_s1,5), ",",  round(angular_freq_s2,5), ",", degree_shift%360)
		# else if(PRINT_TIME_SHIFT__PERIOD_S1__PERIOD_S2_ONLY):
		# 	print(args.s1_filename, ",", args.s2_filename, ",", time_shift/1000, "," ,  round((2*math.pi)/(1/angular_freq_s1),5), ",",round((2*math.pi)/(1/angular_freq_s2),5), ",", degree_shift)


	if(SHOW_PLOT):
		plt.legend()
		plt.show()

	return time_shift/1000, round((2*math.pi)/(1/angular_freq_s1),5), round((2*math.pi)/(1/angular_freq_s2),5)

# t1 = PrettyTable(["file s1             ", "file s2             ", "tstart  ", "tend     ","time_shift (s)", "period s1", "period s2", "angular_freq_s1", "angular_freq_s2", "degree_shift          "])
t1 = PrettyTable(["Sensor x                     ", "Sensor y                     ", "tstart ", "tend   ","fs_C_off","time_shift (s)", "period s1", "period s2", "degree_shift %360"])
t1.align["degree_shift %360"]="r"

print(t1)
min_time_shift = None
min_time_shift_row = None
total_time_shift = 0
first = True


ground_truth_timeshift_data = []
tested_s0_0_s1_timeshift_data = []
tested_s0_1_s1_timeshift_data = []

args = parser.parse_args()
truncate_start = 0 if(args.truncate_start == None) else  int(args.truncate_start)
truncate_end = 0 if(args.truncate_end == None) else  int(args.truncate_end)

s0_0 = np.loadtxt(args.s0_0_filename, delimiter=" ")
s0_1 = np.loadtxt(args.s0_1_filename, delimiter=" ")
s1 = np.loadtxt(args.s1_filename, delimiter=" ")


time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s0_1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s0_1_filename)
ground_truth_timeshift_data.append(time_shift)
time_shift, period_s0_0_tmp, period_s1_tmp = process_data(s0_0, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s1_filename)
tested_s0_0_s1_timeshift_data.append(time_shift)
time_shift, period_s0_1_tmp, period_s1_tmp = process_data(s0_1, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_1_filename, args.s1_filename)
tested_s0_1_s1_timeshift_data.append(time_shift)

# largest_period = np.max(np.array([period_s0_0_tmp, period_s0_1_tmp, period_s1_tmp]))
# smallest_period = np.min(np.array([period_s0_0_tmp, period_s0_1_tmp, period_s1_tmp]))
# median_period = np.min(np.array([period_s0_0_tmp, period_s0_1_tmp, period_s1_tmp]))
# chosen_period = smallest_period

chosen_period = ground_truth_timeshift_data[0]*4 # approx 90degrees X 4

if(X_CORR_BY_WINDOW):
	count = 0 # INCLUDES THE ONE FOR THE WHOLE SIGNAL COLLECTED
	start = truncate_start
	# factor = 2 # using a window one and hapf time the size of the period
	# for i in range(start, start + int(round(period_s1*1000)), int(round(period_s1*100))):
	# for i in range(start, truncate_end - int((period_s1*1000)), int((period_s1*1000)/3)):

	# for i in range(start, truncate_end - 3000, 3000):
	factor = 1.1
	print(colored("chosen period/cycle: ",'yellow'), end="")
	print(chosen_period*1000, int(chosen_period*1000*factor))
	# chosen_period = 2000
	print("chosen period override:", chosen_period)
	# step_window_size = int((chosen_period*1000*factor))
	step_window_size = int((chosen_period*1000)/4)
	step = int(step_window_size/3)

	for i in range(start, truncate_end - step_window_size, step):

	# for i in range(start, truncate_end - int((period_s1*1000)), int((period_s1*1000)/3)):
	# for i in range(start, 200000, int((period_s1*1000)) ):

	# for i in range(1,5):
		count = count + 2
		# start = start + randint(period_s1, period_s1*2)
		# truncate_start = start#i +
		# truncate_end = start + int(i * (round(period_s1*1000)))
		truncate_start = i
		truncate_end = i + step_window_size #int(round(period_s1*1000*factor))
		# print(truncate_start, truncate_end)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s0_1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s0_1_filename)
		ground_truth_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s1_filename)
		tested_s0_0_s1_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_1, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_1_filename, args.s1_filename)
		tested_s0_1_s1_timeshift_data.append(time_shift)
		t1.add_row(["-","-","-","-","-","-","-","-","-"])
		print( "\n".join(t1.get_string().splitlines()[-2:]) )

		truncate_end = int(i + 1.5*chosen_period*1000)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s0_1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s0_1_filename)
		ground_truth_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s1_filename)
		tested_s0_0_s1_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_1, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_1_filename, args.s1_filename)
		tested_s0_1_s1_timeshift_data.append(time_shift)
		t1.add_row(["-","-","-","-","-","-","-","-","-"])
		print( "\n".join(t1.get_string().splitlines()[-2:]) )

		truncate_end = int(i + 2*chosen_period*1000)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s0_1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s0_1_filename)
		ground_truth_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s1_filename)
		tested_s0_0_s1_timeshift_data.append(time_shift)
		time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_1, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_1_filename, args.s1_filename)
		tested_s0_1_s1_timeshift_data.append(time_shift)
		t1.add_row(["-","-","-","-","-","-","-","-","-"])
		print( "\n".join(t1.get_string().splitlines()[-2:]) )

		# truncate_end = int(i + 2.5*chosen_period*1000)
		# time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s0_1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s0_1_filename)
		# ground_truth_timeshift_data.append(time_shift)
		# time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_0, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_0_filename, args.s1_filename)
		# tested_s0_0_s1_timeshift_data.append(time_shift)
		# time_shift, period_s0_0_tmp, period_s0_1_tmp = process_data(s0_1, s1, truncate_start, truncate_end, SMOOTH_CUT_OFF_FREQ, args.s0_1_filename, args.s1_filename)
		# tested_s0_1_s1_timeshift_data.append(time_shift)
		# t1.add_row(["-","-","-","-","-","-","-","-","-"])
		# print( "\n".join(t1.get_string().splitlines()[-2:]) )

		# time_shift,_,_ = process_data(s1, s2, truncate_start, truncate_end,0.1)
		# process_data(s1, s2, truncate_start, truncate_end)
				# t.add_row([args.s1_filename, args.s2_filename, time_shift/1000, round((2*math.pi)/(1/angular_freq_s1),5),round((2*math.pi)/(1/angular_freq_s2),5), round(angular_freq_s1,5), round(angular_freq_s2,5), colored(degree_shift, "cyan")])

				# print(t)

	# t1.add_row(min_time_shift_row)
	# print(t1)

ground_truth_timeshift_data = np.array(ground_truth_timeshift_data)
tested_s0_0_s1_timeshift_data = np.array(tested_s0_0_s1_timeshift_data)
tested_s0_1_s1_timeshift_data = np.array(tested_s0_1_s1_timeshift_data)

tested_s0_0_s1_degreeshift_data = ((tested_s0_0_s1_timeshift_data/ground_truth_timeshift_data)*90)%360
tested_s0_1_s1_degreeshift_data = ((tested_s0_1_s1_timeshift_data/ground_truth_timeshift_data)*90)%360

# range from 180 to -180
# instead of 0 to 360 degrees
# otherwise 0 and 359 are far apart in calculation but close in actual distance
# harder to identify nodes that are further apart by degrees
#
# nodes futher apart, closer to 180/-180 must be identified by other stag on the board
#
tested_s0_0_s1_degreeshift_data[np.where(tested_s0_0_s1_degreeshift_data>180)] = tested_s0_0_s1_degreeshift_data[np.where(tested_s0_0_s1_degreeshift_data>180)]-360
tested_s0_1_s1_degreeshift_data[np.where(tested_s0_1_s1_degreeshift_data>180)] = tested_s0_1_s1_degreeshift_data[np.where(tested_s0_1_s1_degreeshift_data>180)]-360



print(ground_truth_timeshift_data)
print(tested_s0_0_s1_timeshift_data)
print( np.sort(np.round( tested_s0_0_s1_degreeshift_data ,1)).tolist(), np.average(tested_s0_0_s1_degreeshift_data) )
print("from s0_0, 25%: ", round(np.percentile(tested_s0_0_s1_degreeshift_data, lower_percentile),1), "\t, median:",colored(round(np.median(tested_s0_0_s1_degreeshift_data),1),'yellow'), "\t, 75%: ",round(np.percentile(tested_s0_0_s1_degreeshift_data, higher_percentile),1))
hist_s0_0_s1 = plt.hist(tested_s0_0_s1_degreeshift_data, bins=np.arange(-180,181,histogram_bin_size))#bins='auto')  # arguments are passed to np.histogram

print("Histogram highest bar at:", colored(hist_s0_0_s1[1][np.argmax(hist_s0_0_s1[0])],'cyan') ,"degrees")
# plt.show()

print(tested_s0_1_s1_timeshift_data)
print( np.sort(np.round( tested_s0_1_s1_degreeshift_data ,1)).tolist(), np.average(tested_s0_1_s1_degreeshift_data) )
print("from s0_1, 25%: ", round(np.percentile(tested_s0_1_s1_degreeshift_data, lower_percentile),1),"\t, median: ",colored(round(np.median(tested_s0_1_s1_degreeshift_data),1),'yellow'),"\t, 75%: ", round(np.percentile(tested_s0_1_s1_degreeshift_data, higher_percentile),1))
hist_s0_1_s1 = plt.hist(tested_s0_1_s1_degreeshift_data, bins=np.arange(-180,181,histogram_bin_size))  # arguments are passed to np.histogram

print("Histogram highest bar at:", colored(hist_s0_1_s1[1][np.argmax(hist_s0_1_s1[0])],'cyan') ,"degrees")
# plt.show()

# time_shift, period_s1, period_s2 = process_data(s1, s2, truncate_start, truncate_end, 1)
# time_shift, period_s1, period_s2 = process_data(s1, s2, truncate_start, truncate_end, 0.1)

# # print("##################################################################################")
# # t = PrettyTable(["file s1", "file s2", "time_shift (s)", "period s1", "period s2", "angular_freq_s1", "angular_freq_s2", "degree_shift"])

# if(X_CORR_BY_WINDOW):
# 	count = 2 # INCLUDES THE ONE FOR THE WHOLE SIGNAL COLLECTED
# 	start = truncate_start
# 	factor = 2 # using a window one and hapf time the size of the period
# 	# for i in range(start, start + int(round(period_s1*1000)), int(round(period_s1*100))):
# 	# for i in range(start, truncate_end - int((period_s1*1000)), int((period_s1*1000)/3)):
# 	for i in range(start, truncate_end - 3000, 3000):
# 	# for i in range(start, truncate_end - int((period_s1*1000)), int((period_s1*1000)/3)):
# 	# for i in range(start, 200000, int((period_s1*1000)) ):

# 	# for i in range(1,5):
# 		count = count + 2
# 		# start = start + randint(period_s1, period_s1*2)
# 		# truncate_start = start#i +
# 		# truncate_end = start + int(i * (round(period_s1*1000)))
# 		truncate_start = i
# 		truncate_end = i + 10000 #int(round(period_s1*1000*factor))
# 		# print(truncate_start, truncate_end)
# 		time_shift,_,_ = process_data(s1, s2, truncate_start, truncate_end,1)
# 		time_shift,_,_ = process_data(s1, s2, truncate_start, truncate_end,0.1)
# 		# process_data(s1, s2, truncate_start, truncate_end)
# 				# t.add_row([args.s1_filename, args.s2_filename, time_shift/1000, round((2*math.pi)/(1/angular_freq_s1),5),round((2*math.pi)/(1/angular_freq_s2),5), round(angular_freq_s1,5), round(angular_freq_s2,5), colored(degree_shift, "cyan")])

# 				# print(t)

# 	t1.add_row(min_time_shift_row)
# 	print(t1)
# 	print("AVERAGE TIME SHIFT", total_time_shift/count)
# 	print(time_shift_arr)
# 	try:
# 		prev_time_shift_array = np.load("prev_time_shift_array.npy")
# 		print("AVERAGE DEGREE SHIFT BASED ON TIME DIFFERENCE:", np.average(((np.array(time_shift_arr)/prev_time_shift_array)*90)%360)) # prev_time_shift_array is the known one (90%)
# 	except:
# 		traceback.print_exc()
# 	np.save("prev_time_shift_array",time_shift_arr)