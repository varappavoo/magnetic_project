#!/usr/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate, decimate, filtfilt
from scipy.interpolate import spline

from peakdetect import peakdetect
from termcolor import colored
from pandas import Series
import datetime
from prettytable import PrettyTable
from argparse import ArgumentParser

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f1", "--sensor1file", dest="s1_filename", help="collected data from s1", metavar="FILE", required=True)
parser.add_argument("-f2", "--sensor2file", dest="s2_filename", help="collected data from s2", metavar="FILE", required=True)
parser.add_argument("-tstart", "--truncatestart", dest="truncate_start", help="truncate data collected at the end", required=False)
parser.add_argument("-tend", "--truncateend", dest="truncate_end", help="truncate data collected at the end", required=False)

args = parser.parse_args()
truncate_start = 0 if(args.truncate_start == None) else  int(args.truncate_start)
truncate_end = 0 if(args.truncate_end == None) else  int(args.truncate_end)

TIMESTAMP_COL = 0
DATA_COL = 1
DEBUG = 3
SHOW_PLOT = 1
COMPUTE_ANGULAR_FREQUENCY_DEGREE_DRIFT = 1
SMOOTH_WO_SHIFT = 1
NORMALIZE = 1

peak_width = 1000
mean_samples = 1

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


def smoothen_without_shift(sig):
	b, a = signal.butter(2, 0.0001) # a lowpass Butterworth filter with a cutoff of 0.125 times the Nyquist rate, or 125 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
	# b, a = signal.butter(4, 100, 'low', analog=True)
	fgust = signal.filtfilt(b, a, sig, method="gust")
	return fgust
	# fgust -= fgust.mean(); fgust /= (3*fgust.std())
	# plt.plot(s1_t,fgust, label='gust', linestyle="--")

########################################
# REGULARIZE DATASETS
# https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
########################################
def normalize_regularize(sig):
	sig -= sig.mean(); sig /= (3*sig.std())
	return sig


s1 = np.loadtxt(args.s1_filename, delimiter=" ")
s2 = np.loadtxt(args.s2_filename, delimiter=" ")

# if(DEBUG > 10): print(np.min(s1[:,0]),np.max(s1[:,0]))
# if(DEBUG > 10): print(np.min(s2[:,0]),np.max(s2[:,0]))

########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
s1,s2 = align(s1,s2)

s1_t = s1[:,0]
s2_t = s2[:,0]

s1_data = s1[:,DATA_COL]
s2_data = s2[:,DATA_COL]

# s1_data = smoothen_without_shift(s1_data)
# s2_data = smoothen_without_shift(s2_data)
if(SMOOTH_WO_SHIFT):
	s1_data = smoothen_without_shift(s1_data)
	s2_data = smoothen_without_shift(s2_data)
if(NORMALIZE):
	s1_data = normalize_regularize(s1_data)
	s2_data = normalize_regularize(s2_data)

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


###########################################
# CORRELATION
###########################################

if(DEBUG > 10): print("len " + args.s1_filename, len(s1_t), len(s1_data))
if(DEBUG > 10): print("len " + args.s2_filename, len(s2_t), len(s2_data))
xcorr = correlate(s1_data, s2_data)
time_shift = int(len(xcorr)/2) - xcorr.argmax()

xcorr -= xcorr.mean(); xcorr /= xcorr.std() 
#downsampling for plotting
xcorr = xcorr[1:len(xcorr):2]

# if(DEBUG > 10): print(colored("XCorr: " + str(xcorr), "yellow"))
# if(DEBUG > 10): print(colored("Time shift (APPROX - IN TERMS OF SAMPLES):" + str(time_shift), "yellow"))
# if(DEBUG > 10): print(colored("Time shift (COULD BE APPROX):" + str(s1_t[time_shift] - s1_t[0]), "yellow"))
if(DEBUG > 10): print(colored("Time shift:" + str(time_shift) + " ms", "yellow"))
# if(DEBUG > 10): print(len(xcorr), len(s1_data))


# plt.figure(1)
plt.figure(figsize=(13,9))
plt.subplot(211)
# plt.plot(s1[:,0], s1[:,2]-np.min(s1),  label='s1')
plt.plot(s1_t, s1_data,  label=args.s1_filename)
# plt.plot(s1[:,0][mean_samples-1:], s1_data,  label='s1_avg')
# plt.plot(s2[:,0], s2[:,2]-np.min(s2),  label='s2')
plt.plot(s2_t, s2_data,  label=args.s2_filename)

# plt.plot(np.arange(len(xcorr)) + s1_t[0], xcorr, label='xcorr')
if(len(s1_t) >= len(s2_t)):
	xcorr_t = s1_t[0:len(xcorr)]
else:
	xcorr_t = s2_t[0:len(xcorr)]
# plt.plot(xcorr_t, xcorr, label='xcorr')


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

plt.plot(s1_t[peaks_sci_s1_idx], s1_data[peaks_sci_s1_idx], "x")
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

plt.plot(s2_t[peaks_sci_s2_idx], s2_data[peaks_sci_s2_idx], "x")
# plt.plot(s2[peakind], s2_data[peakind], "o")
# plt.plot(s2_t[peaks_peakdetect_s2], s2_data[peaks_peakdetect_s2], "X")

# TAKE 10 INDICES ??????????????
# INDICES INVOLVED IN COMPUTING DIFF IS 10?
try:
	peaks_sci_diff = (peaks_sci_s1_idx[0:6] - peaks_sci_s2_idx[0:6])
	if(DEBUG >2): print(colored(peaks_sci_diff,"red"), np.mean(peaks_sci_diff))
except:
	print("ERROR TRYING TO COMPUTE PEAKS DIFF", len(peaks_sci_s1_idx), len(peaks_sci_s2_idx))
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
	degree_shift = (time_shift/1000)*(360/(angular_freq_avg))

	# if(DEBUG > 0): print("file s1 \t\t", "file s2 \t\t", "time_shift (s)\t", "angular_freq_s1\t", "angular_freq_s2\t", "degree_shift")
	# if(DEBUG > 0): print(args.s1_filename, "\t", args.s2_filename,"\t", time_shift/1000,"\t", round((2*math.pi)/(1/angular_freq_s1),3), "\t\t",round((2*math.pi)/(1/angular_freq_s2),3), "\t",degree_shift)

	if(DEBUG > 0):
		t = PrettyTable(["file s1", "file s2", "time_shift (s)", "period s1", "period s2", "angular_freq_s1", "angular_freq_s2", "degree_shift"])
		t.add_row([args.s1_filename, args.s2_filename, time_shift/1000, (2*math.pi)/(1/angular_freq_s1),  (2*math.pi)/(1/angular_freq_s2), round((2*math.pi)/(1/angular_freq_s1),3),round((2*math.pi)/(1/angular_freq_s2),3), degree_shift])
		print(t)


if(SHOW_PLOT):
	plt.legend()
	plt.show()



	###################################################

# peaks_sci_periodigram_s1_index, _ = find_peaks(periodogram_s1, width=10)
# peaks_sci_periodigram_s2_index, _ = find_peaks(periodogram_s2, width=10)

# if(DEBUG > 10): print(args.s1_filename,"Prominent periods",(1/freqs[peaks_sci_periodigram_s1_index])*2*math.pi)
# if(DEBUG > 10): print(args.s2_filename,"Prominent periods",(1/freqs[peaks_sci_periodigram_s2_index])*2*math.pi)

# plt.plot((1/freqs)*2*math.pi, periodogram_s1, label='periodigram ' + args.s1_filename)
# plt.plot(((1/freqs[peaks_sci_periodigram_s1_index])*2*math.pi), periodogram_s1[peaks_sci_periodigram_s1_index], 'X')
# plt.plot((1/freqs)*2*math.pi, periodogram_s2, label='periodigram ' + args.s2_filename)
# plt.plot(((1/freqs[peaks_sci_periodigram_s2_index])*2*math.pi), periodogram_s2[peaks_sci_periodigram_s2_index], 'X')

# plt.legend()
# plt.show()





# # check this : https://en.wikipedia.org/wiki/Angular_frequency

# # # if(len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2)):
# # # peaks_diff = (peaks_peakdetect_s1 - peaks_peakdetect_s2)
# # # if(DEBUG > 10): print(colored(peaks_diff,"red"),end="")
# # # if(DEBUG > 10): print(colored(np.sum(peaks_diff)/len(peaks_diff),"green"))

# # # elif (len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2))::
# # # # if(DEBUG > 10): print(signal.peak_prominences(s2_data,peakind)[0])

# # # peaks, _ = find_peaks(s2_data)
# # # prominences = peak_prominences(s2_data, peaks)[0]
# # # if(DEBUG > 10): print("peaks and prominences")
# # # if(DEBUG > 10): print(peaks)
# # # if(DEBUG > 10): print(prominences)


# # # plt.plot(s2[:,0][mean_samples-1:][peakind], s2_data[peakind], "o")

# # plt.legend()
# # plt.show()


# # # # b, a = signal.butter(8, 0.125) # cut off 125Hz
# # # b, a = signal.butter(8, 0.015)
# # # if(DEBUG > 10): print(a)
# # # if(DEBUG > 10): print(b)
# # # plt.plot(s1[:,0], s1[:,2])
# # # plt.plot(s1[:,0], signal.filtfilt(b, a , s1[:,2],  padlen=10))
# # # plt.show()
# # # # plt.plot(t, )

# # # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# # # plt.subplot(212)
# # # plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# # # plt.show()
