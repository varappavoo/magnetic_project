#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate
from peakdetect import peakdetect
from termcolor import colored

from argparse import ArgumentParser

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f1", "--sensor1file", dest="s1_filename", help="collected data from s1", metavar="FILE", required=True)
parser.add_argument("-f2", "--sensor2file", dest="s2_filename", help="collected data from s2", metavar="FILE", required=True)
parser.add_argument("-t", "--truncateat", dest="truncate", help="truncate data collected at the end", required=False)

args = parser.parse_args()
truncate = int(args.truncate)

def running_mean(x, N):
	# src: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02

mean_samples = 2000
s1 = np.loadtxt(args.s1_filename, delimiter=" ")
s2 = np.loadtxt(args.s2_filename, delimiter=" ")


########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
########################################################
### ALIGN S1 AND S2 BEFORE XCORR
########################################################
print(np.min(s1[:,0]),np.max(s1[:,0]))
print(np.min(s2[:,0]),np.max(s2[:,0]))

# print(t[:,0], t[:,1], t[:,2])

s1_mov_avg = running_mean((s1[:,2]-np.min(s1)),mean_samples)
s2_mov_avg = running_mean((s2[:,2]-np.min(s2)),mean_samples)

########################################
# REGULARIZE DATASETS
# https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
########################################
s1_mov_avg -= s1_mov_avg.mean(); s1_mov_avg /= (3*s1_mov_avg.std())
s2_mov_avg -= s2_mov_avg.mean(); s2_mov_avg /= (3*s2_mov_avg.std())

##########################################
# OR NORMALIZE BETWEEN ZERO AND ONE
# ONE EXTREME CHANGE CAN HIJACK EVERYTHING?
##########################################
# dx_s1 = np.max(s1_mov_avg) - np.min(s1_mov_avg)
# s1_mov_avg = (s1_mov_avg-np.min(s1_mov_avg))/dx_s1
# dx_s2 = np.max(s2_mov_avg) - np.min(s2_mov_avg)
# s2_mov_avg = (s2_mov_avg-np.min(s2_mov_avg))/dx_s2

########################################

s1_t = s1[:,0][mean_samples-1:]
s2_t = s2[:,0][mean_samples-1:]

############################################
#NORMALIZE TIME
############################################
s1_t = s1_t - s1_t[0]
s2_t = s2_t - s2_t[0]
print("len s1", len(s1_t))
print("len s2", len(s2_t))


###############################################
## TRUNCATE
###############################################
TRUNCATE_AT = truncate
s1_t = s1_t[:TRUNCATE_AT]
s2_t = s2_t[:TRUNCATE_AT]
s1_mov_avg = s1_mov_avg[:TRUNCATE_AT]
s2_mov_avg = s2_mov_avg[:TRUNCATE_AT]

###########################################
# CORRELATION
###########################################


xcorr = correlate(s1_mov_avg, s2_mov_avg)
time_shift = len(xcorr)/2 - xcorr.argmax()

xcorr -= xcorr.mean(); xcorr /= xcorr.std() 
#downsampling for plotting
xcorr = xcorr[1:len(xcorr):2]

print(colored("XCorr: " + str(xcorr), "yellow"))
print(colored("Time shift (APPROX):" + str(time_shift), "yellow"))
print(len(xcorr), len(s1_mov_avg))


# plt.figure(1)
plt.figure(figsize=(13,9))
plt.subplot(211)
# plt.plot(s1[:,0], s1[:,2]-np.min(s1),  label='s1')
plt.plot(s1_t, s1_mov_avg,  label='s1_avg')
# plt.plot(s1[:,0][mean_samples-1:], s1_mov_avg,  label='s1_avg')
# plt.plot(s2[:,0], s2[:,2]-np.min(s2),  label='s2')
plt.plot(s2_t, s2_mov_avg,  label='s2_avg')

# plt.plot(np.arange(len(xcorr)) + s1_t[0], xcorr, label='xcorr')
if(len(s1_t) >= len(s2_t)):
	xcorr_t = s1_t[0:len(xcorr)]
else:
	xcorr_t = s2_t[0:len(xcorr)]
plt.plot(xcorr_t, xcorr, label='xcorr')

print("s1 peaks")
# peakind = signal.find_peaks_cwt(s1_mov_avg, np.arange(1,1000), signal.ricker)
# # peakind = signal.find_peaks_cwt(s1_mov_avg,  wavelet = signal.wavelets.daub, widths=10)
# print(peakind, s1[peakind])#, s1_mov_avg[peakind])

peaks_sci_s1, _ = find_peaks(s1_mov_avg, width=1000)
prominences = peak_prominences(s1_mov_avg, peaks_sci_s1)[0]
print("peaks and prominences")
print(peaks_sci_s1,end="")
print(colored(np.diff(peaks_sci_s1),"cyan"))	
print(prominences)

print("peak detect")
peaks_peakdetect_s1 = peakdetect(s1_mov_avg, lookahead=1000)
print(peaks_peakdetect_s1)
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
print(peaks_peakdetect_s1, end="")
print(colored(np.diff(peaks_peakdetect_s1),"cyan"))

plt.plot(s1_t[peaks_sci_s1], s1_mov_avg[peaks_sci_s1], "x")
# plt.plot(s1[peakind], s1_mov_avg[peakind], "o")
plt.plot(s1_t[peaks_peakdetect_s1], s1_mov_avg[peaks_peakdetect_s1], "+")

# print(signal.peak_prominences(s1_mov_avg,peakind)[0])
# # plt.plot(s1[:,0][mean_samples-1:][peakind], s1_mov_avg[peakind], "x")

print("s2 peaks")
# peakind = signal.find_peaks_cwt(s2_mov_avg, np.arange(1,1000), signal.ricker)
# print(peakind, s2[peakind])

peaks_sci_s2, _ = find_peaks(s2_mov_avg, width=1000)
prominences = peak_prominences(s2_mov_avg, peaks_sci_s2)[0]
print("peaks and prominences")
print(peaks_sci_s2, end="")
print(colored(np.diff(peaks_sci_s2),"cyan"))	
print(prominences)


print("peak detect")
peaks_peakdetect_s2 = peakdetect(s2_mov_avg, lookahead=1000)


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
print(peaks_peakdetect_s2, end="")
print(colored(np.diff(peaks_peakdetect_s2),"cyan"))	

plt.plot(s2_t[peaks_sci_s2], s2_mov_avg[peaks_sci_s2], "x")
# plt.plot(s2[peakind], s2_mov_avg[peakind], "o")
plt.plot(s2_t[peaks_peakdetect_s2], s2_mov_avg[peaks_peakdetect_s2], "+")

peaks_sci_diff = (peaks_sci_s1 - peaks_sci_s2)
print(colored(peaks_sci_diff,"red"),end="")
print(colored(np.sum(peaks_sci_diff)/len(peaks_sci_diff),"green"))

plt.legend()
plt.show()

# # if(len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2)):
# # peaks_diff = (peaks_peakdetect_s1 - peaks_peakdetect_s2)
# # print(colored(peaks_diff,"red"),end="")
# # print(colored(np.sum(peaks_diff)/len(peaks_diff),"green"))

# # elif (len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2))::
# # # print(signal.peak_prominences(s2_mov_avg,peakind)[0])

# # peaks, _ = find_peaks(s2_mov_avg)
# # prominences = peak_prominences(s2_mov_avg, peaks)[0]
# # print("peaks and prominences")
# # print(peaks)
# # print(prominences)


# # plt.plot(s2[:,0][mean_samples-1:][peakind], s2_mov_avg[peakind], "o")

# plt.legend()
# plt.show()


# # # b, a = signal.butter(8, 0.125) # cut off 125Hz
# # b, a = signal.butter(8, 0.015)
# # print(a)
# # print(b)
# # plt.plot(s1[:,0], s1[:,2])
# # plt.plot(s1[:,0], signal.filtfilt(b, a , s1[:,2],  padlen=10))
# # plt.show()
# # # plt.plot(t, )

# # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# # plt.subplot(212)
# # plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# # plt.show()
