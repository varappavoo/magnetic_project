#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from peakdetect import peakdetect
from termcolor import colored

from argparse import ArgumentParser

parser = ArgumentParser(description='plot...?', epilog="?")
parser.add_argument("-f1", "--sensor1file", dest="s1_filename", help="collected data from s1", metavar="FILE", required=True)
parser.add_argument("-f2", "--sensor2file", dest="s2_filename", help="collected data from s2", metavar="FILE", required=True)

args = parser.parse_args()

def running_mean(x, N):
	# src: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02

mean_samples = 500
s1 = np.loadtxt(args.s1_filename, delimiter=" ")
s2 = np.loadtxt(args.s2_filename, delimiter=" ")
# print(t[:,0], t[:,1], t[:,2])

s1_mov_avg = running_mean((s1[:,2]-np.min(s1)),mean_samples)
s2_mov_avg = running_mean((s2[:,2]-np.min(s2)),mean_samples)

s1 = s1[:,0][mean_samples-1:]
s2 = s2[:,0][mean_samples-1:]
# plt.figure(1)
plt.figure(figsize=(13,9))
plt.subplot(211)
# plt.plot(s1[:,0], s1[:,2]-np.min(s1),  label='s1')
plt.plot(s1, s1_mov_avg,  label='s1_avg')
# plt.plot(s1[:,0][mean_samples-1:], s1_mov_avg,  label='s1_avg')
# plt.plot(s2[:,0], s2[:,2]-np.min(s2),  label='s2')
plt.plot(s2, s2_mov_avg,  label='s2_avg')

print("s1 peaks")
# peakind = signal.find_peaks_cwt(s1_mov_avg, np.arange(1,1000), signal.ricker)
# # peakind = signal.find_peaks_cwt(s1_mov_avg,  wavelet = signal.wavelets.daub, widths=10)
# print(peakind, s1[peakind])#, s1_mov_avg[peakind])

peaks, _ = find_peaks(s1_mov_avg, width=1000)
prominences = peak_prominences(s1_mov_avg, peaks)[0]
print("peaks and prominences")
print(peaks)
print(prominences)

print("peak detect")
peaks_peakdetect_s1 = peakdetect(s1_mov_avg, lookahead=1000)
# print(peaks_peakdetect_s1)
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

plt.plot(s1[peaks], s1_mov_avg[peaks], "x")
# plt.plot(s1[peakind], s1_mov_avg[peakind], "o")
plt.plot(s1[peaks_peakdetect_s1], s1_mov_avg[peaks_peakdetect_s1], "+")

# print(signal.peak_prominences(s1_mov_avg,peakind)[0])
# # plt.plot(s1[:,0][mean_samples-1:][peakind], s1_mov_avg[peakind], "x")

print("s2 peaks")
# peakind = signal.find_peaks_cwt(s2_mov_avg, np.arange(1,1000), signal.ricker)
# print(peakind, s2[peakind])

peaks, _ = find_peaks(s2_mov_avg, width=1000)
prominences = peak_prominences(s2_mov_avg, peaks)[0]
print("peaks and prominences")
print(peaks)
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

plt.plot(s2[peaks], s2_mov_avg[peaks], "x")
# plt.plot(s2[peakind], s2_mov_avg[peakind], "o")
plt.plot(s2[peaks_peakdetect_s2], s2_mov_avg[peaks_peakdetect_s2], "+")

# if(len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2)):
# peaks_diff = (peaks_peakdetect_s1 - peaks_peakdetect_s2)
# print(colored(peaks_diff,"red"),end="")
# print(colored(np.sum(peaks_diff)/len(peaks_diff),"green"))

# elif (len(peaks_peakdetect_s1) == len(peaks_peakdetect_s2))::
# # print(signal.peak_prominences(s2_mov_avg,peakind)[0])

# peaks, _ = find_peaks(s2_mov_avg)
# prominences = peak_prominences(s2_mov_avg, peaks)[0]
# print("peaks and prominences")
# print(peaks)
# print(prominences)


# plt.plot(s2[:,0][mean_samples-1:][peakind], s2_mov_avg[peakind], "o")

plt.legend()
plt.show()

# plt.plot(t, )

# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()
