#!/usr/bin/python3
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, correlate, decimate, filtfilt, wiener, medfilt, fftconvolve
from scipy.interpolate import spline
from scipy.interpolate import UnivariateSpline
import pickle

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
DEBUG=0

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

def normalize_regularize(sig):
	sig -= sig.mean(); sig /= (sig.std())
	# sig = normalize([sig], norm='l2')[0] # sklearn.preprocessing
	return sig

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


def calibrate(s):
	COL_X = 1
	COL_Y = 2
	COL_Z = 3
	COL_M = 4


	from peakdetect import peakdetect
	try:
		x_peaks_valleys = peakdetect(s[:,COL_X], lookahead=500)
		y_peaks_valleys = peakdetect(s[:,COL_Y], lookahead=500)
		z_peaks_valleys = peakdetect(s[:,COL_Z], lookahead=500)
		# print(x_peaks_valleys,  s[:,COL_X])#[x_peaks_valleys])


		# if(DEBUG): print(x_peaks_valleys)#, len(np.array(x_peaks_valleys[1])[:,1]))
		if(len(x_peaks_valleys[1]) > 0):
			x_min = np.min(np.array(x_peaks_valleys[1])[:,1])#np.min(s[:,COL_X])
		else:
			if(DEBUG): print("x not enough valleys")
			x_min = np.min(s[:,COL_X])
			pass
		# if(DEBUG): print(x_peaks_valleys)#, len(np.array(x_peaks_valleys[0])[:,1]))
		if(len(x_peaks_valleys[0]) > 0):
			x_max = np.max(np.array(x_peaks_valleys[0])[:,1])#s[:,COL_X])
		else:
			if(DEBUG): print("x not enough peaks")
			x_max = np.max(s[:,COL_X])
			pass

		# if(len(x_peaks_valleys[1]) == 0) and (len(x_peaks_valleys[0]) == 0):
		# 	s[:,COL_X] = 0
		# 	x_min = x_max = 0

		# if(DEBUG): print(y_peaks_valleys)#, len(np.array(y_peaks_valleys[1])[:,1]))
		if(len(y_peaks_valleys[1]) > 0):
			y_min = np.min(np.array(y_peaks_valleys[1])[:,1])
		else:
			if(DEBUG): print("y not enough valleys")
			y_min = np.min(s[:,COL_Y])
			pass
		# if(DEBUG): print(y_peaks_valleys)#, len(np.array(y_peaks_valleys[0])[:,1]))
		if(len(y_peaks_valleys[0]) > 0):
			y_max = np.max(np.array(y_peaks_valleys[0])[:,1])
		else:
			if(DEBUG): print("y not enough peaks")
			y_max = np.max(s[:,COL_Y])			
			pass

		# if(len(y_peaks_valleys[1]) == 0) and (len(y_peaks_valleys[0]) == 0):
		# 	s[:,COL_Y] = 0
		# 	y_min = y_max = 0

		# if(DEBUG): print(z_peaks_valleys)#, len(np.array(z_peaks_valleys[1])[:,1]))		
		if(len(z_peaks_valleys[1]) > 0):
			z_min = np.min(np.array(z_peaks_valleys[1])[:,1])
		else:
			if(DEBUG): print("z not enough valleys")
			z_min = np.min(s[:,COL_Z])
			pass
		# if(DEBUG): print(z_peaks_valleys)#, len(np.array(z_peaks_valleys[0])[:,1]))
		if(len(z_peaks_valleys[0]) > 0):
			z_max = np.max(np.array(z_peaks_valleys[0])[:,1])
		else:
			if(DEBUG): print("z not enough peaks")
			z_max = np.max(s[:,COL_Z])
			pass

		# if(len(z_peaks_valleys[1]) == 0) and (len(z_peaks_valleys[0]) == 0):
		# 	s[:,COL_Z] = 0
		# 	z_min = z_max = 0
	except:
		pass


	offset_x = -(0 - x_min - (x_max - x_min)/2)
	offset_y = -(0 - y_min - (y_max - y_min)/2)
	offset_z = -(0 - z_min - (z_max - z_min)/2)

	if(DEBUG): print(x_min, x_max, y_min, y_max, z_min, z_max)
	s[:,COL_X] = s[:,COL_X] - offset_x
	s[:,COL_Y] = s[:,COL_Y] - offset_y
	s[:,COL_Z] = s[:,COL_Z] - offset_z

	s[:,COL_M] = np.sqrt(s[:,COL_X]**2 + s[:,COL_Y]**2 + s[:,COL_Z]**2)

	return s

def spline(s_time,s_col):
	sp_s = UnivariateSpline(s_time, s_col, s=SPLINE_S)
	s_col = sp_s(s_time)
	return s_col

	# plt.plot(s1_data, label = s1_name + " spline")
	# plt.plot(s2_data, label = s2_name + " spline")

def trim_spline_trail(s, gap):
	s = s[0+gap:len(s)-1-gap]
	return s


def process(s, s_name, truncate_start, truncate_end, show_plot=True):
	# if(DEBUG): print(np.shape(s), s[:,0][0], s[:,0][0]+truncate_start, s[:,0][0]+truncate_end , s[:,0][len(s[:,0]) -1])
	s = s[np.where((s[:,0] >= (s[:,0][0]+truncate_start) ) & (s[:,0] <= (s[:,0][0]+truncate_end) ))]
	# if(DEBUG): print(np.shape(s), s[:,0][0], s[:,0][0]+truncate_start, s[:,0][0]+truncate_end , s[:,0][len(s[:,0]) -1])
	# s = s[truncate_start:truncate_end]
	for i in range(1,4):
		s[:,i] = smoothen_without_shift(s[:,i], smooth_cut_off_freq)
		if(SPLINE):
			s[:,i] = spline(s[:,0],s[:,i])
	
	if(SPLINE):
		s = trim_spline_trail(s, 1000)
		# s[:,i] = normalize_regularize(s[:,i])
		# plt.plot(s[:,0], s[:,i], label=axis[i-1])

	# s = trim_spline_trail(s, 200)
	# if(show_plot):
	# 	for i in range(1,5):
	# 		# s[:,i] = smoothen_without_shift(s[:,i], smooth_cut_off_freq)
	# 		# s[:,i] = normalize_regularize(s[:,i])
	# 		if i < 5:
	# 			plt.plot(s[:,0], s[:,i], label=" " + axis[i-1])
	# 	plt.legend()
	# 	plt.show()

	s = calibrate(s)

	if(show_plot):
		for i in range(1,5):
			# s[:,i] = smoothen_without_shift(s[:,i], smooth_cut_off_freq)
			# s[:,i] = normalize_regularize(s[:,i])
			if i < 5:
				plt.plot(s[:,0], s[:,i], label=" " + axis[i-1])
		plt.legend()
		plt.show()

	s[:,4] = np.sqrt(s[:,1]**2 + s[:,2]**2 + s[:,3]**2)
	# plt.plot(s[:,0], s[:,4], label=s_name + " " + axis[3])
	return s

def align(s1,s2, data_col_s1, data_col_s2):
	# global DATA_COL
	# if(DEBUG > 10): if(DEBUG): print("aligning...")

	# return s1,s2
	s1_t = s1[:,0]
	s2_t = s2[:,0]

	s1_data = s1[:,data_col_s1]
	s2_data = s2[:,data_col_s2]

	# if(DEBUG): print("1")
	# plt.plot(s1_data)#, label = s1_name + " spline")
	# plt.plot(s2_data)#, label = s2_name + " spline")





	s1_series = Series(s1_data,index=[datetime.datetime.fromtimestamp(ts) for ts in s1_t])
	s2_series = Series(s2_data,index=[datetime.datetime.fromtimestamp(ts) for ts in s2_t])

	# DATA_COL = 1

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

	# if(DEBUG > 10): if(DEBUG): print(min_t_possible, max_t_possible)
	s1 = s1[np.where((s1[:,0] >= min_t_possible) & (s1[:,0] <= max_t_possible))]
	s2 = s2[np.where((s2[:,0] >= min_t_possible) & (s2[:,0] <= max_t_possible))]
	

	# spline
	# # x = np.linspace(0,len(s1)/1000,len(s1))
	# sp_s1 = UnivariateSpline(s1[:,0], s1[:,DATA_COL])
	# sp_s2 = UnivariateSpline(s2[:,0], s2[:,DATA_COL])

	# s1_spline = sp_s1(s1[:,0])
	# s2_spline = sp_s2(s2[:,0])


	return s1, s2 #, s1_spline, s2_spline
# def align(s1,s2):
# 	s = UnivariateSpline(s1, s2, s=1)
# 	xs = linspace(-3, 3, len(s1)*1000)
# 	s1 = s(xs)
# 	s2 = s()


def compute_time_shift(file1, file2, truncate_start, truncate_end, show_plot=True):




	# print(file1)
	s1 = np.loadtxt(file1, delimiter=" ")
	s1 = process(s1, file1, truncate_start, truncate_end, show_plot)

	# print(file2)
	s2 = np.loadtxt(file2, delimiter=" ")
	s2 = process(s2, file2, truncate_start, truncate_end, show_plot)

	COL_M = 4
	s1_tm, s2_tm = align(s1, s2, COL_M, COL_M)
	# if(DEBUG): print(len(s1_m), len(s2_m))
	s1_m, s2_m = s1_tm[:,1], s2_tm[:,1]
	# s1_m = normalize_regularize(s1_m)
	# s2_m = normalize_regularize(s2_m)
	# xcorr = correlate(s1_m, s2_m)
	# # xcorr = fftconvolve(s1_data, s2_data)

	# time_shift = (len(xcorr)/2 - xcorr.argmax())

	xcorr = periodic_corr_np(s2_m, s1_m)
	xcorr_max = xcorr.argmax()
	if xcorr_max <= len(s1_m)/2:
		time_shift = xcorr_max
	else:
		time_shift = xcorr_max - len(s1_m)


	# if(DEBUG): print("time_shift", time_shift)

	# if(DEBUG): print(np.shape(s1_m))
	if(show_plot):
		plt.plot(normalize_regularize(s1_m), label="s1")
		plt.plot(normalize_regularize(s2_m), label="s2")
		# plt.plot((s1_m), label="s1")
		# plt.plot((s2_m), label="s2")
		
		plt.plot(normalize_regularize(xcorr), label = "xcorr")

		plt.legend()
		plt.show()

	return time_shift


np.set_printoptions(precision=2)



SPLINE = 0
SPLINE_S = 2
smooth_cut_off_freq = 1
axis = ['x','y','z','m']
# compute_time_shift(file1,file2,truncate_start,truncate_end)