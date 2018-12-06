import os
import numpy as np 

from essentia import *
from essentia.standard import *

import chords_standard as CS

def getChordsPerFrame(filename, sampleRate, frameSize, hopSize):
	"""
	Description:
		This function obtains a dictionary of chords in the form [startTime, endTime, labels] for the entire audio in filename.wav
		using Essentia backend for obtaining HPCP
		[Currently here are the ways to install Essentia : http://essentia.upf.edu/documentation/installing.html]

	Implementation :

		Under the hood Essentia does the following steps in a pipeline to obtain chords
		Short time fourier Transform -> Spectral peaks detected -> HPCP estimated -> HPCP averaged over each segment -> Matched to tone profiles (24 binary profiles 12 major 12 minor triad voicings)

		Essentia also gives a strength value which when = -1 indicates no chord detected.

	Parameters:
		* filename: Complete path of filename.wav to be processed for chord recognition
		* sampleRate: Sampling Rate for the filename (Needed by essentia modules)
		* frameSize: Window size for Short time processing for obtainin chromagram
		* hopSize: Short time processing is done every hopSize number of samples
	"""

	# Initializes the windowing module in essentia (http://essentia.upf.edu/documentation/reference/streaming_Windowing.html)
	# with parameters: type (type of window chosen here as hann), amount of zero padding.
	windowLength = 4*frameSize # includes padding
	run_windowing = Windowing(type='hann', zeroPadding=windowLength)

	# Initializes the fourier spectrum module in essentia (http://essentia.upf.edu/documentation/reference/streaming_Spectrum.html)
	# with parameters size (which indicates the size of audio processed by spectrum)
	run_spectrum = Spectrum(size=windowLength)

	# Initializes the spectral peaks module in essentia which returns peaks given a fourier spectrum
	# (http://essentia.upf.edu/documentation/reference/streaming_SpectralPeaks.html)
	run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                   maxFrequency=4000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")


	# Initializes the Harmonic Pitch Class Profile module in essentia (http://essentia.upf.edu/documentation/reference/streaming_HPCP.html)
	# parameter: harmonics indicates the number of harmonics for frequency contribution.
	# TODO: Experiment with other parameters of HPCP 
	run_hpcp = HPCP(harmonics=5, sampleRate = SAMPLE_RATE)

	# Initializes the chords detection module in essentia (http://essentia.upf.edu/documentation/reference/std_ChordsDetection.html)
	# Parameter windowSize, is the size of the window in seconds for estimating chords inside.
	run_chords_detection = ChordsDetection(hopSize = hopSize,
                                       sampleRate = sampleRate,
                                       windowSize = 0.5)

	# Initializes the essentia pool module which pools together frame wise caluclations of short time features.
	pool = Pool();


	# Loads filename.wav and applies an equal loudness filter on the same
	audio = MonoLoader(filename=filename)()
	audio = EqualLoudness()(audio)

	for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):

		# Running the pipeline of operations for each frame
		# Windowing--> Spectrum --> Picking spectrum peaks --> Calculating HPCP 
		frame = run_windowing(frame)
		spectrum = run_spectrum(frame)
		peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
		hpcp_frame = run_hpcp(peak_frequencies, peak_magnitudes)

		# Adding the calculated HPCP per frame into the pool
		pool.add('hpcp_frame', hpcp_frame)

	# Obtain chords per frame and strength per frame 
	chords_frame, strength_frame = run_chords_detection(pool['hpcp_frame'])

	# Bring the chords per frame and strength per frame into a list of tuples to return 
	chordPerFrame = [c for c in zip(chords_frame, strength_frame)]

	 # TODO: NOW CONVERT THESE PER FRAME VECTORS INTO [STARTTIME ENDTIME CHORDLABEL]

	return chordPerFrame