import os
import numpy as np

# Libraries for loading audio and running a vamp plugin
import vamp
import librosa

def getChords(filename):
	"""
	Description:
		This function takes in the complete path of an audio file  "filename.wav" and returns a dictionary of entries like
		[startTime endTime chordLabel]

	"""

	#TODO: NNLS vamp plugin chord estimation. From :
	k = 1