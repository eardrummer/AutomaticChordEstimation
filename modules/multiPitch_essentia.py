import os
import sys
import numpy as np

from essentia import *
from essentia.standard import *

import chords_standard as CS

def getMultiPitch(filename, sampleRate):
	"""
	Description:
		This function obtains multiple f0 estimates from a segment of audio using essentia's MultiPitch Melodia as backend.
		ref - 1 - [J. Salamon and E. Gómez, "Melody extraction from polyphonic music signals using pitch contour characteristics," IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.]

	Implementation:

		Under the hood essentia follows the Melodia multi pitch algorithm as in [1]
		TODO: Add explaination of the main blocks of the algorithm

	Paramteres:
		* filename : the complete path to the audio file "filename.wav"
		* sampleRate : the sampling rate of the audio file.

	Returns:
		pitch (vector_vector_real): which is an array of lists that contain the f0 candidates per frame in Hz.


	"""


	run_multi_pitch_melodia = MultiPitchMelodia(minFrequency=50,
												maxFrequency=1500)