import os
import numpy as np

#Used for chord recognition operations on the audio
import madmom
from madmom.audio.chroma import DeepChromaProcessor

import utils
import evaluation
import time

def initializeMadmom(fmin, fmax):
	"""
	Description:
		This function Initializes the deep chroma processor functions and the recognition module built inside madmom
		and returns the functions that can be used for these API calls

	Parameters:
		* fmin: This is the minimum frequency of the filterbank
		* fmax: This is the maximum frequency of the filterbank
	"""
	dcp = DeepChromaProcessor(fmin=fmin,
							  fmax=fmax,
							  unique_filters=True)
	decode = madmom.features.chords.DeepChromaChordRecognitionProcessor()

	return dcp, decode

def getChords(filename):
	"""
	Description:
		This function obtains a dictionary of entries like [startTime endTime chordLabel] for the entire audio in filename.wav
		using Madmom backend.

	"""

	# This initializes the madmom modules for chroma,
	# fmin and fmax are the minimum and maximum frequencies of the filterbank respectively.
	# Link to documentation: https://madmom.readthedocs.io/en/latest/modules/audio/chroma.html
	dcp = DeepChromaProcessor(fmin=65,
							  fmax=2000,
							  unique_filters=True)

	# This initializes the madmom module for chord recognition from deep chroma obtained in the previous step
	# They allow us to input our own Conditional Random Field (CRF) model to be used.
	# This only returns maj and min chords (No sevenths/inversions etc identified)
	# Link to documentation: https://madmom.readthedocs.io/en/latest/modules/features/chords.html
	crfProcessor = madmom.features.chords.CRFChordRecognitionProcessor()

	# This initializes the madmom module for chord recognition from CNN according to Filip[2016]
	#  Filip Korzeniowski and Gerhard Widmer, “A Fully Convolutional Deep Auditory Model for Musical Chord Recognition”, Proceedings of IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2016
	cnnProcessor = madmom.features.chords.CNNChordFeatureProcessor()


	# IMPLEMENTATION DETAILS:
	# Here we Implement chord recognition using madmom modules in the following way
	# AudioFile --> CNN features --> Conditional Random Field --> Chord labels
	# Link in documentation: https://madmom.readthedocs.io/en/latest/modules/features/chords.html

	# This is used to calculate time taken by the function
	t1 = time.time()

	# Obtaining CNN Features
	features = cnnProcessor(filename)

	# Obtains chord labels as [startTime endTime chordLabel] from CNN features using CRF model
	chords = crfProcessor(features)

	print(f"Chord Estimation for {filename}\n ")
	print("Processing took %.02f seconds"%((time.time()-t1)))

	return chords


def ACR(dataPath, annotationsPath, outputPath):
	"""
	Description:
		This function performs automatic chord estimation on all the "filename.wav" in the dataPath and writes them into outputPath
		It then validates against the corresponding "filename.lab" annotations in annotationsPath and reports an average CSR (Chord Symbol Recall)

	Implementation Details:
		This function has 3 stages

			1. Annotations stage: This takes the "filename.csv" files in dataPath and creates "<annotaionsPath>/filename.lab" files in annotationsPath according to mirex guidelines
			2. Chord Estimation stage: This writes the estimated chords into a "<outputPath>/filename.lab"
			3. Evaluation stage: This evaluates the estimated files with the annotation files and reports average CSR results


	Parameters:
		>>dataPath must have all the

			filename.wav
			filename.csv

		pairs that need to be analyzed for chord recognition

		>>annotationsPath has

			filename.lab

		which are created from annotation (CSV) files which have the ground truth to evaluate estimated chords against and report accuracy

		>>outputPath is where estimated files

			est_filename.lab

		are created and logged.


	"""

	# If TRUE chords will be estimated for every file. If False, chords will be obtained from previous estimations
	# If False, algorithm skips the Chord Estimation stage
	REPROCESS_CHORDS_FLAG = False

	# If TRUE annotations will be obtained for every file. If False, pre-obtained annotations will be used
	# If False, algorithm skips the Annotations stage
	ANNOTATION_FLAG = False

	if not os.path.exists(annotationsPath):
		try:
		    os.makedirs(annotationsPath)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	if not os.path.exists(outputPath):
		try:
		    os.makedirs(outputPath)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	# Create Sub Folder in OUTPUTS by name of method to compare different methods in the future
	#method = "acousticMicReggae"
	method = ""
	outputPath = os.path.join(outputPath, f"OUTPUT_MADMOM_{method}")

	if not os.path.exists(outputPath):
		try:
		    os.makedirs(outputPath)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	# 1. Making Annotation .lab files from .csv files in mirex format
	print("1. Preparing Annotations Files")

	if len(os.listdir(annotationsPath)) == 0 or ANNOTATION_FLAG:

		# Creates annotation files as filename.lab for every filename.csv in dataPath.
		utils.makeAnnotationsFolder(dataPath, annotationsPath)

	# 2. Chord Estimation Algorithm
	print("2. Chord Estimation Step")

	if len(os.listdir(outputPath)) == 0 or REPROCESS_CHORDS_FLAG:

		for subdir, dirs, files in os.walk(dataPath):
				for filename in files:

					# This splits the filename into its components to easily use only a subset of the files. example only process the 'mic','pop' files
					f = filename.split("_")

					# For Debugging smaller segment of Dataset uncomment the lines below as per requirement
					if filename.endswith('.wav'):
					#if filename.endswith('.wav') and f[1] == 'mic' and f[2] == 'reggae':
					#if filename.endswith('.wav') and filename == 'acoustic_mic_pop_5_78BPM.wav':

						filePath = os.path.join(subdir, filename)

						# This function returns a dictionary of chord labels with start and end times from filePath
						chords = getChords(filePath)

						# Here we use list comprehension to convert the dictionary into a list of [startTime endTime Labels]
						chordEstimates = [[chords['start'][i], chords['end'][i], chords['label'][i]]  for i in range(len(chords))]

						newfile = f"{filename[:-4]}.lab"

						with open(os.path.join(outputPath, newfile), "w") as f:
							for i in range(len(chordEstimates)):
								f.write(f"{chordEstimates[i][0]} {chordEstimates[i][1]} {chordEstimates[i][2]}")
								if i < len(chordEstimates)-1:
									f.write("\n")
		print("*** Done Processing ***")


	# 3. Evaluating
	print("3. Evaluation Step")
	evaluation.CSRMeans(estFolder=outputPath, refFolder=annotationsPath, genre="SS")