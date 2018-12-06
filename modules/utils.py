import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import templates
import matplotlib.pyplot as plt
import shutil
import csv

TEMPLATE_PATH = "D:/Home/Independent_Research/AutomaticChordEstimation/ACRProject/resources/templates/"

def matchTemplate(chromaVector=[1,0,0,0,1,0,0,1,0,0,0,0]):
	"""
	Description:

	Given a 12 dimensional chroma vector from a frame like [0.8, 0.6, 0.1 .... ]
	It returns best matching chord Label from the template file.

	"""

	#Normalizing Vector before matching
	chromaVector = chromaVector/np.max(chromaVector)
	chromaVector[chromaVector < 0.3] = 0

	templateFileName = os.path.join(TEMPLATE_PATH,'triads.json')

	if not os.path.isfile(templateFileName):
		templates.createSevenths()

	templateList = templates.load(templateFileName)

	max = 0
	chordTone = ['N', 0.0]
	for key in templateList:
		normalizingFactor = np.sum(templateList[key])
		c = np.correlate(templateList[key], chromaVector)/normalizingFactor
		if max < c:
			max = c
			chordTone = [key, c]

	return chordTone

def matchNoteLevel(chromaVector=[1,0,0,0,1,0,0,0,0,0,0,0]):
	"""
	Description:
		(EXPERIMENTAL)
	Given a 12 dimensional chroma vector from a frame like [0.8, 0.6, 0.1 .... ]
	It returns best matching chord Label from the template file.

	Experimenting with all types of templates 'allCombinations.json'

	"""

	#Normalizing Vector before matching
	chromaVector = chromaVector/np.max(chromaVector)
	chromaVector[chromaVector < 0.3] = 0

	templateList = templates.load(os.path.join(TEMPLATE_PATH,'allCombinations.json'))

	max = 0
	chordTone = ['N', 0.0]
	for key in templateList:
		normalizingFactor = np.sum(templateList[key])
		c = np.correlate(templateList[key], chromaVector)/normalizingFactor
		if max < c:
			max = c
			chordTone = [key, c]

	return chordTone

def getVoicingThreshold(audio, hopLength, method='rmsEnergy'):
	"""
	Description:
		This function takes an audio file (read as an array; example obtain from audio=librosa.load(filename.wav)) and returns a threshold of voicing energy.

	"""

	if method == 'rmsEnergy':
		rmsEnergy = librosa.feature.rmse(audio,
										frame_length=2*hopLength,
										hop_length=hopLength,
										center=True)

		peaks, _ = find_peaks(rmsEnergy[0], distance=30)
		threshold = 0.15 * np.mean([rmsEnergy[0][p] for p in peaks])

	return threshold

def getVoicingActivity(audioSegment, hopLength, method='rmsEnergy', threshold=0.0):
	"""
	Description:
		Given an audio file as an array (obtain from audio=librosa.load(filename.wav)) this returns a binary array
		[1,0,0,1,1,1,1,...] where each element indicates the voicing nature of a frame of audio (determined by hopLength)

	"""


	if method == 'rmsEnergy':
		if threshold == 0.0: print(f"ERROR:Threshold was NOT calculated before utils.getVoicingActivity().")

		rmsEnergy = librosa.feature.rmse(audioSegment,
								frame_length=2*hopLength,
								hop_length=hopLength,
								center=True)

		voicingActivity = [1] * rmsEnergy.shape[1]

		for k in range(rmsEnergy.shape[1]):
			if rmsEnergy[0][k] < threshold:
				voicingActivity[k] = 0

		# DEBUG plots:
		#plt.plot(rmsEnergy[0])
		#plt.hlines(thres, 0, len(rmsEnergy[0]))

	elif method == 'periodicity':
		frames = librosa.util.frame(audioSegment,
								    frame_length=2*hopLength,
									hop_length=hopLength)
		for indx in range(len(frames)):
			rxx = librosa.autocorrelate(frames[indx], max_size=10)

	else:
		raise ValueError(f"ERROR:Voicing Activity detection Method: {method} INVALID.")

	return voicingActivity

def getOnsets(audio, sampleRate, hopLength):
	"""
	Description:
		Given an audio file as an array (obtain from audio=librosa.load(filename.wav))
		this returns a list of sample numbers where onsets are found to occur.


	"""

	o_env = librosa.onset.onset_strength(audio, sampleRate,
										aggregate=np.mean,
										detrend=True,
										fmax=8000,
										n_mels=256,
										lag=50,
										hop_length=hopLength)

	onset_samples = librosa.onset.onset_detect(onset_envelope=o_env,
											units='samples',
											sr=sampleRate,
											hop_length=hopLength,
											backtrack=True,
											pre_max=20,
											post_max=20,
											pre_avg=100,
											post_avg=100,
											delta=0.1,
											wait=0.5 * int(sampleRate/hopLength))
	onset_boundaries = np.concatenate([[0], onset_samples, [len(audio)]])
	#onset_times = librosa.samples_to_time(onset_boundaries, sr = sampleRate)

	return onset_boundaries

def segmentByOnsets(onset_boundaries, minAudioLength):
	"""
	Description:
		This function takes a list of sample numbers that represents onset occurences in the audio and returns tuples
		[[start Segment, end Segment],...]
		 where each segment of audio between onsets starts and ends. It ignores segments of less than minAudioLength Samples

	Parameters:
		minAudioLength is in Samples (Obtain as minAudioLength = minimumTimeinSeconds * sampleRate)

	"""

	audioSegmentBoundaries = []
	k = 0

	while k < (len(onset_boundaries) - 1):

		start_boundary = onset_boundaries[k]
		# Skip onset boundaries that are less than minAudioLength Samples away from previous onset boundary
		while onset_boundaries[k+1] - onset_boundaries[k] < minAudioLength:
			k += 1
			if k == len(onset_boundaries)-1:
				k -= 1
				break

		end_boundary = onset_boundaries[k+1]

		segment = [start_boundary, end_boundary]
		audioSegmentBoundaries.append(segment)

		k += 1

	return audioSegmentBoundaries

def makeSingleFolderDataset(dataPath, finalPath):

	"""
	Helper function:

	This helper function can be used to read from the directory structure of IDMT-SMT-Guitar/dataset4
	 and get all the wav files and corresponding chord annotations

		filename.wav
		filename.csv

	into one common directory in finalPath/allData for processing

	"""

	INPUT_PATH = dataPath
	RESOURCE_PATH = os.path.join(finalPath, 'allData/')

	if not os.path.exists(RESOURCE_PATH):
		try:
		    os.makedirs(RESOURCE_PATH)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	dirs = [d for d in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH,d))]

	for d in dirs:
		for subdir, dirs, files in os.walk(os.path.join(INPUT_PATH,d)):
			for file in files:

				d = d.replace(" ","_")
				newName = f"{d}_{file}"

				if file.endswith(".wav"):
					#print(f"copying {file} to {newName}")
					shutil.copy(os.path.join(subdir,file), os.path.join(RESOURCE_PATH, newName))

				s = subdir.split('\\')[-1]
				if file.endswith(".csv") and not dirs and s == 'chords':
					shutil.copy(os.path.join(subdir,file), os.path.join(RESOURCE_PATH, newName))

	print("Successfully created single Folder for dataset")

def makeAnnotationsFolder(dataPath, annotationsPath):

	"""
	Helper Function:

	PARAMETERS:

		>>dataPath should contain all the
			filename.wav
			filename.csv

		>>annotationsPath is destination, where the csv files are processed into ground truth .lab files in this function
			filename.lab


	"""

	for subdir, dirs, files in os.walk(dataPath):
			for file in files:
				if file.endswith(".csv"):

					with open(os.path.join(subdir,file)) as csv_file:
						csv_reader = csv.reader(csv_file, delimiter=',')
						rows = []
						for row in csv_reader:
							rows.append(row)

					mirEvalRows = []
					entry = dict()
					for row in rows:
						c = row[1].split(":")
						# Check if this row contains chord Label or just beat number i.e Is there a term after ':' or len(c) > 1
						if len(c) > 1:

							if "startTime" in entry.keys():
								entry["endTime"] = row[0]
								mirEvalRows.append([entry["startTime"], entry["endTime"], entry["chordName"]])
								entry = dict()

								entry["startTime"] = row[0]
								entry["chordName"] = convertChordLabelToHARTE(c[1])

							else:
								entry["startTime"] = row[0]
								entry["chordName"] = convertChordLabelToHARTE(c[1])

					newfile = f"{file[:-4]}.lab"
					with open(os.path.join(annotationsPath, newfile), "w") as f:
						for i in range(len(mirEvalRows)):
							f.write(f"{mirEvalRows[i][0]} {mirEvalRows[i][1]} {mirEvalRows[i][2]}")
							if i < len(mirEvalRows)-1:
								f.write("\n")

	print("Successfully converted csv files into lab files")

def convertChordLabelToHARTE(chordLabel):
	"""
	Helper function:

	Description:
		This takes in a chordLabel as in the chord annotations in IDMT-SMT Guitar dataset and converts them into
		standard chord label syntax according to HARTE
		[Harte, C. (2010). Towards automatic extraction of harmony information from music signals (Doctoral dissertation)]

	"""

	if chordLabel == 'NC':
		return 'N'

	shortHandList = ['maj', 'min', 'dim', 'aug', 'maj7', 'min7', '7', 'dim7', 'hdim7', 'minmaj7', 'maj6', 'min6', '9', 'maj9', 'min9', 'sus2', 'sus4']

	names = chordLabel.split('/')
	rootName = names[0][0]
	modifier = ""
	shortHand = ""
	#inversion  = "" if len(names) == 1 else names[1]
	#intervals = ""

	if len(names[0]) == 1:
		return rootName

	if names[0][1] in ['b','#']:
	    modifier = names[0][1]
	    extraName = names[0][2:]
	else:
	    extraName = names[0][1:]

	bestShortHandLength = 0
	for c in shortHandList:
	    if is_subseq(c, extraName):
	        if bestShortHandLength < len(c):
	            shortHand = c
	            bestShortHandLength = len(c)

	# TODO: add extra intervals also to the HARTE label name, currently IDMT-SMT Guitar Dataset doesn't have them comma separated
	#intervals = extraName.split(shortHand)[1]

	# Collect all the parts of the HARTE label and return

	finalName =  rootName + modifier
	if shortHand != "":
		finalName = finalName + ":" + shortHand

	# Removed Inversion for now TODO: Raises Invalid Scale Degree error in Mir Eval
	#if inversion != "":
	#	finalName = finalName + "/"+ inversion

	return finalName



def is_subseq(x, y):
	"""
	Helper function:

	Returns bool if string x is a subsequence of string y

	"""
	x = list(x)
	for letter in y:
		if x and x[0] == letter:
			x.pop(0)

	return not x




def getEndPoints(audioSegment, sampleRate):
	#TODO: Find end points for an audio from here
	k = 1

def isHarmonicContent(audioSegment, sampleRate):
	f0_estimate = freq_from_AutoCorrelation(audioSegment, sampleRate)
	# Non Harmonic signal has non periodic autocorrelation and estimated fundamental will be high.
	# TODO: Show more evidence.
	if f0_estimate < 90:
		return True

	else:
		return False


def freq_from_AutoCorrelation(audioSegment, sampleRate):
	"""
	Description:
		This function estimates a single frequency from the autocorrelation.

		It first calculates the autocorrelation (keeping only half of it. Since autocorrelation is symmetric)
		Then it finds the


	"""
	corr = fftconvolve(audioSegment, audioSegment[::-1], mode='full')
	corr = corr[len(corr)//2:]

	d = np.diff(corr)
	start = find(d > 0)[0]

	peak = np.argmax(corr[start:]) + start
	px, py = parabolic(corr, peak)


	return sampleRate/px

def plotCepstrum(audioSegment, sampleRate):

	cepstrum = librosa.feature.mfcc(audioSegment, sampleRate, n_mfcc=50)
	return cepstrum


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)

    return (xv, yv)