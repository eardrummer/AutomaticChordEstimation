import os
import numpy as np 
import scipy
from collections import Counter
import librosa
from . import utils
from . import evaluation
import time

def getChromagramCQT(audioSegment, sampleRate, hopLength, tuning=True):
	"""
	Description: 
		This function calculates short time chromagram using a constant Q transform.
	
	Arguments:
		audioSegment: a length of audio read into an array (For example : use audioSegment = librosa.load(filename.wav))
		sampleRate: sample rate of the audio file
		hopLength: determines time resolution of the short time chromagram. Corresponds to number of samples captured in each frame
		tuning (bool): If true, it calculates a tuning estimate to adjust chromagram with (check librosa.estimate_tuning())

	Returns:
		short time chromagram: 12 x N dimensional (N = number of frames of size HopLength in audioSegment)
	"""

	if tuning:
		tuningDiff = librosa.estimate_tuning(y=audioSegment, sr=sampleRate, resolution=1e-2)
	else:
		tuningDiff = 0

	chroma_cq = librosa.feature.chroma_cqt(audioSegment,
											sr = sampleRate,
											C=None,
											hop_length=hopLength, 
											fmin=None, norm=np.inf, threshold=0.0, tuning=tuningDiff, 
											n_chroma=12, n_octaves=7, 
											window=None, 
											bins_per_octave=36, cqt_mode='full');

	return chroma_cq

def getChromagramSTFT(audioSegment, sampleRate, FFTLength, hopLength, tuning=True):
	"""
	Description: 
		This function calculates short time chromagram using an STFT of FFTLength every hopLength samples
	
	Arguments:
		audioSegment: a length of audio read into an array (For example : use audioSegment = librosa.load(filename.wav))
		sampleRate: sample rate of the audio file
		hopLength: determines time resolution of the short time chromagram. Corresponds to number of samples captured in each frame
		tuning (bool): If true, it calculates a tuning estimate to adjust chromagram with (check librosa.estimate_tuning())

	Returns:
		short time chromagram: 12 x N dimensional (N = number of frames of size HopLength in audioSegment)
	"""

	if tuning:
		tuningDiff = librosa.estimate_tuning(y=audioSegment, sr=sampleRate, resolution=1e-2)
	else:
		tuningDiff = 0

	chroma_stft = librosa.feature.chroma_stft(audioSegment,
											sr = sampleRate,
											S=None,
											n_fft=FFTLength,
											hop_length=hopLength, 
											norm=np.inf, tuning=tuningDiff);

	return chroma_stft

def getChordFromSequence(chordSubList):
	"""
	Description:
		This function gets a single chord which is the most common in a list of chords.

	Arguments:
		chordSubList : Its a list of tuples [[ChordName, confidence],...]
					   example:chordSubList =  [['N', 1.0], ['G:maj', array([0.7957629])], ['G:maj', array([0.81679583])]]

	Returns:
		a single tuple [ChordName, confidence] where chordName is the most commonly occuring chord in the sequence
													and confidence is the mean of the confidences of that chord occurence in the sequence

	Usage: (for the argument example)
		getChordFromSequence(chordSubList)
		>> ['G:maj', 0.806279365]

	"""

	# TODO: If 2 sequences are candidates for the most commonly occuring chord Label in the subList keep the one with greater mean confidence

	labels = [chordSubList[k][0] for k in range(len(chordSubList))]

	counter = Counter(labels)
	chord = counter.most_common(1)[0][0]

	confidences = []
	for k in range(len(labels)):
		if labels[k] == chord:
			confidences.append(chordSubList[k][1])

	confidence = np.mean(confidences)

	return [chord, confidence]

def aggregateChordList(chordList, hopLength):
	"""
	Description:
		Given a list of frame wise [chordNames, confidences], it returns a set of subsequences of contiguous chord occurences

	Arguments:
		chordList: frame wise list of tuples [[chordName, confidence],...]
					example: [['N', 1.0], ['G:maj', array([0.7957629])], ['G:maj', array([0.81679583])]]
		hopLength: hopLength in samples chosen for our frame-wise chord Labels

	Returns:
		a dictinoary chordLocations= {[['chordLabel': 'N',
									  'startSample': 0,
									  'endSample': 1*hopLength],

									  ['chordLabel': 'G:maj',
									  'startSample': 1*hopLength,
									  'endSample': 2*hopLength]]}

	"""

	chordLocations = []
	chordSubList = []

	indx = 0

	while indx < len(chordList):

		# PROCESSING the beginning if it is a chunk of NO CHORD region
		if chordList[indx][0] == 'N' and indx == 0:
			while chordList[indx][0] == 'N':
				indx += 1
				if indx == len(chordList): break
			chordLocations.append({'chordLabel':'N', 'startSample':0, 
								'endSample':(indx-1) * hopLength, 'confidence':1.0})
			continue

		# PROCESSING the next chunk of NO CHORD regions after a chord region. 

		# Now an 'N' represents the end of a valid chord region
		# So add the previous valid chord region into the chordLocations and then also add the 'N' no chord region until its end
		if chordList[indx][0] == 'N' and indx > 0:

			endFrame = indx

			while chordList[indx][0] == 'N':
				indx += 1
				if indx == len(chordList): break

			[c, confidence] = getChordFromSequence(chordSubList)
			chordSubList = []
			startSample = startFrame * hopLength
			endSample = endFrame * hopLength

			chordLocations.append({'chordLabel':c, 'startSample':startSample, 'endSample':endSample, 'confidence':confidence})

			# Add the no chord region to chordLocations
			chordLocations.append({'chordLabel':'N', 'startSample':endSample, 
								'endSample':(indx-1) * hopLength, 'confidence':1.0})

			continue

		# This means the chord region hasnt ended and we need to loop ahead until we find its end and then add it to our chordLocations list
		elif chordList[indx][0] != 'N':
			startFrame = indx
			startSample = startFrame * hopLength

			while chordList[indx][0] != 'N':
				chordSubList.append(chordList[indx])
				indx += 1
				if indx == len(chordList): 

					# Add the chord region since it reached end of List
					[c, confidence] = getChordFromSequence(chordSubList)
					chordLocations.append({'chordLabel':c, 'startSample':startSample, 'endSample':indx * hopLength, 'confidence':confidence})
					break

			continue



	# SubList contains a list of chord labels corresponding to a single event/onset and should be replaced by a single chordLabel
	return chordLocations

def aggregateAcrossOnsets(chordList, minTimeBetweenChange=0.1):

	"""
	Description:
		This function returns a list of chordLabels with their corresponding startTimes and endTimes with consecutively occuring chordLabels aggregated together.
		This is done after obtaining a chord list for every onset block and when you want to put them together.

	Arguments:
		chordList = [[startTime, endTime, chordLabel, confidence], ...]
		example >> chordList = [19.098412698412698, 19.655691609977325, 'D:maj', 0.6968047452708281], 
				  [19.61795918367347, 20.128798185941044, 'D:maj', 0.7748168849058874], 
				  [20.125895691609976, 20.77605442176871, 'D:maj', 0.7261305234465213], 
				  [20.73251700680272, 21.382675736961453, 'D:maj', 0.7523081577258233]

		minTimeBetweenChange in seconds is the minimum time of occurence for a chordLabel to be considered valid for including (~ 0.1 seconds)
	
	Returns:
		FinalChordList = [[startTime, endTie, chordLabel, meanConfidence], ... ]

		>>aggregateAcrossOnsets(chordList, 0.1)
		>>[19.098412698412698, 21.382675736961453, 'D:maj', 0.737515077837265]
		

	"""

	FinalChordList = []

	# chordChanges stores chordList element and its index everytime consecutive chordLabels change
	chordChanges = [[chordList[i], i] for i in range(len(chordList)) if chordList[i][2] != chordList[i-1][2]]
	
	for k in range(len(chordChanges) - 1):

		# Copy the chord label of the current chord
		chordLabel = chordChanges[k][0][2]

		# Copy the start time of the current chord
		startTime = chordChanges[k][0][0]

		# Get end time of the current chord from the last occurence of the chord in chordList. (index of next chord change - 1)
		endTime = chordList[chordChanges[k+1][1] - 1][1]

		# Calculate confidences as average of all the confidences
		confidence = np.mean([chordList[p][3] for p in range(chordChanges[k][1], chordChanges[k+1][1])])

		if np.float64(endTime) - np.float64(startTime) > minTimeBetweenChange:
			FinalChordList.append([startTime, endTime, chordLabel, confidence])


	FinalChordList.append(chordChanges[-1][0])

	return FinalChordList


def getChordsPerSegment(chromagram, voicingActivity, hopLength):
	"""
	Description:
		This function returns frame wise [[chordLabels, confidence], ... ]from chormagram. If voicingActivity for a frame = 0, it is given a label of 'N' = no chord

	Arguments:
		chromagram: This is the frame wise chromagram 12xN dimensions. (N is number of frames)
		voicingActivity: This is of N dimensions binary vector = [1,0,0,1,1,1, ...] where 0 represents No guitar activity and 1 represents guitar activity.
		hopLength: In samples, is the hop length used for obtaining each frame of chromagram

	Returns:
		a dictionary of chords aggregated in the segment:
														  {[['chordLabel': 'N',
														  'startSample': <starting sample number>,
														  'endSample': <ending sample number>,

														  ['chordLabel': 'G:maj',
														  'startSample': <starting sample number>,
														  'endSample': <ending sample number>]]}

														  Also See (aggregateChordList())
	"""

	chordList = []
	for indx in range(len(voicingActivity)):

		if voicingActivity[indx] == 0:
			c = ['N', 1.0]
		else:
			c = utils.matchTemplate(chromagram[:,indx])
		chordList.append(c)

	# DEBUG: Printing of chord List before aggregation
	#print(chordList)

	aggregatedList = aggregateChordList(chordList, hopLength)
	return aggregatedList

def getChords(filename, hopOnset=64, hopChroma=512, debug=False):
	"""
	Description:
		Given a filename it returns the chords identified in the file, in the form of a list of [startTime, endTime, chordLabel, confidence]

	Arguments:
		filename: Full path of the file to be processed : <FullPath>/filename.wav
		hopOnset: The hop size (in samples) to be considered for finding onsets in the file (must give a good time resolution)
		hopChroma: The hop size (in samples) to be considered for finding chromagrams (must give good frequency resolution)
		debug: (bool) When True: It displays intermediary chordLists. (for debugging purposes)

	Return:
		A list [[startTime, endTime, ChordLabel, confidence], ... ] for the entire file

	"""
	# This Flag is used as True if you want to aggregate all the chordLabels across Onsets at the end.
	# Experiment Results show this loses a lot of information when chords fluctuate rapidly (This should be fixed with a language model or transition probabilities instead)
	AGGREGATE_FLAG = True

	try:
		audio, sampleRate = librosa.load(filename)
	except:
		print(f"Couldnt Load {filename}")

	t1=time.time()

	# Finding Voicing Activity of the file. Returns a threshold under which to consider unvoiced
	voicing_threshold = utils.getVoicingThreshold(audio, hopChroma, method='rmsEnergy')

	# Segmenting Audio By Onsets; minimum length of a segment to be processed = 0.1seconds
	minAudioLength = 0.1 * sampleRate

	onset_boundaries = utils.getOnsets(audio, sampleRate, hopOnset)
	segment_boundaries = utils.segmentByOnsets(onset_boundaries, minAudioLength)

	chordList = []
	for segmentIndex in range(len(segment_boundaries)):

		# Segment Audio between Onset boundaries
		audioSegment = audio[segment_boundaries[segmentIndex][0]:segment_boundaries[segmentIndex][1]]

		# Obtain chromagram for audio segment: Choose one method below
		#chromagram = getChromagramCQT(audioSegment, sampleRate, hopChroma, tuning=True)
		chromagram = getChromagramSTFT(audioSegment, sampleRate, FFTLength=2*hopChroma, hopLength=hopChroma, tuning=True)

		# Obtain voicing activity by frame
		voicing_activity = utils.getVoicingActivity(audioSegment, hopChroma, method='rmsEnergy', threshold=voicing_threshold)

		# Obtain the estimated chords in each segment
		chordsInSegment = getChordsPerSegment(chromagram, voicing_activity, hopChroma)

		# Fix the startSamples and endSamples in absolute samples and convert to time
		for i in range(len(chordsInSegment)):
			chordsInSegment[i]['startSample'] += segment_boundaries[segmentIndex][0]
			chordsInSegment[i]['endSample'] += segment_boundaries[segmentIndex][0]

			chordsInSegment[i]['startTime'] = librosa.core.samples_to_time(chordsInSegment[i]['startSample'], sampleRate)
			chordsInSegment[i]['endTime'] = librosa.core.samples_to_time(chordsInSegment[i]['endSample'], sampleRate)

			#Append a list with the predicted chords
			chordList.append([chordsInSegment[i]['startTime'], chordsInSegment[i]['endTime'], chordsInSegment[i]['chordLabel'], chordsInSegment[i]['confidence']])

	if AGGREGATE_FLAG:
		FinalChordList = aggregateAcrossOnsets(chordList, minTimeBetweenChange=0.1)
	else:
		FinalChordList = chordList

	if debug:
		print(FinalChordList)

	print(f"Chord Estimation for {filename}\n ")
	print("Processing took %.02f seconds"%((time.time()-t1)))

	return FinalChordList


def ACR(dataPath, annotationsPath, outputPath):

	"""
	Description:
		This function performs chordRecognition on all "filenamw.wav" in the dataPath and writes them into outputPath, and uses the ground truth in "annotationsPath" to report CSR reults on accuracy of algorithm
		Three parts of this algorithm:

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
	REPROCESS_CHORDS_FLAG = True

	# If TRUE annotations will be obtained for every file. If False, pre-obtained annotations will be used
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
	method = "onset64_chroma2048_AGGREGATES_ChromaSTFT2x_acousticMicPop"
	#method = "temp"
	outputPath = os.path.join(outputPath, f"OUTPUT_LIBROSA_{method}")
	if not os.path.exists(outputPath):
		try:
		    os.makedirs(outputPath)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	# 1. Making Annotation .lab files from .csv files in mirex format
	print("1. Preparing Annotations Files")

	if len(os.listdir(annotationsPath)) == 0 or ANNOTATION_FLAG:

		utils.makeAnnotationsFolder(dataPath, annotationsPath)

	# 2. Chord Estimation Algorithm
	print("2. Chord Estimation Step")

	if len(os.listdir(outputPath)) == 0 or REPROCESS_CHORDS_FLAG:

		for subdir, dirs, files in os.walk(dataPath):
				for filename in files:
					f = filename.split("_")

					# For Debugging smaller segment of Dataset uncomment the lines below as per requirement
					#if filename.wndswith('.wav'):

					if filename.endswith('.wav') and f[1] == 'mic' and f[2] == 'pop':
					#if filename.endswith('.wav') and filename == 'acoustic_mic_pop_4_100BPM.wav':

						filePath = os.path.join(subdir, filename)

						chordEstimates = getChords(filePath, hopOnset=64, hopChroma=2048)
						newfile = f"{filename[:-4]}.lab"

						with open(os.path.join(outputPath, newfile), "w") as f:
							for i in range(len(chordEstimates)):
								f.write(f"{chordEstimates[i][0]} {chordEstimates[i][1]} {chordEstimates[i][2]}")
								if i < len(chordEstimates)-1:
									f.write("\n")
		print("*** Done Processing ***")

	# 3. Evaluating
	print("3. Evaluation Step")
	evaluation.CSRMeans(estFolder=outputPath, refFolder=annotationsPath, genre="")


#### Other Functions that might be useful ####################################################################

def getChordBoundaries(chordList, minFramesBetweenChange):
	"""
	Given a list of chordLabels as a list of tuples (ChordLabel, confidence)
	This returns a list of frame numbers where the ChordLabel changes considering the minimum number of frames between chord changes


	Example : chordList = 

	['G:min7', array([0.65304446])]
	['A#:min7', array([0.63790271])]
	['A#:maj7', array([0.54862729])]
	['A#:maj7', array([0.50102561])]
	['A#:maj7', array([0.44573409])]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]
	['A#:min7', array([0.64486818])]
	['A#:min7', array([0.75743437])]
	['C#:maj7', array([0.7487779])]
	['C#:maj7', array([0.76910111])]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]
	['N', 1.0]

	when passed with 
	getChordBoundaries(chordList, minFramesBetweenChange=5)
	>>[12, 18]

	getChordBoundaries(chordList, minFramesBetweenChange=2)
	>>[2, 5, 18]

	"""
	changeFrames = []
	#minTimeBetweenChange = 0.1 # Minimum length a chord plays in seconds
	#minFramesBetweenChange = int(minTimeBetweenChange * sampleRate / hopLength)

	numberOfFrames = len(chordList)

	for k in range(numberOfFrames):
		if k == 0:
			changeFrames.append(k)

		elif chordList[k][0] != chordList[changeFrames[-1]][0]:
			if changeFrames[-1] < k - minFramesBetweenChange:
				changeFrames.append(k)
			else:
				changeFrames.pop()

				if len(changeFrames) == 0:
					changeFrames.append(k)
				elif chordList[k][0] != chordList[changeFrames[-1]][0]:
					changeFrames.append(k)
                
	changeFrames.append(numberOfFrames-1)
	return changeFrames