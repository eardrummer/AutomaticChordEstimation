import os
import json
import itertools

TEMPLATE_PATH = "D:/Home/Independent_Research/AutomaticChordEstimation/ACRProject/resources/templates/"

def createTriads(templatePath=TEMPLATE_PATH):
	"""
	Description:
		This function creates binary templates for triads
		The 12 dimensional binary templates are indexed from C. 

		Example C:maj = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

		This writes the templates into a json file 'triads.json' in TEMPLATE_PATH

		(Experimental: Choosing template value at root = 1, and other values = 0.9, to give preference to root in matching chordLabels)

	"""
	template = dict()
	major = ['C:maj','C#:maj','D:maj','D#:maj','E:maj','F:maj','F#:maj','G:maj','G#:maj','A:maj','A#:maj','B:maj']
	minor = ['C:min','C#:min','D:min','D#:min','E:min','F:min','F#:min','G:min','G#:min','A:min','A#:min','B:min']

	for chord in range(12):
		template[major[chord]] = list()
		template[minor[chord]] = list()

		for note in range(12):
			template[major[chord]].append(0)
			template[minor[chord]].append(0)

	for chordRoot in range(12):
		for note in range(12):
			if note == 0:
				template[major[chordRoot]][(note + chordRoot) % 12] = 1
				template[minor[chordRoot]][(note + chordRoot) % 12] = 1
			elif note == 7:
				template[major[chordRoot]][(note + chordRoot) % 12] = 1
				template[minor[chordRoot]][(note + chordRoot) % 12] = 1			
			elif note == 4:
				template[major[chordRoot]][(note + chordRoot) % 12] = 1
			elif note == 3:
				template[minor[chordRoot]][(note + chordRoot) % 12] = 1

	with open(os.path.join(templatePath,"triadsBinary.json"),"w") as fp:
		json.dump(template, fp, sort_keys=False)
		#print(">Saved triad templates successfully to JSON file!")

	return template

def createSevenths(templatePath=TEMPLATE_PATH):
	"""
	Description:
		This function creates binary templates for sevenths
		The 12 dimensional binary templates are indexed from C. 

		Example C:maj7 = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]

		This writes the templates into a json file 'sevenths.json' in TEMPLATE_PATH

		(Experimental: Choosing template value at root = 1, and other values = 0.9, to give preference to root in matching chordLabels)


	"""
	template = dict()
	template = createTriads(templatePath)

	majSeventh = ['C:maj7','C#:maj7','D:maj7','D#:maj7','E:maj7','F:maj7','F#:maj7','G:maj7','G#:maj7','A:maj7','A#:maj7','B:maj7']
	minSeventh = ['C:min7','C#:min7','D:min7','D#:min7','E:min7','F:min7','F#:min7','G:min7','G#:min7','A:min7','A#:min7','B:min7']
	domSeventh = ['C:7','C#:7','D:7','D#:7','E:7','F:7','F#:7','G:7','G#:7','A:7','A#:7','B:7']

	for chord in range(12):
		template[majSeventh[chord]] = list()
		template[minSeventh[chord]] = list()
		template[domSeventh[chord]] = list()

		for note in range(12):
			template[majSeventh[chord]].append(0)
			template[minSeventh[chord]].append(0)
			template[domSeventh[chord]].append(0)

	for chordRoot in range(12):
		for note in range(12):
			if note == 0:
				template[majSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[minSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[domSeventh[chordRoot]][(note + chordRoot) % 12] = 1

			elif note == 7:
				template[majSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[minSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[domSeventh[chordRoot]][(note + chordRoot) % 12] = 1

			elif note == 4:
				template[majSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[domSeventh[chordRoot]][(note + chordRoot) % 12] = 1
                
			elif note == 3:
				template[minSeventh[chordRoot]][(note + chordRoot) % 12] = 1

			elif note == 11:
				template[majSeventh[chordRoot]][(note + chordRoot) % 12] = 1

			elif note == 10:
				template[domSeventh[chordRoot]][(note + chordRoot) % 12] = 1
				template[minSeventh[chordRoot]][(note + chordRoot) % 12] = 1

	with open(os.path.join(templatePath,"seventhsBinary.json"),"w") as fp:
		json.dump(template, fp, sort_keys=False)
		#print(">Saved seventh templates successfully to JSON file!")

	return template

def createAllCombinations(templatePath = TEMPLATE_PATH):
	"""
	Experimental: Trying out all combinations of notes for creating exhaustive list of templates to match against

	"""

	notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
	allCombinations = dict()

	# two note templates
	combinationOf2 = findSubsets(notes,2)

	for combinationSet in combinationOf2:
		template = dict()
		for note in notes:
			template[note] = 0

		for c in combinationSet:
			template[c] = 1

		t = [0]*12
		for note in range(12):
			if template[notes[note]] == 1:
				t[note] = 1

		allCombinations[str(combinationSet)] = t


	# Three note templates
	combinationOf3 = findSubsets(notes,3)

	for combinationSet in combinationOf3:
		template = dict()
		for note in notes:
			template[note] = 0

		for c in combinationSet:
			template[c] = 1

		t = [0]*12
		for note in range(12):
			if template[notes[note]] == 1:
				t[note] = 1

		allCombinations[str(combinationSet)] = t

	with open(os.path.join(templatePath,"allCombinations.json"),"w") as fp:
		json.dump(allCombinations, fp, sort_keys=False)



def load(filename=os.path.join(TEMPLATE_PATH,'sevenths.json')):
	"""
	Description: 
		This function loads a template from the json file in TEMPLATE_PATH or from the given argument filename

	"""

	with open(filename) as f_in:
		try:
			return(json.load(f_in))
		except:
			print('ERROR: Template File Not found')


def findSubsets(S,m):
	"""
	Helper function: (uses import itertools)

		This finds all combinations of m from a set of S
		>>findSubsets(['a','b','c'], 2)
		>>{('a', 'b'), ('a', 'c'), ('b', 'c')}
	"""
	return set(itertools.combinations(S,m))