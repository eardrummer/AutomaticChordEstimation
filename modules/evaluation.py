import os
import numpy as np
import mir_eval

def CSR(estFile, refFile, debug=False):
	"""
	Description:
		This function takes estimated filename.lab and reference annotations filename.lab and reports CSR values using mir_eval

	"""
	(ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(refFile)
	(est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(estFile)

	est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels,
	                                           ref_intervals.min(), ref_intervals.max(),
	                                           mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
	(intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
	durations = mir_eval.util.intervals_to_durations(intervals)

	comparisons = mir_eval.chord.root(ref_labels, est_labels)
	s1 = mir_eval.chord.weighted_accuracy(comparisons, durations)

	comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
	s2 = mir_eval.chord.weighted_accuracy(comparisons, durations)

	comparisons = mir_eval.chord.mirex(ref_labels, est_labels)
	s3 = mir_eval.chord.weighted_accuracy(comparisons, durations)

	if debug:
		print(f"Evaluation for {estFile}: \nCSR Root: {s1}\nCSR MajMin: {s2}\nCSR mirex: {s3}")

	return [s1, s2, s3]

def CSRMeans(estFolder, refFolder, genre=""):
	"""
	Description:
		This function reports the mean CSR values for all the files in the estimated folder (estFolder) compared against the ground truth (refFolder)

	"""

	csr = []

	for subdir, dirs, files in os.walk(estFolder):
		for filename in files:

			# Extracting genre for IDMT-SMT Guitar style
			#f = filename.split("_")[1]

			# Extracting genre for GuitarSet style filename
			file_genre = filename.split("_")[1].split("-")[0][:-1]

			if filename.endswith('.lab') and (file_genre == genre or genre == ""):

				print(f"Evaluating for {filename}:\n")

				estFilePath = os.path.join(estFolder, filename)
				refFilePath = os.path.join(refFolder, filename)

				csr.append(CSR(estFilePath, refFilePath, debug=True))

	csrMeans = np.mean(np.asarray(csr), axis=0)
	print("OverAllEvaluation Results: \n")
	print(f"mean CSR Root {genre} : {csrMeans[0]} \nmean CSR MajMin {genre} : {csrMeans[1]}\nmean CSR mirex {genre} : {csrMeans[2]}")