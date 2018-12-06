# AutomaticChordEstimation
A Project folder for performing Automatic Chord Estimation. This is not a maintained package. Only contains experiments to perform chord estimation on monotimbral polyphonic audio

In this project we compile and implement 4 types of Automatic Chord Estimation methods available in different open source libraries.

* Essentia and Librosa (basic template matching from chromagram obtained from Constant-Q)  [[Ref]](http://essentia.upf.edu/documentation/reference/std_ChordsDetection.html)  

* Chordino – VAMP plugin (NNLS chroma obtained from template matching with 88 templates for each tone followed by a dynamic bayesian network Language model to predict chord names) [[Ref]](http://www.isophonics.net/nnls-chroma)  

* Madmom – ( a simpler implementation of the 3 layer DNN to obtain chromagrm-like vector from spectrogram followed by a conditional random field language model to predict chord names) [[Ref]](https://madmom.readthedocs.io/en/latest/modules/features/chords.html)

Tutorial Notebooks (Shall be updated soon) provide an easy look at how to use these modules for performing Chord Recognition on your own local files.
