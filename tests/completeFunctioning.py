from modules import chords_standard as CS 

dataPath = 'D:/Home/Independent_Research/AutomaticChordEstimation/ACRProject/resources/allData'
annotationsPath = 'D:/Home/Independent_Research/AutomaticChordEstimation/ACRProject/resources/annotations/'
outputPath = 'D:/Home/Independent_Research/AutomaticChordEstimation/ACRProject/output/'

CS.ACR(dataPath, annotationsPath, outputPath)