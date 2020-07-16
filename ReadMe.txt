
Instructions :

1. This program was programmed on python version 3.7.4

2. Install necessary libraries : numpy, librosa, os, pandas, matplotlib, PIL, pathlib, csv, keras, warnings, pickle, sklearn
IPython, random, split_folders

3. Programs have to be run seperately as below:

	a. musicClassifier.py - Program for 3 different types of data visualization
	b. ANN.py - program which uses nueral network classifier
	c. CNN.py - program which uses convolutional nueral network classifier
	d. SVM.py - program which uses support vector machine classifier

4. Please must include - train audio files in codefiles/data/train   and test audio files in codefiles/data/test

5. Individual program with already extracted data should not take more than an hour.

6. There finalOutput.csv files which provide the predictions obtained under various classifier.

7. ConfusionMatrix in code files stores all the confusion matrix.

8. trainDataSetAll and testDataSetAll has all the feature extracted from train and test data.

9. In data folder, trainViz and testViz provide different form of data representation.

10. All the data(images) are removed to make size smaller. In data/trainViz and data/testViz, contains 3 different forms only for a single mp3.If needed for all mp3, then musicClassifier.py has to be run keeping appropriate paths for train and test data.

11. The file in data/CNNData has data in the form of spectrogram arranged genre wise for training. The data will be split in CNNDataSplit and CNNDataTestSet has images for finla testing.







