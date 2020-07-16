import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

# loading data

trainData = pd.read_csv('trainDatasetAll.csv')
testData = pd.read_csv('testDatasetAll.csv')
testIdx = pd.read_csv(r'.\data\test_idx.csv',error_bad_lines=False).to_dict()

#-----data split

dfTrain = trainData.iloc[:1800, :]  # .75
dfTest = trainData.iloc[1800:, :]  #.25

#convert features into matrix for split data
featuresSplitTrain = dfTrain[['chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'	
]].as_matrix()
featuresSplitTest = dfTest[['chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'	
]].as_matrix()
genreSplitTrain = dfTrain['label'].tolist()
genreSplitTest  = dfTest['label'].tolist()

#convert features into matrix for  train data

features = trainData[['chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'	
]].as_matrix()
featuresTest = testData[['chroma_stft','rmse','spectral_centroid','spectral_bandwidth','rolloff','zero_crossing_rate','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'	
]].as_matrix()
genre = trainData['label'].tolist()
featuresHeads = trainData.columns.values[1:-1].tolist()

# model training for SVM for split data

model = svm.SVC(kernel='linear')
model.fit(featuresSplitTrain, genreSplitTrain)
predictionsTest = model.predict(featuresSplitTest)
accuracy = accuracy_score(genreSplitTest,predictionsTest)  # accuracy for the split data
print("Accuracy for split data is ")
print(accuracy)

#this funtion creates confusion matrix

def createConfusionMatrix():
    confusionMatrix = confusion_matrix(genreSplitTest,predictionsTest)
    print(confusionMatrix)
    labels = ['rock', 'pop','folk','instrument','EDM','hip-hop']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusionMatrix)
    plt.title('Confusion matrix of the classifier : SVM')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
createConfusionMatrix()

# model training for SVM for  data

model = svm.SVC(kernel='linear')
model.fit(features, genre)
predictionsFinal = model.predict(featuresTest)

# below functions helps to produce output file which has ID and its predictions

finalOutput = list()
def findAccuracyTestData():
    numberOfPredictions = predictionsFinal.shape[0]
   
    for i in range(numberOfPredictions):
        tempList = list()
        tempList.append(testIdx["new_id"][i])
        tempList.append(predictionsFinal[i])
        finalOutput.append(tempList)
    return finalOutput


def listOfListToCSV():
    finalOutputList = findAccuracyTestData()
    final =pd.DataFrame(finalOutputList,columns=['id','genre'])
    final.to_csv('finalOutputFinal(SVM).csv', index = False)
    print("file is out ")

listOfListToCSV()
