import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
import keras
from keras.models import Sequential
import warnings 
warnings.filterwarnings('ignore')
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from numpy import array
from sklearn.model_selection import KFold

# ----------import all the necessary data

classSet = pd.read_csv(r'.\data\train.csv',error_bad_lines=False)
testIdx = pd.read_csv(r'.\data\test_idx.csv',error_bad_lines=False).to_dict()
classSetTest = pd.read_csv(r'.\data\test_idx.csv',error_bad_lines=False)



# def fiveFoldCrossValidationAndCI():
#     crossValidation = array(pd.read_csv('trainDatasetAll.csv'))
# # prepare cross validation
#     kfold = KFold(5, True, 1)
# # enumerate splits
#     for train, test in kfold.split(crossValidation):
# 	    print('train: %s, test: %s' % (crossValidation[train], crossValidation[test]))
        

# fiveFoldCrossValidationAndCI()

# below function extracts necessary features and stores in csv file for training set

def featureExtractionTrain():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print("feature extraction")
    file = open('trainDatasetAll.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    path = r'.\data\train'
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        songname = path+"\\"+filename
        y, sr = librosa.load(songname, mono=True, duration=30)
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'

        classExtracted= str(classSet.loc[classSet['new_id']==int(filename.lstrip("0").split(".")[0]),['genre']])
        genret = classExtracted.split(" ")[-1]
        to_append += f' {genret}'

        file = open('trainDatasetAll.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

# data pre processing

data = pd.read_csv('trainDatasetAll.csv') # import features of training data set
data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the Labels
genreList = data.iloc[:, -1] # select column with classes
encoder = LabelEncoder()
y = encoder.fit_transform(genreList)#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # split into 0.7 : 0.3

# setting up model for ANN

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
classifier = model.fit(X_train,y_train,epochs=100,batch_size=128)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test,np.argmax(predictions,axis=1))
print("Accuracy for Split data is ")
print(accuracy)

# this function displays and plots confusion matrix for above prediction

def getConfusionMatrix():
    confusionMatrix = confusion_matrix(y_test,np.argmax(predictions,axis=1))
    print(confusionMatrix)

    labels = ['rock', 'pop','folk','instrument','EDM','hip-hop']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusionMatrix)
    plt.title('Confusion matrix of the classifier : ANN')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
getConfusionMatrix()

# this fucntion extracts feature for testing data and stored in csv

def featureExtractionTest():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print(header)

    print("feature extraction test")
    file = open('testDatasetAll.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    path = r'D:\University\Academics\_ML\project\P3\data\test'
    folder = os.fsencode(path)
    counter = 0
    for file in os.listdir(folder):
        counter += 1
        print(counter)
        filename = os.fsdecode(file)
        songname = path+"\\"+filename
        y, sr = librosa.load(songname, mono=True, duration=30)
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'

    file = open('testDatasetAll.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


data = pd.read_csv('testDatasetAll.csv')
data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the Labels
scaler = StandardScaler()
X_testExt = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
predictionsFinal = model.predict(X_testExt) # prediction is made for testin data

# Below function stores the ID and prediction in csv file

finalOutput = list()
def findAccuracyTestData():
    numberOfPredictions = predictionsFinal.shape[0]
   
    for i in range(numberOfPredictions):
        tempList = list()
        tempList.append(testIdx["new_id"][i])
        tempList.append(np.argmax(predictionsFinal[i]))
        finalOutput.append(tempList)
    return finalOutput

def listOfListToCSV():
    finalOutputList = findAccuracyTestData()
    final =pd.DataFrame(finalOutputList,columns=['id','genre'])
    final.to_csv('finalOutputFinal(ANN).csv', index = False)
    print("file is out ")

listOfListToCSV()


