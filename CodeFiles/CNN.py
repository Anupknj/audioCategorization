import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
import keras
import warnings
warnings.filterwarnings('ignore')
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD
import split_folders
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# load necessary data
classSet = pd.read_csv(r'.\data\train.csv',error_bad_lines=False)
trainData = pd.read_csv('trainDatasetAll.csv')
testData = pd.read_csv('testDatasetAll.csv')
testIdx = pd.read_csv(r'.\data\test_idx.csv',error_bad_lines=False).to_dict()
genres = 'Rock Pop Folk Instrumental Electronic Hip-Hop'.split() # labels

path = r'.data\train'
folder = os.fsencode(path)

# this function generates spectogram for every file and stores in folder name which is label
def getSpectogram():
    counter = 0
    for file in os.listdir(folder):
        counter += 1
        print(counter)
        filename = os.fsdecode(file)
        songname = path+"\\"+filename
        y, sr = librosa.load(songname, mono=True, duration=5)
        print(y.shape)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB')
        plt.axis('off')
        name = str(classSet.loc[classSet['new_id']==int(filename.lstrip("0").split(".")[0]),['genre']])
        genret = name.split(" ")[-1]
        plt.savefig(f'./data/CNNData/{genret}/{filename[:-3].replace(".", "")}.png')
        plt.savefig(f'./data/CNNData/{genret}/{filename[:-3].replace(".", "")}.png')

        plt.clf()


# split the folders into train and val in the ration of 0.7 and 0.3
split_folders.ratio('./data/CNNData', output="./data/CNNDataSplit", seed=1337, ratio=(.7, .3))

#  functions for configuration images of train and test

train_datagen = ImageDataGenerator(
        rescale=1./255, # rescale all pixel values from 0-255, so aftre this step all our pixel values are in range (0,1)
        shear_range=0.2, #to apply some random tranfromations
        zoom_range=0.2, #to apply zoom
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# call the above function for train and test

training_set = train_datagen.flow_from_directory(
        './data/CNNDataSplit/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
test_set = test_datagen.flow_from_directory(
        './data/CNNDataSplit/val',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False )

testSet = test_datagen.flow_from_directory(
        './data/CNNDataTestSet',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False )


# train the moedl for CNN

model = Sequential()
input_shape=(64, 64, 3)
#1st hidden layer
model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#2nd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#3rd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))
#Add fully connected layer.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
#Output layer
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()


# set the values for model fit
epochs = 200
batch_size = 8
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit_generator(training_set,steps_per_epoch=100,epochs=100,validation_data=test_set,validation_steps=200)
accuracy = model.evaluate_generator(generator=test_set, steps=50)
print("Accuracy is :")
print(accuracy)

predictionsTestSet = model.predict(test_set)


# creates confusion matrix for the split data
def createConfusionMatrix():
    confusionMatrix = confusion_matrix(test_set,np.argmax(predictionsTestSet,axis=1))
    print(confusionMatrix)

    labels = ['rock', 'pop','folk','instrument','EDM','hip-hop']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusionMatrix)
    plt.title('Confusion matrix of the classifier : CNN')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# predictions for the testing data
predictionsTestSet = model.predict(testSet)


# below are the function to return file with ID and predictions
finalOutput = list()
def findAccuracyTestData():
    numberOfPredictions = predictionsTestSet.shape[0]
    for i in range(numberOfPredictions):
        tempList = list()
        tempList.append(testIdx["new_id"][i])
        tempList.append(np.argmax(predictionsTestSet[i]))
        finalOutput.append(tempList)
    return finalOutput


def listOfListToCSV():
    finalOutputList = findAccuracyTestData()
    final =pd.DataFrame(finalOutputList,columns=['id','genre'])
    final.to_csv('finalOutputFinal(CNN).csv', index = False)
    print("file is out ")

listOfListToCSV()





