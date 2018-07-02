import os
from random import randint

from keras.applications.imagenet_utils import preprocess_input
from scipy import misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
import cv2
from generator import dataGenerator as d
from loader import loadAsScalars
"""
Double Check all datas to see if something is wrong during process data augmentation
"""


#Define the img size
IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

#Define directories
baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
testDir = os.path.join(baseDir, "test/test.csv")
roads_loubna_dir = os.path.join(baseDir, "roads_loubna")
ranking_dir = os.path.join(baseDir, "rankingNoSigmoid")
activation_dir = os.path.join(ranking_dir, "activation")
check_dir = os.path.join(ranking_dir, "checkdata")
models_dir = os.path.join(baseDir, "models")

model = load_model(os.path.join(models_dir, "scoreNetworkNoSigmoid.h5"))

validationLeft, validationRight, validationLabels, namesLeft, namesRight = loadAsScalars(validationDir)

def mutliplePredict(data, nb_predict):
    """
    Compute multipes predict on same dataset
    :param data: the array of puictures
    :param nb_predict: numbre of predictions that you wannt
    :return: the mean of predictions for each value
    """
    scores_array = []

    for i in range(nb_predict):
        scores_array.append(model.predict(data))

    scores_array = np.array(scores_array)
    scores_array = scores_array * (-1)
    mean_array = np.mean(scores_array, axis=0)

    return mean_array


def compareScoreWithLabels(scoresLeft, scoresRight, labels):
    """
    Compare The score and the labels to see how many scores leaded to the good compariaosn (with labels)
    :param scoresLeft:
    :param scoresRight:
    :param labels:
    :return: integer symbolizeing how many correct response
    """
    correct = 0

    for i in range(len(scoresLeft)):
        if scoresLeft[i,0] > scoresRight[i,0]:
            if labels[i] == 0:
                correct += 1

        else:
            if labels[i] == 1:
                correct += 1

    return correct

accuracies = []

for i in range(1,10):
    print(i)
    meanValidationLeft = mutliplePredict(validationLeft, i)
    meanValidationRight = mutliplePredict(validationRight, i)
    print(meanValidationLeft[0,0])
    print(meanValidationRight[0, 0])
    accuracies.append(compareScoreWithLabels(meanValidationLeft, meanValidationRight, validationLabels)/validationLabels.shape[0])


plt.figure()
plt.plot(range(1,10),accuracies)
plt.xlabel("number of predictions for each pictures")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy en fonction du nombre de prédictions")