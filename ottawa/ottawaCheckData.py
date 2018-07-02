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

K.set_learning_phase(False)
"""
Double Check all datas to see if something is wrong during process data augmentation
"""

def normalize(distribution_array):
    """
    normalize the distributed array by giving it a mean of 0 and a standard deviation of 1
    :param distribution_array:  the array of values to normalize
    :return:  the normalized array
    """
    mean = np.mean(distribution_array)
    ecartType = np.std(distribution_array)
    normalized = distribution_array - mean
    normalized = normalized / ecartType
    normalized = normalized * (-1)
    return  normalized

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


def loadImage(name):
    img  = misc.imread(os.path.join(roads_loubna_dir, name))
    img  = misc.imresize(img, (IMG_SIZE, IMG_SIZE))
    return img


def heatmap(picture, picture_path):
    image = np.expand_dims(picture, axis=0)
    last_conv_layer = model.layers[0].get_layer('block5_conv4')
    grads = K.gradients(model.output[:, 0], last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([image])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)

    img = cv2.imread(picture_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img




validationLeft, validationRight, validationLabels, namesLeft, namesRight = loadAsScalars(validationDir)

validationLeft_score = model.predict(validationLeft)
validationLeft_score = validationLeft_score * (-1)

validationRight_score = model.predict(validationRight)
validationRight_score = validationRight_score * (-1)


prediction = []
correct = 0

for i in range(len(validationLeft_score)):
    if validationLeft_score[i] > validationRight_score[i]:
        if validationLabels[i] == 0:
            prediction.append("prédiction vraie")
            correct += 1
        else:
            prediction.append("prédiction fausse")
    else:
        if validationLabels[i] == 0:
            prediction.append("prédiction fausse")
        else:
            prediction.append("prédiction vraie")
            correct += 1

dict = {0: "left winner", 1: "right winner"}


for i in range(len(validationLabels)):
    plt.figure()

    leftImage = validationLeft[i]
    rightImage = validationRight[i]
    leftPath = os.path.join(roads_loubna_dir, namesLeft[i])
    rightPath = os.path.join(roads_loubna_dir, namesRight[i])
    heatmapLeft  = heatmap(leftImage, leftPath)
    heatmapRight = heatmap(rightImage, rightPath)
    cv2.imwrite(os.path.join(activation_dir, "validationSetLeft" + str(i) + "score" + str(validationLeft_score[i]) +  ".jpg"), heatmapLeft)
    cv2.imwrite(os.path.join(activation_dir,"validationSetRight" + str(i) + "score" + str(validationRight_score[i]) +  ".jpg"), heatmapRight)

    plt.subplot(2, 2, 1)
    plt.title("score : " + str(validationLeft_score[i]))
    plt.imshow(loadImage(leftPath))

    plt.suptitle(dict[validationLabels[i]] + " : " +  prediction[i], fontsize=16)

    plt.subplot(2, 2 , 2)
    plt.title("score : " + str(validationRight_score[i]))
    plt.imshow(loadImage(rightPath))

    plt.subplot(2, 2 , 3)
    plt.imshow(loadImage(os.path.join(activation_dir, "validationSetLeft" + str(i) + "score" + str(validationLeft_score[i]) +  ".jpg")))

    plt.subplot(2, 2 , 4)
    plt.imshow(loadImage(os.path.join(activation_dir,"validationSetRight" + str(i) + "score" + str(validationRight_score[i]) +  ".jpg")))

    plt.savefig(os.path.join(check_dir, "validationSet" + str(i) + ".jpg"))
