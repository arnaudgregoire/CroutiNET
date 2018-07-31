import scipy
from scipy import misc
import numpy as np
from PIL import Image
import os
import csv
from keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing import image as keras_image
from scipy import misc

#baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"
baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")

#correctImgDir = os.path.join(baseDir,"correctImg")

correctImgDir = os.path.join(baseDir,"roads_loubna")

IMG_SIZE = 224

def loadImage(name):
    img =  keras_image.load_img(os.path.join(correctImgDir, name), target_size=(IMG_SIZE, IMG_SIZE))
    x = keras_image.img_to_array(img)
    return  x

def loadImageFix(name):
    img  = misc.imread(os.path.join(correctImgDir, name))
    img  = misc.imresize(img, (IMG_SIZE, IMG_SIZE))
    return img

def loadAsScalars(path):
    leftImages = []
    rightImages = []
    labels = []
    namesLeft = []
    namesRight = []
    with open(path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        for line in reader:
            if line != [] and line[2] != '0.5':
                leftImages.append(loadImageFix(line[0]))
                rightImages.append(loadImageFix(line[1]))
                labels.append(int(line[2]))
                namesLeft.append(line[0])
                namesRight.append(line[1])

    leftImages = np.array(leftImages)
    rightImages = np.array(rightImages)

    labels = np.array(labels)

    leftImages = preprocess_input(x=np.expand_dims(leftImages.astype(float), axis=0))[0]
    rightImages = preprocess_input(x=np.expand_dims(rightImages.astype(float), axis=0))[0]

    leftImages = leftImages.astype('float32')# / 255
    rightImages = rightImages.astype('float32')# / 255

    return (leftImages, rightImages, labels, namesLeft, namesRight)


def load(path):
    leftImages = []
    rightImages = []
    labels = []
    namesLeft = []
    namesRight = []
    with open(path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        for line in reader:
            if line != [] and line[2] != '0.5':
                leftImages.append(loadImage(line[0]))
                rightImages.append(loadImage(line[1]))
                labels.append(int(line[2]))
                namesLeft.append(line[0])
                namesRight.append(line[1])

    leftImages = np.array(leftImages)
    rightImages = np.array(rightImages)
    labels = np.array(labels)


    leftImages = preprocess_input(x=np.expand_dims(leftImages.astype(float), axis=0))[0]
    rightImages = preprocess_input(x=np.expand_dims(rightImages.astype(float), axis=0))[0]

    leftImages = leftImages.astype('float32')# / 255
    rightImages = rightImages.astype('float32')# / 255

    #print('before')
    #print(labels.shape)
    labels = to_categorical(labels)
    #print('after')
    #print(labels.shape)

    return (leftImages, rightImages, labels, namesLeft, namesRight)
