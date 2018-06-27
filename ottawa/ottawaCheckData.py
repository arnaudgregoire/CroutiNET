import os
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

from generator import dataGenerator as d
from loader import loadAsScalars


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
check_dir = os.path.join(ranking_dir, "checkdata")
models_dir = os.path.join(baseDir, "models")

model = load_model(os.path.join(models_dir, "scoreNetworkNoSigmoid.h5"))
#
# all_results = np.loadtxt(trainDir, str, delimiter=',')
#
# duelsDF = pd.DataFrame(all_results, None, ['left_id', 'right_id', 'winner'])
# duelsDF['left_id'] = roads_loubna_dir + "/" + duelsDF['left_id']
# duelsDF['right_id'] = roads_loubna_dir + "/" + duelsDF['right_id']
# #print(duelsDF)
#
# mask_yes = duelsDF['winner'] == '1'
# yes = duelsDF[mask_yes]
#
# mask_no = duelsDF['winner'] == '0'
# no = duelsDF[mask_no]
#
validationLeft, validationRight, validationLabels = loadAsScalars(validationDir)
#
#
# # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
# # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
# # sizes.
# yesno = yes.sample(20).append(no.sample(20))
# print('yesno created')
# labels = dict(zip([str(x) for x in yesno.index.tolist()],
#                   [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
# print('labels created')
#
# partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}
# print('partition created')
#
# batchSizeAug = len(yesno.index.tolist())
# print('batcgSizeAug created')
# # Set-up variables for augmentation of current batch of yesno in partition
# params = {
#     'dim_x': IMG_SIZE,
#     'dim_y': IMG_SIZE,
#     'dim_z': 3,
#     'batch_size': batchSizeAug,
#     'shuffle': True
# }
# print('params created')
#
# datagenargs = {
#     'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
#     'shear_range': 0.1,
#     'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
# }
# print('datagenargs created')
#
# training_generator = d.myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
#                                                              datagenargs=datagenargs)
# print('training generator created')
#
# X, y = training_generator.__next__()
# print('X,y created')
# # zero center images
# X = np.array(X)
#
# X_left_score = normalize(model.predict(X[0]))
# X_right_score = normalize(model.predict(X[1]))
validationLeft_score = normalize(model.predict(validationLeft))
validationRight_score = normalize(model.predict(validationRight))
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

# for i in range(0, X.shape[1]):
#     plt.figure()
#
#     plt.subplot(1, 2, 1)
#     plt.title("score : " + str(X_left_score[i]))
#     leftImage = X[0, i]
#     plt.imshow(leftImage)
#
#     plt.suptitle(dict[y[i]] + '(data generated set)', fontsize=16)
#
#     plt.subplot(1, 2 , 2)
#     plt.title("score : " + str(X_right_score[i]))
#     rightImage = X[1, i]
#     plt.imshow(rightImage)
#
#     plt.savefig(os.path.join(check_dir,"datageneratedTrainSet" + str(i) +".jpg"))

for i in range(len(validationLeft_score)):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title("score : " + str(validationLeft_score[i]))
    leftImage = validationLeft[i]
    plt.imshow(leftImage)

    plt.suptitle(dict[validationLabels[i]] + " : " +  prediction[i], fontsize=16)

    plt.subplot(1, 2 , 2)
    plt.title("score : " + str(validationRight_score[i]))
    rightImage = validationRight[i]
    plt.imshow(rightImage)

    plt.savefig(os.path.join(check_dir, "validationSet" + str(i) + ".jpg"))
