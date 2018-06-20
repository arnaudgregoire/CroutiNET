from random import randint
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from model3 import converge_model, con_model
from loader import load
from generator import dataGenerator as d
import numpy as np
import pandas as pd
import os


"""
Hyper parameters optimisation using Hyperopt and Hyperas for ComparaisonCroutinet with Dense layers on top insted of Convolutionnals layers

Compute hyper parameters optimisation on given model and save it

"""

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
model_save = os.path.join(baseDir, "modelWithDataAugmentationDense.h5")

def data():
    """
    Here we load the training set and the validation set with their labels without any data augmentation
    In fact, we don't want data augmentation to influence the optimisation of hyperparameters
    :return: training set, validation set, training labels, validation labels
    """
    baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
    trainDir = os.path.join(baseDir, "train/train.csv")
    validationDir = os.path.join(baseDir, "validation/validation.csv")
    trainLeft, trainRight, trainLabels = load(trainDir)
    validationLeft, validationRight, validationLabels = load(validationDir)

    X_train = [trainLeft, trainRight]
    y_train = trainLabels
    X_test = [validationLeft, validationRight]
    y_test = validationLabels

    return X_train, X_test, y_train, y_test


def model(X_train, X_test, y_train, y_test):
    """
    Here is the function that hyperas will call to optimize our hyperparameter, all variables in double brackets "{{"
    will be optmize by hyperas
    :param X_train: the training set
    :param X_test: the validation set
    :param y_train: the training labels
    :param y_test: the validation labels
    :return: the accuracy of the model after the fit, a boolean to know if everithing works fine, the model that we want to save
     (here ComparaisonCroutinet with Dense layer on top instead of convolutionnal layers)
    """
    model = Sequential()
    model.add(con_model())
    model.add(Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096, 8192])}}, activation='relu', name="block_converge_2"))  # ,input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Dropout({{uniform(0,0.5)}}))
    model.add(Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096, 8192])}}, activation='relu', name="block_converge_3"))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096, 8192])}}, activation='relu', name="block_converge_4"))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='softmax', name="block_converge_5k"))

    sgd = SGD(lr=1e-4, decay={{choice([1e-4, 1e-5, 1e-6])}}, momentum={{uniform(0, 0.9)}}, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(
        [X_train[0], X_train[1]],
        y_train,
        batch_size=16,
        epochs=30,
        validation_data=( [X_test[0], X_test[1]], y_test))

    score, acc = model.evaluate([X_test[0], X_test[1]], y_test, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    print(best_run)
    best_model.save(model_save)
