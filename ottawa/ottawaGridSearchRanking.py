import os

from hyperas import optim
from hyperopt import STATUS_OK, Trials, tpe
from keras import Sequential
from keras.applications import VGG19
from keras.layers import Flatten, Dense, Dropout
from hyperas.distributions import choice, uniform
from loader import loadAsScalars
from modelRanking import create_meta_network

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
model_save = os.path.join(baseDir, "hyperasScoreModel.h5")


def data():
    baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
    trainDir = os.path.join(baseDir, "train/train.csv")
    validationDir = os.path.join(baseDir, "validation/validation.csv")
    trainLeft, trainRight, trainLabels = loadAsScalars(trainDir)
    validationLeft, validationRight, validationLabels = loadAsScalars(validationDir)

    X_train = [trainLeft, trainRight]
    y_train = trainLabels
    X_test = [validationLeft, validationRight]
    y_test = validationLabels

    return X_train, X_test, y_train, y_test

def model(X_train, X_test, y_train, y_test):
    IMG_SIZE = 224
    INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=INPUT_DIM)
    for layer in feature_extractor.layers[:18]:
        layer.trainable = False

    base_network = Sequential()
    base_network.add(feature_extractor)
    base_network.add(Flatten())
    base_network.add(Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096])}}, activation='relu', name="block_converge_2"))
    base_network.add(Dropout({{uniform(0, 0.5)}}))
    base_network.add(Dense({{choice([64, 128, 256, 512, 1024, 2048, 4096])}}, activation='relu', name="block_converge_3"))
    base_network.add(Dropout({{uniform(0, 0.5)}}))
    base_network.add(Dense(1, name="block_converge_5k"))

    model = create_meta_network(INPUT_DIM, base_network)

    model.fit(
        [X_train[0], X_train[1]],
        y_train,
        batch_size=16,
        epochs=20,
        validation_data=( [X_test[0], X_test[1]], y_test))

    score, acc = model.evaluate([X_test[0], X_test[1]], y_test, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': base_network}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print(best_run)
    best_model.save(model_save)