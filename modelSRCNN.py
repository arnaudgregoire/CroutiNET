from keras import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
import numpy as np
import keras.backend as K

def psnr(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def build_model(INPUT_SHAPE):
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(64, (9, 9), activation='relu', input_shape=INPUT_SHAPE, padding="valid"))
    SRCNN.add(Conv2D(32, (1, 1), activation='relu', padding="same"))
    SRCNN.add(Conv2D(1, (5, 5), activation='linear', padding="valid"))
    adam = Adam(lr=0.0001)
    SRCNN.compile(optimizer=adam, loss='mse', metrics=[psnr])
    return SRCNN