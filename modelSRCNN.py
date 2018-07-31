from keras import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

def build_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', use_bias=True, input_shape=(64, 64, 1),  padding="valid"))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', use_bias=True,  padding="same"))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(filters=1,kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', use_bias=True,  padding="valid"))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return SRCNN