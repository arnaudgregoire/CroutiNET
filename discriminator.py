import keras
from keras import layers



def build_discriminator(height, width, channels):

    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)#0.4 vanilla
    #x = layers.BatchNormalization()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,#0.0008 vanilla
        clipvalue=1.0,
        decay=1e-8)

    discriminator.compile(optimizer=discriminator_optimizer,
    loss='binary_crossentropy')

    return discriminator
