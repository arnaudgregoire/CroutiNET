from keras import Input, Model, Sequential
from keras.applications import VGG19
from keras.layers import Dropout, Flatten, Dense, Subtract, Activation
import keras.backend
from keras.optimizers import SGD

IMG_SIZE = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

def create_base_network(input_dim):
    """
    The main part of the network, the one who give scores to each pictures
    :param input_dim: Dimension of input pictures during training
    :return: keras object model
    """
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=input_dim)
    for layer in feature_extractor.layers[:18]:
        layer.trainable = False

    m = Sequential()
    m.add(feature_extractor)
    m.add(Flatten())
    m.add(Dense(4096, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(4096, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1, name="block_converge_5k"))
    return m

def create_meta_network(input_dim, base_network):
    """
    Secondn network that teach the first how to rank pictures
    :param input_dim: dimensions of pictures
    :param base_network: keras model object
    :return: keras object model
    """
    input_left = Input(shape=input_dim)
    input_right = Input(shape=input_dim)

    left_score = base_network(input_left)
    right_score = base_network(input_right)

    # subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid")(diff)
    model = Model(inputs = [input_left, input_right], outputs = prob)
    sgd = SGD(lr=1e-5, decay=1e-4, momentum=0.8, nesterov=True)
    model.compile(optimizer = sgd, loss = "binary_crossentropy", metrics=['accuracy'])

    return model

#base_network = create_base_network(INPUT_DIM)
#model = create_meta_network(INPUT_DIM, base_network)