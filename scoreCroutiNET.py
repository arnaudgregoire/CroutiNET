from keras import Input, Model, Sequential
from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dropout, Flatten, Dense, Subtract, Activation
from keras.optimizers import SGD
import numpy as np
import scipy.misc as misc
import os
import csv
import matplotlib.pyplot as plt

IMG_SIZE = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)
IMG_DIR = "your img dir"

trainCsv_dir = "your training csv dir"
validationCsv_dir ="your validation csv dir"


def show(listHistory, smooth):
    """
    show in matplotlib resuts the results of training models with data augmentation
    :param listHistory:
    :param smooth: boolean to knwo if the user want to have his points smoothed or not
    :return:
    """
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for i in range(len(listHistory)):
        acc.extend(listHistory[i].history['acc'])
        val_acc.extend(listHistory[i].history['val_acc'])
        loss.extend(listHistory[i].history['loss'])
        val_loss.extend(listHistory[i].history['val_loss'])

    epochs = range(1, len(acc) + 1)
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    if smooth:
        plt.plot(epochs, smooth_curve(val_acc), 'r', label='Validation acc')
    else:
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training accuracy and Validation accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if smooth:
        plt.plot(epochs, smooth_curve(val_loss), 'r', label='Validation loss')
    else:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Validation loss and Training loss')
    plt.legend()
    plt.show()

def smooth_curve(points, factor=0.8):
    """
    Smoothing the curve of a plot
    :param points: the points to smooth
    :param factor: the factor of smoothing
    :return: the smoothing points
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def loadImageFix(name):
    """
    from an name, load the coresponding array of pixels
    :param name: name of the pictures
    :return: array of pixels, constiuting the picture
    """
    img  = misc.imread(os.path.join(IMG_DIR, name))
    img  = misc.imresize(img, (IMG_SIZE, IMG_SIZE))
    return img

def loadAsScalars(path):
    """
    load a
    :param path: the csv file were all your duels are saved
    Mine has the format :
    name of left image, name of right image, label
    :return: a tuple containing arrays of pictures of left, right images, their name, and labels
    """
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

def train():
    """
    Here you must modify the part below just to load your own data
    trainLeft is the array of left pictures in duels, shape (yourNumberOfTrainingData,IMG_SIZE, IMG_SIZE, 3)
    trainRight is the array of right pictures in duels, shape (yourNumberOfTrainingData,IMG_SIZE, IMG_SIZE, 3)
    trainLabels is the arrray of labels in duels, 0 or 1, shape (yourNumberOfTrainingData,1)
    :return:
    """
    # load training and validation set with labels as scalars between 0 and 1
    trainLeft, trainRight, trainLabels, trainNamesLeft, trainNamesRight = loadAsScalars(trainCsv_dir)
    validationLeft, validationRight, validationLabels, validationNamesLeft, validationNamesRight= loadAsScalars(validationCsv_dir)

    # Here is the architecture of ScoreCroutinet that we create below
    base_network = create_base_network(INPUT_DIM)
    model = create_meta_network(INPUT_DIM, base_network)

    # We fit the model to the training set
    history = model.fit(
        [trainLeft, trainRight],
        trainLabels,
        batch_size=16,
        epochs=30,
        validation_data=([validationLeft, validationRight], validationLabels))

    show([history], False)

