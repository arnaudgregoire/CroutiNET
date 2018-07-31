import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import scipy.misc as misc

K.set_learning_phase(False)

# Define directories
baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
save_dir      = os.path.join(baseDir, "testgan")
model_dir     = os.path.join(baseDir, "models")
model = load_model(os.path.join(model_dir, "scoreNetworkNoSigmoid.h5"))

# Define picture size
IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)



def truncate(number, digits) -> float:
    """
    truncate a float number to another float number wth the correct number of digits
    :param number: the float number
    :param digits: the number of digits that this number shoud have
    :return: the trucated number
    """
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper

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


def computeScore(folder):
    imagesNames = [f for f in os.listdir(os.path.join(save_dir, folder))]
    for i in range(len(imagesNames)-1):
        if imagesNames[i] == "top"or imagesNames[i] == "bottom" or imagesNames[i] == "heatmaps" or imagesNames[i] == "resized":
            imagesNames.pop(i)
    pictures = []
    # Here we load in a big array all pictures as arrays (a & b are just to print the % of loading)
    i = 0
    a = 0
    b = 0
    print("loading pictures")
    for name in imagesNames:

        a = np.floor(i * 100 / len(imagesNames))
        if a != b:
            print(str(int(a)) + "%")
            b = a
        pictures.append(
            image.img_to_array(image.load_img(os.path.join(os.path.join(save_dir, folder),name), target_size=(IMG_SIZE, IMG_SIZE))))
        i += 1

    print("pictures as array")
    pictures = np.array(pictures)

    print("preprocess pictures")
    pictures = preprocess_input(x=np.expand_dims(pictures.astype(float), axis=0))[0]

    print("pictures values as float32")
    pictures = pictures.astype('float32')

    # We predict the score of each pictures using ScoreCroutinet
    prediction = model.predict(pictures)

    # We save those score in a csv
    with open(os.path.join(save_dir, folder + "Score.csv"), 'w') as csvfileWriter:
        for k in range(len(imagesNames)):
            csvfileWriter.write("{},{}\n".format(imagesNames[k], str(truncate(prediction[k, 0], 4))))

    df = pd.DataFrame.from_items([('idx', range(len(imagesNames))), ('name', imagesNames), ('score', prediction[:,0] * -1)])

    directory = os.path.join(save_dir, folder)

    scores = df["score"].tolist()

    plt.figure()
    plt.hist(scores, 50)
    plt.title(folder + " histogram (mean : " + str(np.mean(scores)) + ")")
    plt.xlabel("scores")
    plt.ylabel("fréquence d'apparition")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(directory, folder + ".png"))

    heatmap_directory = os.path.join(directory,"heatmaps")

    if not os.path.exists(heatmap_directory):
        os.mkdir(heatmap_directory)

    for i in range(len(pictures)):
        cv2.imwrite(os.path.join(heatmap_directory,"heatmap" + str(i) + ".jpg"),heatmap(pictures[i],os.path.join(directory, imagesNames[i])))

        img = misc.imread(os.path.join(heatmap_directory,"heatmap" + str(i) + ".jpg"))
        img = misc.imresize(img, (640, 640))

        real_img = misc.imread(os.path.join(directory,imagesNames[i]))
        real_img =  misc.imresize(real_img, (640, 640))

        plt.figure()
        plt.subplot(1,2,1)
        plt.title("heatmap score : " + str(scores[i]))
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(real_img)
        plt.legend()
        plt.savefig(os.path.join(heatmap_directory,"comparaisonheatmap" + str(i) + "score" + str(scores[i]) +".jpg"))


computeScore("bottom_pictures")
