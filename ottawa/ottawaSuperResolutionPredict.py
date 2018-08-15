# -*- coding: utf-8 -*-
from os.path import isfile, join, exists
import numpy as np
from keras.models import load_model
from scipy import misc
from os import listdir, makedirs
import matplotlib.pyplot as plt
import keras.backend as K
import math

baseDir                 = r"D:\Arnaud\data_croutinet\ottawa\data"
model_dir               = join(baseDir, "models")
super_dir               = join(baseDir, "superResolution")
train_dir               = join(super_dir, "train")
preprocessed_dir        = join(super_dir, "preprocessed")
preprocessed_train      = join(preprocessed_dir, "train")
preprocessed_validation = join(preprocessed_dir, "validation")
result_dir              = join(super_dir,"result")
experiment_dir         = join(super_dir,"experiment")
testPicture             = join(result_dir, "generated1600.png")
upscaledPicture         = join(result_dir,"superResolutionTest.png")
interpolatedPicture     = join(result_dir,"interpolatedTest.png")

output_size  = 640
input_size = output_size +12

def load(dir):
    array = np.array([misc.imread(join(dir, f), mode='YCbCr') for f in listdir(dir)])
    return array

def psnr_normal(target, ref):
    diff = ref - target
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

def ycbcr2rgb(im):
    """
    Convert a Ycbcr array to rgb array
    :param im: the Ycbcr np array
    :return: a RGB array
    """
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def psnr(y_true, y_pred):
    """
    peak signal to noise ratio
    :param y_true:
    :param y_pred:
    :return:
    """
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

#We load the model with the custom metric peak signal to noise ratio redefined above
m = load_model(join(model_dir,"SRCNN_gaussian_blur_check.h5"), custom_objects={'psnr': psnr})

#We load pictures in directory, the loaded picture are in Ycbcr format
picture = load(experiment_dir)

#We scale by interpolation the input picture to the desired input size
bicubic_picture = misc.imresize(picture[0], (input_size,input_size), 'bicubic')

# we extract the channel Y of the picture
Y_scaled = bicubic_picture[:,:,0]
Y_channel = bicubic_picture[:,:,0]

# We preprocess the picture to make it fit the requirement of input network
Y_scaled = np.expand_dims(Y_scaled.astype(float), axis=0)
Y_scaled = np.expand_dims(Y_scaled.astype(float), axis=3)
Y_scaled = Y_scaled / 255

#We compute the new Y channel values by using the network
prediction = m.predict(Y_scaled).reshape(output_size,output_size)

#we deprocess it
prediction = prediction *255
np.putmask(prediction, prediction > 255, 255)
np.putmask(prediction, prediction < 0, 0)

# We descaled by 12 pixels the input picture to make it fit the dimension of output picutre
computed_picture = misc.imresize(picture[0], (output_size,output_size), 'bicubic')

#We replace the Y channel of the picture by the one computed by the network
computed_picture[:,:,0] = prediction

plt.figure()
plt.subplot(1,2,1)
plt.title("Y (luminance) channel input")
plt.matshow(Y_channel, fignum=False, cmap="Greys_r")
plt.colorbar()
plt.subplot(1,2,2)
plt.title("Y (luminance) channel output")
plt.matshow(prediction, fignum=False, cmap="Greys_r")
plt.colorbar()
plt.show()


plt.figure()
plt.subplot(2,2,1)
plt.title("interpolation (size : " + str(bicubic_picture.shape[0]) + "x" + str(bicubic_picture.shape[0]) + ")")
plt.imshow(ycbcr2rgb(bicubic_picture))
plt.subplot(2,2,2)
plt.title("super resolution (size : " + str(computed_picture.shape[0]) + "x" + str(computed_picture.shape[0]) + ")")
plt.imshow(ycbcr2rgb(computed_picture))
plt.subplot(2,2,3)
plt.title("original input low resolution (size : " + str(picture[0].shape[0]) + "x" + str(picture[0].shape[0]) + ")")
plt.imshow(ycbcr2rgb(picture[0]))
plt.subplot(2,2,4)
plt.title("original high resolution picture (size : 640x640)")
plt.imshow(misc.imread(join(result_dir, "highResolutionOriginal.jpg"), mode='RGB'))
plt.show()

misc.imsave(upscaledPicture, ycbcr2rgb(computed_picture))
misc.imsave(interpolatedPicture, ycbcr2rgb(bicubic_picture))

bicubic_picture = misc.imresize(bicubic_picture,(output_size,output_size))

print(psnr_normal(computed_picture, misc.imread(join(result_dir, "highResolutionOriginal.jpg"), mode='YCbCr')))
print(psnr_normal(bicubic_picture, misc.imread(join(result_dir, "highResolutionOriginal.jpg"), mode='YCbCr')))

