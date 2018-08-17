# -*- coding: utf-8 -*-
from os.path import isfile, join, exists
import numpy as np
from scipy import misc
from scipy import ndimage
from os import listdir, makedirs
import matplotlib.pyplot as plt

baseDir                 = r"D:\Arnaud\data_croutinet\ottawa\data"
model_dir               = join(baseDir, "models")
super_dir               = join(baseDir, "superResolution")
train_dir               = join(super_dir, "train")
preprocessed_dir        = join(super_dir, "preprocessed_gaussian_blur_sigma1")
preprocessed_train      = join(preprocessed_dir, "train")
preprocessed_validation = join(preprocessed_dir, "validation")

scale = 3.0
input_size = 33
label_size = 21
pad = int((input_size - label_size) / 2)
stride = 14

def setupDir(dir):
    if not exists(dir):
        makedirs(dir)
    if not exists(join(dir, "input")):
        makedirs(join(dir, "input"))
    if not exists(join(dir, "label")):
        makedirs(join(dir, "label"))

setupDir(preprocessed_train)
setupDir(preprocessed_validation)

count = 1
for f in listdir(train_dir):
    f = join(train_dir, f)
    if not isfile(f):
        continue

    image = misc.imread(f, mode='RGB')

    w, h, c = image.shape
    w -= w % 3
    h -= h % 3
    image = image[0:w, 0:h]

    blurred_r = ndimage.gaussian_filter(image[:,:,0],sigma=1)
    blurred_g = ndimage.gaussian_filter(image[:,:,1], sigma=1)
    blurred_b = ndimage.gaussian_filter(image[:,:,2], sigma=1)

    blurred = np.zeros(image.shape)

    for i in range(blurred.shape[0]):
        for j in range(blurred.shape[1]):
            blurred[i, j ,0] = blurred_r[i, j]
            blurred[i, j, 1] = blurred_g[i, j]
            blurred[i, j, 2] = blurred_b[i, j]

    scaled = misc.imresize(blurred, 1.0/scale, 'bicubic')
    scaled = misc.imresize(scaled, scale/1.0, 'bicubic')

    # scaled_without_blur = misc.imresize(image, 1.0/scale, 'bicubic')
    # scaled_without_blur = misc.imresize(scaled_without_blur, scale/1.0, 'bicubic')
    #
    # scaled_without_blur_nearest = misc.imresize(image, 1.0 / scale, 'nearest')
    # scaled_without_blur_nearest = misc.imresize(scaled_without_blur_nearest, scale / 1.0, 'nearest')
    #
    # plt.figure()
    # plt.title("original")
    # plt.imshow(image)
    # plt.figure()
    # plt.title("without blur")
    # plt.imshow(scaled_without_blur)
    # plt.figure()
    # plt.title("blurred")
    # plt.imshow(scaled)
    # plt.figure()
    # plt.title("nearest")
    # plt.imshow(scaled_without_blur_nearest)
    # break


    for i in range(0, h - input_size + 1, stride):
        for j in range(0, w - input_size + 1, stride):
            sub_img = scaled[j : j + input_size, i : i + input_size]
            sub_img_label = image[j + pad : j + pad + label_size, i + pad : i + pad + label_size]

            if count%7 == 0 :
                misc.imsave(join(preprocessed_validation, "input", str(count) + '.bmp'), sub_img)
                misc.imsave(join(preprocessed_validation, "label", str(count) + '.bmp'), sub_img_label)

            else:
                misc.imsave(join(preprocessed_train, "input", str(count) + '.bmp'), sub_img)
                misc.imsave(join(preprocessed_train, "label", str(count) + '.bmp'), sub_img_label)

            count += 1