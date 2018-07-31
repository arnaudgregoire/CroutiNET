import os
from keras.preprocessing import image
import  numpy as np
import scipy.misc as sc

# Define directories
baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
save_dir      = os.path.join(baseDir, "testgan")


def upscale(folder, size):
    dir = os.path.join(save_dir, folder)
    imagesNames = [f for f in os.listdir(dir)]
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
            image.img_to_array(image.load_img(os.path.join(dir,name), target_size=(size, size))))
        i += 1

    print("pictures as array")
    pictures = np.array(pictures)

    os.mkdir(os.path.join(dir,"resized"))
    resized_dir = os.path.join(dir,"resized")

    for i in range(len(pictures)):
        sc.imsave(os.path.join(resized_dir, 'resized' + str(i) + '.jpg'), pictures[i])

upscale("bottom_pictures", 800)