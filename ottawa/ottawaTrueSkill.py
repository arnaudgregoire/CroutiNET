import os
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from croutiPoint import CroutiPoint
from keras.preprocessing import image
import numpy as np
import random as rd
import trueskill

'''
File used to compute scores with ComparaisonCroutinet and Trueskill
'''

#Define the img size
IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

#Define directories
baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir      = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
roadsDir      = os.path.join(baseDir, "roads")
model_dir     = os.path.join(baseDir, "models")
network_save  = os.path.join(model_dir, "modelWithDataAugmentation5.h5")
samples_dir   = os.path.join(baseDir, "samples_roads")

#Here is the architecture of ComparaisonCroutinet
model = load_model(network_save)

#We load all roads pictues names of ottawa
imagesNames = [f for f in os.listdir(roadsDir)]
pictures    = []
croutipoints = []

# Here we load in a big array all pictures as arrays (a & b are just to print the % of loading)
i=0
a=0
b=0
print("loading pictures")
for name in imagesNames:
    a = np.floor(i*100/len(imagesNames))
    if a != b :
        print(str(int(a)) + "%")
        b = a
    pictures.append(image.img_to_array(image.load_img(os.path.join(roadsDir, name), target_size=(IMG_SIZE, IMG_SIZE))))
    i+=1

print("pictures as array")
pictures = np.array(pictures)

print("preprocess pictures")
pictures = preprocess_input(x=np.expand_dims(pictures.astype(float), axis=0))[0]

print("pictures values as float32")
pictures = pictures.astype('float32')

croutipoints = [CroutiPoint(imagesNames[i], pictures[i]) for i in range(len(imagesNames))]

for i in range(len(croutipoints)):
    print(i)
    contenders = [rd.randint(0,len(croutipoints)-1) for k in range(100)]
    predictions = model.predict([np.array([croutipoints[i].pixels for k in range(100)]), np.array([croutipoints[contenders[k]].pixels for k in range(100)])])
    for j in range(len(predictions)):
        if(predictions[j][0] > predictions[j][1]):
            croutipoints[i].rating, croutipoints[j].rating = trueskill.rate_1vs1(croutipoints[i].rating, croutipoints[j].rating)
        else:
            croutipoints[j].rating, croutipoints[i].rating = trueskill.rate_1vs1(croutipoints[j].rating, croutipoints[i].rating)


for k in range(len(croutipoints)):
    print(croutipoints[k].rating)

idx = range(len(croutipoints))

df = pd.DataFrame.from_items([('name',[croutipoints[k].name for k in range(len(croutipoints))]),
                              ('mu', [croutipoints[k].rating.mu for k in range(len(croutipoints))]),
                              ('sigma', [croutipoints[k].rating.sigma for k in range(len(croutipoints))])
                              ])

df.to_csv(os.path.join(baseDir,"trueskillScores200Duels.csv"))