import os
from croutiPoint import  CroutiPoint
from loader import loadAsScalars
import trueskill
import random as rd
import  numpy as np

"""
Used to compute the average accuracy of ComparaisonCroutinet with Trueskill 0.7831858407079646
"""

#Define the img size
from keras.models import load_model

IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

#Define directories
baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir = os.path.join(baseDir, "validation/validation.csv")
roads_loubna_dir = os.path.join(baseDir, "roads_loubna")
models_dir = os.path.join(baseDir, "models")

model = load_model(os.path.join(models_dir, "modelWithDataAugmentation5.h5"))

left, right, labels, namesLeft, namesRight = loadAsScalars(trainDir)

croutipointsLeft = [CroutiPoint(namesLeft[i], left[i]) for i in range(len(namesLeft))]
croutipointsRight = [CroutiPoint(namesRight[i], right[i]) for i in range(len(namesRight))]

# Training left points
for i in range(len(croutipointsLeft)):
    print(i)
    contenders = [rd.randint(0,len(croutipointsLeft)-1) for k in range(30)]
    predictions = model.predict([np.array([croutipointsLeft[i].pixels for k in range(30)]), np.array([croutipointsLeft[contenders[k]].pixels for k in range(30)])])
    for j in range(len(predictions)):
        if(predictions[j][0] > predictions[j][1]):
            croutipointsLeft[i].rating, croutipointsLeft[j].rating = trueskill.rate_1vs1(croutipointsLeft[i].rating, croutipointsLeft[j].rating)
        else:
            croutipointsLeft[j].rating, croutipointsLeft[i].rating = trueskill.rate_1vs1(croutipointsLeft[j].rating, croutipointsLeft[i].rating)

# Training Right points
for i in range(len(croutipointsRight)):
    print(i)
    contenders = [rd.randint(0,len(croutipointsRight)-1) for k in range(30)]
    predictions = model.predict([np.array([croutipointsRight[i].pixels for k in range(30)]), np.array([croutipointsRight[contenders[k]].pixels for k in range(30)])])
    for j in range(len(predictions)):
        if(predictions[j][0] > predictions[j][1]):
            croutipointsRight[i].rating, croutipointsRight[j].rating = trueskill.rate_1vs1(croutipointsRight[i].rating, croutipointsRight[j].rating)
        else:
            croutipointsRight[j].rating, croutipointsRight[i].rating = trueskill.rate_1vs1(croutipointsRight[j].rating, croutipointsRight[i].rating)

croutiLabels = [croutipointsLeft[i].rating.mu - croutipointsRight[i].rating.mu for i in range(len(croutipointsRight))]

#compute Accuracy
correct = 0

for i in range(len(labels)):
    if labels[i] == 0 and croutiLabels[i] > 0:
        correct += 1

    if labels[i] == 1 and croutiLabels[i] < 0:
        correct += 1

print(correct/len(labels))

# 0.7831858407079646