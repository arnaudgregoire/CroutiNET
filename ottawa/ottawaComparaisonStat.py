import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
shp_dir = os.path.join(base_dir, "rankingNoSigmoid")

# Shapefile with points on roads with bearing of roads and segment grouping variable. There 105303 points in this file
dsRecent = ogr.Open(os.path.join(shp_dir,"scoresRecentPoint.dbf"))
dsOld = ogr.Open(os.path.join(shp_dir,"scoresOldPoint.dbf"))
# Get layer
layerRecent = dsRecent.GetLayer()
layerOld = dsOld.GetLayer()
# Create list of road scores
newScores=[row.GetField("score") for row in layerRecent]
oldScores=[row.GetField("score") for row in layerOld]

for i in range(len(newScores)-1):
    newScores[i]= -newScores[i]

for i in range(len(newScores)-1):
    oldScores[i]= -oldScores[i]

newScoresMean = np.mean(newScores)
oldScoresMean = np.mean(oldScores)

plt.figure()
plt.hist(newScores,50)
plt.title("Histogramme ScoreCroutinet new roads (moyenne :" + str(newScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()
plt.show()

plt.figure()
plt.hist(newScores,50)
plt.title("Histogramme ScoreCroutinet old roads (moyenne :" + str(oldScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()
plt.show()
