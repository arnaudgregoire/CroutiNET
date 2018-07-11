import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr
import scipy.stats as sc
import pandas as pd


"""
used to compute the correlation between Comparaison Croutinet + trueskill and Walkscore/bikescore
"""

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
shp_dir = os.path.join(base_dir, "roads_shp")


# Shapefile with points on roads with bearing of roads and segment grouping variable. There 105303 points in this file
ds = ogr.Open(os.path.join(shp_dir,"comparaisonScore20DuelsWalkscore.dbf"))

# Get layer
layer = ds.GetLayer()

# Create list of road scores
scores =[[row.GetField("mu"), row.GetField("walkscore_"), row.GetField("walkscor_1")] for row in layer]

trueskillScores = []
walkScores = []
bikeScores = []

for line in scores:
    trueskillScores.append(float(line[0]))
    walkScores.append(float(line[1]))
    bikeScores.append(float(line[2]))


trueskillScoresMean = np.mean(trueskillScores)
walkScoresMean = np.mean(walkScores)
bikeScoresMean = np.mean(bikeScores)


plt.figure()
plt.hist(walkScores,50)
plt.title("Histogramme walkScores (moyenne :" + str(walkScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("walkScores")
plt.legend()
plt.show()

plt.figure()
plt.hist(bikeScores,50)
plt.title("Histogramme bikeScores (moyenne :" + str(bikeScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("bikeScores")
plt.legend()
plt.show()

plt.figure()
plt.hist(trueskillScores,50)
plt.title("Histogramme trueskillScoresMean (moyenne :" + str(trueskillScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()
plt.show()

plt.figure()
plt.plot(trueskillScores, walkScores, 'o')
plt.title('Correlation trueskillScores/walkScores (' + str(sc.spearmanr(trueskillScores,walkScores)) + " )")
plt.xlabel('trueskillScores')
plt.ylabel('walkScores')
plt.show()

plt.figure()
plt.plot(trueskillScores, bikeScores, 'o')
plt.title('Correlation trueskillScores/bikeScores (' + str(sc.spearmanr(trueskillScores,bikeScores)) + " )")
plt.xlabel('trueskillScores')
plt.ylabel('bikeScores')
plt.show()
