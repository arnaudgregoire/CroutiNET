import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr
import scipy.stats as sc
import pandas as pd

"""
used to compute the correlation between ScoreCroutinet and Comparaison Croutinet + trueskill
"""

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
shp_dir = os.path.join(base_dir, "roads_shp")


# Shapefile with points on roads with bearing of roads and segment grouping variable. There 105303 points in this file
ds = ogr.Open(os.path.join(shp_dir,"comparaisonScore20Duels.dbf"))

# Get layer
layer = ds.GetLayer()

# Create list of road scores
scores =[[row.GetField("trueskil_1"), row.GetField("scoresPoin")] for row in layer]

trueskillScores = []
croutiScores = []

for line in scores:
    trueskillScores.append(line[0])
    croutiScores.append(line[1])

croutiScoresMean = np.mean(croutiScores)
trueskillScoresMean = np.mean(trueskillScores)

plt.figure()
plt.hist(croutiScores,50)
plt.title("Histogramme croutiScoresMean (moyenne :" + str(croutiScoresMean) + ")")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
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
plt.plot(croutiScores, trueskillScores, 'o')
plt.title('Correlation croutiScores/trueskillScores (' + str(sc.spearmanr(croutiScores,trueskillScores)) + " )")
plt.xlabel('croutiScores')
plt.ylabel('trueskillScores')
plt.show()

