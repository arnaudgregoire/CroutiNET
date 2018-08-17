import os
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import shutil as sh

"""
Add latitude and longitude to each point with given id
Save 50 pictures with top scores and 50 pictures with worst score

This file execute a script that takes 2 files in input :

scores2.csv :The first is scores of each points (in .csv)
roads_dictionnary.csv : The second is a dictionnary that contain at least the longitude, the latitude and the id of each point

"""

#Define src
# the main directory
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
# the file with name of the picture, latiude, longitude like "roads_recent_correct_dictionnary"
roads_dictionnary_dir = os.path.join(base_dir, "roads_dictionnary.csv")
# the roads folder with all pictures (that were used during training) like "recent_roads"
roads_dir = os.path.join(base_dir, "roads")
# the score csv like "scoresRecentRoads.csv"
scores_dir = os.path.join(base_dir, "trueskillScores20Duels.csv")
#the new score csv with the latitude and the longitude
merged_dir = os.path.join(base_dir, "trueskillScoreLatLong20Duels.csv")
#a folder where save the the top 50 pictures with highest scores and the 50 bottom pictures with lowest scores
modelFolder = os.path.join(base_dir, "trueskill20Duels")
# the two subfolder
bottom_50_dir = os.path.join(modelFolder, "bottom50")
top_50_dir = os.path.join(modelFolder, "top50")

# Build 2 dataframes
#score_results = np.loadtxt(scores_dir, str, delimiter=',')
roads_result = np.loadtxt(roads_dictionnary_dir, str, delimiter=',')

#scoresDF = pd.DataFrame(score_results, None, ['name', 'score'])
scoresDF = pd.read_csv(scores_dir)
roadsDF = pd.DataFrame(roads_result, None, ['long', 'lat', 'heading', 'id'])

# The name that pictures have is a combinaison of of year/id/heading that we split in 3 column here
scoresDF['year'] = scoresDF['name'].str[0:4]
scoresDF['id'] = scoresDF['name'].str[5:27]
scoresDF['heading'] = scoresDF['name'].str[28:41]

# Join on respectives ids of both dataframes
merged = scoresDF.merge(roadsDF, on='id')

# Save csv
merged.to_csv(merged_dir)

plt.figure()
merged['mu'] = pd.to_numeric(merged['mu'])
plt.hist(merged['mu'].tolist(),50)
plt.ylabel("Frequence d'apparition")
plt.xlabel("mu value")
plt.title("Histogramme of scores values (mu)")
plt.legend()
plt.show()


plt.figure()
plt.title("Histogramme of scores values (sigma)")
merged['sigma'] = pd.to_numeric(merged['sigma'])
plt.hist(merged['sigma'].tolist(),50)
plt.ylabel("Frequence d'apparition")
plt.xlabel("sigma value")
plt.title("Histogramme of sigma values (sigma)")
plt.legend()
plt.show()

top50 = merged.nlargest(50, 'mu')
bottom50 = merged.nsmallest(50,'mu')

for index, row in top50.iterrows() :
    sh.copy(os.path.join(roads_dir, row['name']),os.path.join(top_50_dir, row['name']))

for index, row in bottom50.iterrows() :
    sh.copy(os.path.join(roads_dir, row['name']),os.path.join(bottom_50_dir, row['name']))
