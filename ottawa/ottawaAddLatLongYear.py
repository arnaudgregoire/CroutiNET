import os
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import shutil as sh

"""
Add latitude and longitude and year to each point with given id

This file execute a script that takes 2 files in input :

scores2.csv :The first is scores of each points (in .csv)
roads_dictionnary.csv : The second is a dictionnary that contain at least the longitude, the latitude and the id of each point

"""

#Define src
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
roads_dictionnary_dir = os.path.join(base_dir, "roads_old_correct_dictionnary.csv")
roads_dir = os.path.join(base_dir, "old_roads")
scores_dir = os.path.join(base_dir, "scoresOldRoads.csv")
merged_dir = os.path.join(base_dir, "mergedOldRoads.csv")

# Build 2 dataframes
score_results = np.loadtxt(scores_dir, str, delimiter=',')
roads_result = np.loadtxt(roads_dictionnary_dir, str, delimiter=',')

scoresDF = pd.DataFrame(score_results, None, ['name', 'score'])
roadsDF = pd.DataFrame(roads_result, None, ['long', 'lat', 'heading','year', 'id'])


# The name that pictures have is a combinaison of of year/id/heading that we split in 3 column here
scoresDF['year'] = scoresDF['name'].str[0:4]
scoresDF['id'] = scoresDF['name'].str[5:27]
scoresDF['heading'] = scoresDF['name'].str[28:41]

#We transform score from string to float
scoresDF['score'] = scoresDF['score'].astype(float)

#We save the score into a list to compute numpy statistics stuff on it
score_list = scoresDF['score'].tolist()

#Here we compute the mean of the scores
mean = np.mean(score_list)

#and here the standard deviation
standard_deviation = np.std(score_list)

#Here we normalize both distributions
scoresDF['normalize_score'] = scoresDF['score'] - mean
scoresDF['normalize_score'] = scoresDF['normalize_score'] / standard_deviation

#Here we invert the distribution ( in fact, normal score given by ScoreCroutinet is the lowest value of ScoreCroutinet give the highest walkalbilty)
# Thats the opposite in Walkscore and thats why we need this inversion
scoresDF['normalize_score_inverted'] = - scoresDF['normalize_score']


# Join on respectives ids of both dataframes
merged = scoresDF.merge(roadsDF, on='id')

# Save csv
merged.to_csv(merged_dir)

plt.figure()
plt.title("Histogramme of scores values")
merged['normalize_score_inverted'].diff().hist()




