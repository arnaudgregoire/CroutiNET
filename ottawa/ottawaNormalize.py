import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Script used  to normalize the distribution of ScoreCroutinet 
'''

#Define directories
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
merged_dir = os.path.join(base_dir, "merged2.csv")
normalize_dir = os.path.join(base_dir, "normalize_score2.csv")

#We load Croutinet score
score_result = np.loadtxt(merged_dir, str, delimiter=',')
scoreDF = pd.DataFrame(score_result, None, ['field_1', 'name', 'score', 'year', 'id', 'heading_x', 'long', 'lat', 'heading_y'])

# We drop fields that don't interest us
scoreDF = scoreDF.drop('heading_x', axis=1)
scoreDF = scoreDF.drop('field_1', axis=1)
scoreDF = scoreDF.drop(0)

#We transform score from string to float
scoreDF['score'] = scoreDF['score'].astype(float)

#We save the score into a list to compute numpy statistics stuff on it
score_list = scoreDF['score'].tolist()

#Here we compute the mean of the scores
mean = np.mean(score_list)

#and here the standard deviation
standard_deviation = np.std(score_list)

#Here we normalize both distributions
scoreDF['normalize_score'] = scoreDF['score'] - mean
scoreDF['normalize_score'] = scoreDF['normalize_score'] / standard_deviation

#Here we invert the distribution ( in fact, normal score given by ScoreCroutinet is the lowest value of ScoreCroutinet give the highest walkalbilty)
# Thats the opposite in Walkscore and thats why we need this inversion
scoreDF['normalize_score_inverted'] = - scoreDF['normalize_score']

# Here we plot the result to see if everything went fine
a = scoreDF['normalize_score_inverted'].tolist()
plt.figure()
plt.hist(a,50)
plt.title("Histogramme ScoreCroutinet (inversé)")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()
plt.show()


scoreDF.to_csv(normalize_dir)
