import os
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import shutil as sh

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
roads_dictionnary_dir = os.path.join(base_dir, "roads_dictionnary.csv")
roads_dir = os.path.join(base_dir, "roads")
scores_dir = os.path.join(base_dir, "scores2.csv")
merged_dir = os.path.join(base_dir, "merged2.csv")
modelFolder = os.path.join(base_dir, "rankingNoSigmoid")
bottom_50_dir = os.path.join(modelFolder, "bottom50")
top_50_dir = os.path.join(modelFolder, "top50")


score_results = np.loadtxt(scores_dir, str, delimiter=',')
roads_result = np.loadtxt(roads_dictionnary_dir, str, delimiter=',')

scoresDF = pd.DataFrame(score_results, None, ['name', 'score'])
roadsDF = pd.DataFrame(roads_result, None, ['long', 'lat', 'heading', 'id'])

scoresDF['year'] = scoresDF['name'].str[0:4]
scoresDF['id'] = scoresDF['name'].str[5:27]
scoresDF['heading'] = scoresDF['name'].str[28:41]

merged = scoresDF.merge(roadsDF, on='id')

merged.to_csv(merged_dir)

plt.figure()
plt.title("Histogramme of scores values")
merged['score'] = pd.to_numeric(merged['score'])
merged['score'].diff().hist()

top50 = merged.nlargest(50, 'score')
bottom50 = merged.nsmallest(50,'score')

for index, row in top50.iterrows() :
    sh.copy(os.path.join(roads_dir, row['name']),os.path.join(top_50_dir, row['name']))

for index, row in bottom50.iterrows() :
    sh.copy(os.path.join(roads_dir, row['name']),os.path.join(bottom_50_dir, row['name']))
