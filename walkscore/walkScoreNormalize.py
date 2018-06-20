import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Define directories
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
merged_dir = os.path.join(base_dir, "walkscore_dictionnary.csv")
normalize_dir = os.path.join(base_dir, "normalize_walkscore_dictionnary.csv")

score_result = np.loadtxt(merged_dir, str, delimiter=',')
scoreDF = pd.DataFrame(score_result, None, ['long', 'lat', 'id', 'walkscore', 'bikescore'])

scoreDF = scoreDF.drop(scoreDF[scoreDF['walkscore'] == 'None'].index)
scoreDF = scoreDF.drop(scoreDF[scoreDF['bikescore'] == 'None'].index)

scoreDF['walkscore'] = scoreDF['walkscore'].astype(float)
scoreDF['bikescore'] = scoreDF['bikescore'].astype(float)

walkscore_list = scoreDF['walkscore'].tolist()
mean_walkscore = np.array(walkscore_list).mean()
standard_deviation_walkscore = np.array(walkscore_list).std()

bikescore_list = scoreDF['bikescore'].tolist()
mean_bikescore = np.array(bikescore_list).mean()
standard_deviation_bikescore = np.array(bikescore_list).std()

scoreDF['normalize_walkscore'] = scoreDF['walkscore'] - mean_walkscore
scoreDF['normalize_walkscore'] = scoreDF['normalize_walkscore'] / standard_deviation_walkscore

scoreDF['normalize_bikescore'] = scoreDF['bikescore'] - mean_bikescore
scoreDF['normalize_bikescore'] = scoreDF['normalize_bikescore'] / standard_deviation_bikescore

a = scoreDF['normalize_walkscore'].tolist()
b = scoreDF['normalize_bikescore'].tolist()

scoreDF.to_csv(normalize_dir)

plt.figure()
plt.hist(a,50)
plt.title("Histogramme normalize walkscore")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()

plt.figure()
plt.hist(b,50)
plt.title("Histogramme normalize bikescore")
plt.ylabel("Frequence d'apparition")
plt.xlabel("score")
plt.legend()
plt.show()


