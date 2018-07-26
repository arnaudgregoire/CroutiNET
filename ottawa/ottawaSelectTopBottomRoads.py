import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

#Define Directories
baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
roadsDir = os.path.join(baseDir, "roads")
df_dir = os.path.join(baseDir, "trueskillScores20Duels.csv")
save_dir = os.path.join(baseDir, "trueskill20Duels")

df = pd.read_csv(df_dir)

top_roads = df.nlargest(2000, 'mu')
bottom_roads = df.nsmallest(2000, 'mu')

top = os.path.join(save_dir, "top2k")
bottom = os.path.join(save_dir, "bottom2k")

if not os.path.exists(top):
    os.mkdir(os.path.join(save_dir,"top2k"))

if not os.path.exists(bottom):
    os.mkdir(os.path.join(save_dir, "bottom2k"))

top = os.path.join(save_dir, "top2k")
bottom = os.path.join(save_dir, "bottom2k")

top_names = top_roads['name'].tolist()
top_scores = top_roads['mu'].tolist()

bottom_names = bottom_roads['name'].tolist()
bottom_scores = bottom_roads['mu'].tolist()

plt.figure()
plt.title("Top scores histogram (mean : " + str(np.mean(top_scores)) + " )")
plt.hist(top_scores, 50)
plt.xlabel("top scores")
plt.ylabel("fréquence d'apparition")
plt.legend()
plt.show()

plt.figure()
plt.title("Bottom scores histogram (mean : " + str(np.mean(bottom_scores)) + " )")
plt.hist(bottom_scores, 50)
plt.xlabel("bottom scores")
plt.ylabel("fréquence d'apparition")
plt.legend()
plt.show()

for i in range(len(top_names)):
    shutil.copy(os.path.join(roadsDir, top_names[i]),
                os.path.join(top, top_names[i] + ".png"))
    shutil.copy(os.path.join(roadsDir, bottom_names[i]),
                os.path.join(bottom, bottom_names[i] + ".png"))