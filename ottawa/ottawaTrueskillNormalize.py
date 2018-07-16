import pandas as pd
import os
import matplotlib.pyplot as plt

#Define directories
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
scores_dir = os.path.join(base_dir, "trueskillScores20Duels.csv")

df = pd.read_csv(scores_dir)

df['mu'] = pd.to_numeric(df['mu'])
df['sigma'] = pd.to_numeric(df['sigma'])

df['normalize_mu'] = df['mu'] - df['mu'].mean()
df['normalize_mu'] = df['normalize_mu'] / df['normalize_mu'].std()

mus = df['normalize_mu'].tolist()
plt.figure()
plt.hist(mus, 50)
plt.xlabel("normalize mu")
plt.ylabel("fréquence d'apparition")
plt.title("histogramme normalisé de mu (ComparaisonCroutinet)")

df.to_csv(os.path.join(base_dir, "trueskillScores20DuelsNormalize.csv"))