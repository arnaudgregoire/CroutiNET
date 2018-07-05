import pandas as pd
import os
import scipy.stats as sc
import matplotlib.pyplot as plt

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"

df = pd.read_csv(os.path.join(base_dir, "walkscore_compare.csv"))

print("walkscore score correlation")
print(sc.spearmanr(df['score'].tolist(), df['walkscore'].tolist()))

print("bikescore score correlation")
print(sc.spearmanr(df['score'].tolist(), df['bikescore'].tolist()))

print("normalize_walkscore normalize_score_inverted correlation")
print(sc.spearmanr(df['normalize_score_inverted'].tolist(), df['normalize_walkscore'].tolist()))

print("normalize_bikescore normalize_score_inverted correlation")
print(sc.spearmanr(df['normalize_score_inverted'].tolist(), df['normalize_bikescore'].tolist()))


print("walkscore score correlation")
print(sc.pearsonr(df['score'].tolist(), df['walkscore'].tolist()))

print("bikescore score correlation")
print(sc.pearsonr(df['score'].tolist(), df['bikescore'].tolist()))

print("normalize_walkscore normalize_score_inverted correlation")
print(sc.pearsonr(df['normalize_score_inverted'].tolist(), df['normalize_walkscore'].tolist()))

print("normalize_bikescore normalize_score_inverted correlation")
print(sc.pearsonr(df['normalize_score_inverted'].tolist(), df['normalize_bikescore'].tolist()))

plt.figure()
plt.plot(df['normalize_score_inverted'].tolist(), df['normalize_walkscore'].tolist(), 'o')
plt.title('Correlation Croutiscore/Walkscore')
plt.xlabel('normalize_score_inverted')
plt.ylabel('normalize_walkscore')
plt.show()

plt.figure()
plt.plot(df['normalize_score_inverted'].tolist(), df['normalize_bikescore'].tolist(), 'o')
plt.title('Correlation Croutiscore/Bikecore')
plt.xlabel('normalize_score_inverted')
plt.ylabel('normalize_bikescore')
plt.show()