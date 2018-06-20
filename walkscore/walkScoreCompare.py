import os
import pandas as pd

#Define directories
base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
merged_dir = os.path.join(base_dir, "walkscore_compare.csv")
walkscore_normalize_dir = os.path.join(base_dir, "normalize_walkscore_dictionnary.csv")
croutiscore_normalize_dir = os.path.join(base_dir, "normalize_score2.csv")

#build 2 dataframes, one with ScoreCroutinet and one with Walkscore
walkscoreDf = pd.read_csv(walkscore_normalize_dir, sep=",")
croutiscoreDf = pd.read_csv(croutiscore_normalize_dir, sep=",")

#remove old index
walkscoreDf = walkscoreDf.drop('Unnamed: 0', axis=1)
croutiscoreDf = croutiscoreDf.drop('Unnamed: 0', axis=1)

#merge 2 dataframes

merged = croutiscoreDf.merge(walkscoreDf, on='id')

merged['differenceWalkscoreCroutiscore'] = merged['normalize_walkscore'] - merged['normalize_score_inverted']
merged['differenceBikescoreCroutiscore'] = merged['normalize_bikescore'] - merged['normalize_score_inverted']

merged.to_csv(merged_dir)