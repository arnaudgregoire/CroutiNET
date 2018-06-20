import requests
import os
import json
import numpy as np
import pandas as pd

base_dir = r"D:\Arnaud\data_croutinet\ottawa\data"
roads_dictionnary_dir = os.path.join(base_dir, "roads_dictionnary.csv")
walkscore_dir = os.path.join(base_dir, "walkscore_dictionnary.csv")

API_KEYS = []
EVIL_API_KEYS = []

roads_result = np.loadtxt(roads_dictionnary_dir, str, delimiter=',')
roadsDF = pd.DataFrame(roads_result, None, ['long', 'lat', 'heading', 'id'])
roadsDF['walkScore'] = None
roadsDF['bikeScore'] = None


def getScores(lat, long, api_key):
    url = "http://api.walkscore.com/score"
    params = {
        # maximum permitted size for free calls
        "format": "json",
        "lat": lat,
        "lon": long,
        "wsapikey": api_key,
        "transit": 1,
        "bike": 1
    }

    response = requests.get(url, params=params, stream=True)
    #print(response.url)
    res = json.loads(response.content)
    try:
        scores = (res['walkscore'], res['bike']['score'])
    except:
        print("scores not found")
        scores = False
    #print(scores)
    return scores

def computeScore(first, last, api_key):
    for i in range(first,last):
        print(str(np.floor((i - first) * 100/ (last - first)))+ "%")
        scores = getScores(roadsDF.iloc[i]['lat'], roadsDF.iloc[i]['long'],api_key)
        if scores != False:
            roadsDF.loc[i,'walkScore'] = scores[0]
            roadsDF.loc[i,'bikeScore'] = scores[1]

    longitudes = []
    latitudes = []
    ids = []
    walkScores = []
    bikeScores = []

    for i in range(first, last):
        longitudes.append(roadsDF.iloc[i]['long'])
        latitudes.append(roadsDF.iloc[i]['lat'])
        ids.append(roadsDF.iloc[i]['id'])
        walkScores.append(roadsDF.iloc[i]['walkScore'])
        bikeScores.append(roadsDF.iloc[i]['bikeScore'])

    with open(walkscore_dir, 'a') as f:
        for i in range(len(longitudes)):
            f.write("{},{},{},{},{}\n".format(longitudes[i], latitudes[i], ids[i], walkScores[i], bikeScores[i]))

#24657 items in roadsDF

computeScore(0,4500,API_KEYS[0])
computeScore(4501,9000,API_KEYS[1])
computeScore(9001,12000,EVIL_API_KEYS[0])
computeScore(12001,15000,EVIL_API_KEYS[1])
computeScore(15001,18000,EVIL_API_KEYS[2])
computeScore(18001,21000,EVIL_API_KEYS[3])
computeScore(21001,24656,EVIL_API_KEYS[4])


