import csv
import  os
import streetview
from download.sawada import api_download2
from scipy import misc


"""
Script used to download all pictures corresponding to locations of placePulse points
"""

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/"
wealthyDir = os.path.join(baseDir,"votesWealthy.csv")
imgDir = os.path.join(baseDir,"img")
API_KEY = "your_api_key"
lats = []
longs = []
ids = []

with open(wealthyDir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')
    for row in reader:
            ids.append(row[0])
            ids.append(row[1])
        #if row[3] not in lats:
            lats.append(row[3])
        #if row[5] not in lats:
            lats.append(row[5])
        #if row[4] not in longs:
            longs.append(row[4])
        #if row[6] not in longs:
            longs.append(row[6])
            reader.__next__()



for i in range(len(lats)):
    panIds = streetview.panoids(lats[i], longs[i])
    if len(panIds) > 0:
        pid = len(panIds) - 1
        img = api_download2(panIds[pid]["panoid"], ids[i], 0, imgDir, API_KEY, fov=80, pitch=0,year=panIds[pid]["year"])
