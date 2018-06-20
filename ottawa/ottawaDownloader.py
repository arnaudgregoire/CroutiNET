import numpy as np
from scipy import misc
#import gist
import streetview
from osgeo import ogr
import random
import pandas as pan
import os


'''
Script modified from amaury's downloader.py.  Specifically this randomly selects a point within a road segment for each road segment.
Then, from that list of random points, a subset is randomly selected for downlading.  For each point with more than one
available panorama, a single one is randomly selected for download at a given location.
A shapefile must be available with the random points and must contain the downstreet bearing as well as a field that identifies
each road segment.
I used polyline to segment tool in arcgis, followed by the add geometry bearing tool and finally the vertices to segments and a spatial join from segments to vertices
to create the shapefile used here.
'''

API_KEY_1 = "your_api_key"
API_KEY_2 = "your_api_key"

# ------------------------------------------------------------------------------

base_dir = r"C:\Users\msawada\Desktop\arnaud\croutinet\ottawa"
shp_dir = os.path.join(base_dir, "shp")

"""
Downloads Google StreetView images of random points in Ottawa
:param n_loc: number of locations to download. CAUTION: some points have no images, so it's not the exact number of subdirectories created
"""

# Output folder for panaorama images
DIRECTORY = os.path.join(base_dir, "roads")

# Shapefile with points on roads with bearing of roads and segment grouping variable. There 105303 points in this file
ds = ogr.Open(os.path.join(shp_dir,"roadpointswithbearing2.dbf"))

# number of samples for each road segment, e.g., how many points on a road segment do you want to get photos at
nptsSample = 1

# number of subsamples for all grouped road segments, e.g., there are 26010 unique road segments
nsubsample = 26010

# Get layer
layer = ds.GetLayer()

# Create list of road segment identifiers
elist=[row.GetField("RD_SEGMENT") for row in layer]

# Create list of indices which is the same length as the number of rows in the
#   shapefile containing the x,y,bearing information (could also get an object id - if base 0 in shapefile)
idx = range(0,len(elist))

# Merge the list of idices and segment identifiers into a pandas dataframe for sampling
df=pan.DataFrame.from_items([('idx',idx),('elist',elist)])

# Sample n points on each road segment.  Each road segment contains the same identifier
#   so group by identifier and then sample n points within each group outputting to the new dataframe.

tt=df.groupby('elist').apply(pan.DataFrame.sample, n=nptsSample).reset_index(drop=True)

# Get the IDs from the random sample dataframe, tt, in order to identify the rows in the original
#   shapefile at which to get the panorama
ransample=tt['idx'].tolist()


# Get a set of n random subsamples from grouped road segments
#n_loc=[random.randint(0,len(ransample)) for i in range(0,nsubsample)]

# Download for each random segment subsample
# also write a csv file to map locations for validation

def download(start_idx, end_idx, api_key):
    with open(os.path.join(base_dir, "roads_dictionnary.csv"), 'a') as f:
        for i in range(start_idx, end_idx) :#range(0,tt.shape[0])

            print('%.2f' % ((i - start_idx) * 100 / (end_idx - start_idx)) + " %")

            # get feature geometry and bearing from shapefile
            feature = layer[ransample[i]]

            lon = feature.GetGeometryRef().GetX()

            lat = feature.GetGeometryRef().GetY()

            heading = feature.GetField("BEARING")

            # Get the number of panaoramas at the location
            panIds = streetview.panoids(lat, lon)

            # Randomly select one of the n panoramas at this location
            if len(panIds) > 0:

                pid = random.randint(0,len(panIds)-1)
                f.write("{},{},{},{}\n".format(lon, lat, heading, panIds[pid]["panoid"]))
                img = streetview.api_download(panIds[pid]["panoid"], heading, DIRECTORY, api_key, fov=80, pitch=0)

download(0, int(nsubsample/2),API_KEY_1)
download(int(nsubsample/2) +1 ,nsubsample -1,API_KEY_2)

#download(0,5,API_KEY_2)