import os

from keras.models import load_model

from modelRanking import create_base_network, create_meta_network
from loader import  loadAsScalars
from representation.representation import show

'''
File used to train a model without data augmentation
'''

#Define the img size
IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

#Define directories
baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir      = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")

base_network_save = os.path.join(baseDir, "scoreNetworkRetrain2.h5")
ranking_network_save = os.path.join(baseDir, "rankingNetworkRetrain.h5")

base_network_save2 = os.path.join(baseDir, "scoreNetworkRetrain3.h5")

#load training and validation set with labels as scalars between 0 and 1
trainLeft, trainRight, trainLabels                = loadAsScalars(trainDir)
validationLeft, validationRight, validationLabels = loadAsScalars(validationDir)

#Here is the architecture of ScoreCroutinet that we create below
base_network = load_model(base_network_save)
model = create_meta_network(INPUT_DIM, base_network)

#We fit the model to the training set
history = model.fit(
        [trainLeft, trainRight],
        trainLabels,
        batch_size=16,
        epochs=30,
        validation_data=([validationLeft, validationRight], validationLabels))

#We show the result and save the network
show([history], False)
base_network.save(base_network_save2)