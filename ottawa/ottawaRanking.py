import os
from modelRanking import create_base_network, create_meta_network
from loader import  loadAsScalars
from representation.representation import show

IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir      = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")

base_network_save = os.path.join(baseDir, "scoreNetworkNoSigmoid.h5")

trainLeft, trainRight, trainLabels                = loadAsScalars(trainDir)
validationLeft, validationRight, validationLabels = loadAsScalars(validationDir)

base_network = create_base_network(INPUT_DIM)
model = create_meta_network(INPUT_DIM, base_network)

history = model.fit(
        [trainLeft, trainRight],
        trainLabels,
        batch_size=16,
        epochs=30,
        validation_data=([validationLeft, validationRight], validationLabels))

show([history], False)
base_network.save(base_network_save)