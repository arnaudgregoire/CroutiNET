import os
from random import randint

import numpy as np
import pandas as pd
from keras.models import load_model

from generator import dataGenerator as d
from loader import loadAsScalars
from modelRanking import create_base_network, create_meta_network
from representation.representation import show

IMG_SIZE  = 224
INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"

trainDir = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
testDir = os.path.join(baseDir, "test/test.csv")
roads_loubna_dir = os.path.join(baseDir, "roads_loubna")
hyperas_model_dir = os.path.join(baseDir, 'hyperasScoreModel.h5')
base_network_save = os.path.join(baseDir, "scoreNetworkNoSigmoidHyperas.h5")
ranking_network_save = os.path.join(baseDir, "rankNetworkNoSigmoidHyperas.h5")
histories = []


all_results = np.loadtxt(trainDir, str, delimiter=',')

duelsDF = pd.DataFrame(all_results, None, ['left_id', 'right_id', 'winner'])
duelsDF['left_id'] = roads_loubna_dir + "/" + duelsDF['left_id']
duelsDF['right_id'] = roads_loubna_dir + "/" + duelsDF['right_id']
#print(duelsDF)

mask_yes = duelsDF['winner'] == '1'
yes = duelsDF[mask_yes]

mask_no = duelsDF['winner'] == '0'
no = duelsDF[mask_no]

base_network = load_model(hyperas_model_dir)
model = create_meta_network(INPUT_DIM, base_network)

validationLeft, validationRight, validationLabels = loadAsScalars(validationDir)

# For batch training, the number of iterations of training model
n_iter = 50
for iteration in range(n_iter):
    print(iteration / n_iter)

    # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
    # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
    # sizes.
    yesno = yes.sample(500).append(no.sample(500))
    print('yesno created')
    labels = dict(zip([str(x) for x in yesno.index.tolist()],
                      [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
    print('labels created')

    partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}
    print('partition created')

    batchSizeAug = len(yesno.index.tolist())
    print('batcgSizeAug created')
    # Set-up variables for augmentation of current batch of yesno in partition
    params = {
        'dim_x': IMG_SIZE,
        'dim_y': IMG_SIZE,
        'dim_z': 3,
        'batch_size': batchSizeAug,
        'shuffle': True
    }
    print('params created')

    datagenargs = {
        'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
    }
    print('datagenargs created')

    training_generator = d.myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
                                                                 datagenargs=datagenargs)
    print('training generator created')

    X, y = training_generator.__next__()
    print('X,y created')
    # zero center images
    X = np.array(X)
    print('X as arrray created')


    history = model.fit(
        [X[0], X[1]],
        y,
        batch_size=16,
        epochs=10,
        validation_data=([validationLeft, validationRight], validationLabels))
    histories.append(history)

show(histories, False)

base_network.save(base_network_save)
model.save(ranking_network_save)
