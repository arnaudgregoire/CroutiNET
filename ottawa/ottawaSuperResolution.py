import os

from keras.callbacks import ModelCheckpoint

import preprocessSRCNN as pd
from modelSRCNN import build_model
from representation.representation import show

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
model_dir = os.path.join(baseDir, "models")

def train():
    data, label = pd.read_training_data(os.path.join(model_dir, "crop_train.h5"))
    val_data, val_label = pd.read_training_data(os.path.join(model_dir, "test.h5"))

    model = build_model()

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')

    history = model.fit(data, label, batch_size=24, validation_data=(val_data, val_label),
    callbacks=[checkpoint], shuffle=True, epochs=5)

    show([history],False)

train()