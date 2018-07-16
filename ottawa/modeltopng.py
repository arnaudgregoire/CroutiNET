from keras.models import load_model
import os
from keras.utils import plot_model


#Define directories
baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
models_dir = os.path.join(baseDir, "models")
model = load_model(os.path.join(models_dir, "scoreNetworkNoSigmoid.h5"))
plot_model(model, to_file=os.path.join(baseDir,'scoreCroutiNET.png'))