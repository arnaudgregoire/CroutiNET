import os
from keras.preprocessing import image
import numpy as np
from discriminator import build_discriminator
from gan import build_gan
from generator import build_generator
from loader import loadAsScalars
import scipy.misc as sc
import keras
import matplotlib.pyplot as plt

#Define Hyperparameters
latent_dim = 64
height     = 64
width      = 64
channels   = 3
iterations = 1000000
batch_size = 20

#Define directories
baseDir       = r"D:\Arnaud\data_croutinet\ottawa\data"
trainDir      = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
save_dir      = os.path.join(baseDir, "testgan")

generator = build_generator(latent_dim, channels)
discriminator = build_discriminator(height, width, channels)
gan = build_gan(latent_dim, generator, discriminator)

leftImages, rightImages, y_train, namesLeft, namesRight = loadAsScalars(trainDir)

x_train = np.zeros((leftImages.shape[0],width,height,channels))

for i in range(leftImages.shape[0]):
    x_train[i] = sc.imresize(leftImages[i],(width,height))
    x_train[i] = x_train[i].astype('float32') / 255.

# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
# x_train = x_train[y_train.flatten() == 6]
# x_train = x_train.reshape(
# (x_train.shape[0],) +
# (height, width, channels)).astype('float32') / 255.


start = 0
discriminator_losses = []
adversarial_losses = []

for step in range(iterations):
    #Samples random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    #Decodesthem to fake images
    generated_images = generator.predict(random_latent_vectors)

    #Combines them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    #Assembles labels, discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])

    #Adds random noise to the labels
    labels += 0.05 * np.random.random(labels.shape)

    #trains the discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    #Samples random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size,
                                                   latent_dim))

    # Assembles labels that say "these are all real images"
    misleading_targets = np.zeros((batch_size, 1))

    #Trains the generator ( via the gan model where the discriminator weights are frozen )
    a_loss = gan.train_on_batch(random_latent_vectors,
                                misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0



    if step % 100 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss:', d_loss)
        discriminator_losses.append(d_loss)

        print('adversarial loss:', a_loss)
        adversarial_losses.append(a_loss)

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,
                              'generatedMultiDropout' + str(step) + '.png'))
        #img = image.array_to_img(real_images[0] * 255., scale=False)
        #img.save(os.path.join(save_dir,
        #                      'real' + str(step) + '.png'))

plt.figure()
plt.plot(range(len(discriminator_losses)),discriminator_losses, label='discriminator_losses')
plt.plot(range(len(adversarial_losses)),adversarial_losses, label='adversarial_losses')
plt.xlabel("iteration")
plt.ylabel("loss value")
plt.legend()
plt.show()