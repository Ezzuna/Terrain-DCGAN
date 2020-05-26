from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.python.client import device_lib
import time #For tracking computation time
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt



print(tf.test.is_gpu_available())
print(device_lib.list_local_devices())
start_time = time.time()


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

previewRows = 4
previewCols = 7
previewPadding = 5

EPOCHS = 100000
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128
IMAGE_CHANNELS = 1
SAVE_FREQ = 100
NOISE_SIZE = 100

trainingData = np.load(os.path.join('D:\\Python\\', 'SameColourRemoved2.npy'))  #change to your data directory

#Currently not changed from Keras documentation fit for an RGB image NN. Results need to be tested to see if
#Further editing is required.

def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
    input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)


def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation='relu', input_dim = noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    for i in range(GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))
    input = Input(shape=(noise_size,))
    generated_image = model(input)

    return Model(input, generated_image)

def save_images(cnt, noise):
    image_array = np.full((
       previewPadding + (previewRows * (IMAGE_SIZE + previewPadding)),
        previewPadding + (previewCols * (IMAGE_SIZE + previewPadding)), 3),
        255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c +
                        IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    output_path = 'outputColourRemovedFinal'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)


image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = Adam(1.5e-4, 0.5)
discriminator = build_discriminator(image_shape)

discriminator.compile(loss='binary_crossentropy',
optimizer = optimizer, metrics = ['accuracy'])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)
discriminator.summary()
discriminator.trainable = False

validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss='binary_crossentropy',
optimizer = optimizer, metrics = ['accuracy'])
y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
cnt = 1

discriminatorResults = []
discriminatorResultsSample = []
generatorResults = []
generatorResultsSample = []

discriminatorLossResults = []
discriminatorLossResultsSample = []
generatorLossResults = []
generatorLossResultsSample = []


for epoch in range(EPOCHS):
    idx = np.random.randint(0, trainingData.shape[0], BATCH_SIZE)
    x_real = trainingData[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
    x_fake = generator.predict(noise)

    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)

    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
    generator_metric = combined.train_on_batch(noise, y_real)

    discriminatorResults.append(100 * discriminator_metric[1])
    generatorResults.append(100 * generator_metric[1])
    discriminatorLossResults.append(discriminator_metric[0])
    generatorLossResults.append(generator_metric[0])

    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_noise)
        cnt += 1

        print(f"{epoch} epoch, Discriminator Loss: {discriminator_metric[0]}, Generator Loss: {generator_metric[0]}")
        print(f"{epoch} epoch, Discriminator MSE: {100 * discriminator_metric[1]}, Generator MSE: {100 * generator_metric[1]}")
        discriminatorResultsSample.append(100 * discriminator_metric[1])
        generatorResultsSample.append(100 * generator_metric[1])
        discriminatorLossResultsSample.append(discriminator_metric[0])
        generatorLossResultsSample.append(generator_metric[0])

print("--- %s seconds ---" % (time.time() - start_time))
import pandas

df = pandas.DataFrame(data={"Discriminator Accuracy": discriminatorResults, "Generator Accuracy": generatorResults})
dfSample100 = pandas.DataFrame(data={"Discriminator Accuracy": discriminatorResultsSample, "Generator Accuracy": generatorResultsSample})

dfLoss = pandas.DataFrame(data={"Discriminator Loss": discriminatorLossResults, "Generator Loss": generatorLossResults})
dfLossSample100 = pandas.DataFrame(data={"Discriminator Loss": discriminatorLossResultsSample, "Generator Loss": generatorLossResultsSample})
df.to_csv("./ColourRemoved3.csv", sep=',',index=False)
dfSample100.to_csv("./ColourRemoved3Sampler.csv", sep=',',index=False)
dfLoss.to_csv("./ColourRemoved3Loss.csv", sep=',',index=False)
dfLossSample100.to_csv("./ColourRemoved3SamplerLoss.csv", sep=',',index=False)
