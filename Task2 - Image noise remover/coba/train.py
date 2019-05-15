# Code inspired by
# https://blog.keras.io/building-autoencoders-in-keras.html
# (also visit this post to learn more about autoencoders)
#
# 0. Import all the necessary elements
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import random

def get_noisy_mnist(noise_factor=0.05, data_ratio=.20):

    x_pos = []
    base_path = 'SMILEs/'
    positive_smiles = base_path + 'positives/positives7/'
    negative_smiles = base_path + 'negatives/negatives7/'
    for img in os.listdir(positive_smiles):
        x_pos.append(mpimg.imread(positive_smiles + img))

    x_pos = np.array(x_pos)

    x_train = x_pos[int(x_pos.shape[0] * data_ratio):]
    x_test = x_pos[:int(x_pos.shape[0] * data_ratio)]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))  # adapt this if using `channels_first` image data format

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train, x_train_noisy, x_test, x_test_noisy


# 3. Getting data.
x_train, x_train_noisy, x_test, x_test_noisy = get_noisy_mnist()

plt.figure(figsize=(8, 10))
plt.subplot(3,2,1).set_title('normal')
plt.subplot(3,2,2).set_title('noisy')
plt.tight_layout()
n = 6
for i in range(1,n+1,2):
    # 2 columns with good on left side, noisy on right side
    ax = plt.subplot(3, 2, i)

    rand_index = random.randint(0, len(x_train))
    ori = np.squeeze(x_train[rand_index])
    noise = np.squeeze(x_train_noisy[rand_index])
    # plot normal images
    plt.imshow(ori, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot noisy images
    ax = plt.subplot(3,2,i+1)
    plt.imshow(noise, cmap=plt.cm.gray)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.show()

# 1. Building model.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# 2. Compiling model.
model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.summary()

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# 4. Training data.
model.fit(x_train_noisy, x_train,
                epochs=50, # to get significant results change to 100
                batch_size=32,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')

s1 = model.predict(x_test_noisy)

plt.figure(figsize=(12, 12))
plt.subplot(3,4,1).set_title('normal')
plt.subplot(3,4,2).set_title('noisy')
plt.subplot(3,4,4).set_title('denoised')
n = 3
for i in range(1,12,4):
    img_index = random.randint(0,len(x_test_noisy))
    # plot original image
    ax = plt.subplot(3, 4, i)
    plt.imshow(np.squeeze(x_test[img_index]), cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot noisy images
    ax = plt.subplot(3,4,i+1)
    plt.imshow(np.squeeze(x_test_noisy[img_index]), cmap=plt.cm.gray)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    # plot denoised sequential
    ax = plt.subplot(3,4,i+3)
    plt.imshow(np.squeeze(s1[img_index]), cmap=plt.cm.gray)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.tight_layout()
plt.show()