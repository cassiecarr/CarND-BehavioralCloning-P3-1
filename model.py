import csv
import os
import cv2
import numpy as np
import utils

# Read the csv file for image path
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		samples.append(line)

# Seperate the data into training and validation sets
import sklearn
from sklearn.model_selection import train_test_split
sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define a generator to preprocess images, save images and 
# add augmented images to the dataset
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			measurements, images = utils.preprocess(batch_samples)

			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# Import needed Keras functions
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras import optimizers

# Call generator for training and validation data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Appy NVIDIA Architecture using Keras model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (66,200,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

adam = optimizers.Adam(lr=0.00001)
model.compile(loss = 'mse', optimizer = adam, metrics=['mse'])

# Generate the model
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, \
	validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5, \
	verbose=1)

# Save the model
model.save('model.h5')
import gc; gc.collect()

# Print the keys contained in the history object
print(history_object.history.keys())

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss')
