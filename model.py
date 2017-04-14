import csv
import os
import cv2
import numpy as np

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			correction = 0.2
			for sample in samples:
				for i in range (3):
					source_path = sample[i]
					filename = source_path.split('/')[-1]
					current_path = 'data/IMG/' + filename
					image = cv2.imread(current_path)
					images.append(image)
					measurement = float(sample[3])
					# Left image
					if i == 1 and measurement > 0.2:
						measurement = measurement + correction
					# Right image
					if i == 2 and measurement > 0.2:
						measurement = measurement - correction
					measurements.append(measurement)
					augmented_image = cv2.flip(image,1)
					images.append(augmented_image)
					augmented_measurement = measurement*-1.0
					measurements.append(augmented_measurement)
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse', 'accuracy'])
history_object = model.fit_generator(train_generator, 
	samples_per_epoch=len(train_samples), validation_data=validation_generator, 
	nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')
import gc; gc.collect()

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss')
