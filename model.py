import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
	for i in range (3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		# Left image
		if i == 1:
			measurement = measurement + 0.65
		# Right image
		if i == 2:
			measurement = measurement - 0.1
		measurements.append(measurement)
		augmented_image = cv2.flip(image,1)
		images.append(augmented_image)
		augmented_measurement = measurement*-1.0
		measurements.append(augmented_measurement)



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse', 'accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

import gc; gc.collect()