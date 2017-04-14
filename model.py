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
		# print(i)
		# print(filename)
		# print(measurement)
		# Left image
		if i == 1 and measurement > 0.2:
			measurement = measurement + correction
		# Right image
		if i == 2 and measurement > 0.2:
			measurement = measurement - correction
		# print(measurement)
		measurements.append(measurement)
		augmented_image = cv2.flip(image,1)
		images.append(augmented_image)
		augmented_measurement = measurement*-1.0
		measurements.append(augmented_measurement)



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

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

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse', 'accuracy'], verbose=1)
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

import gc; gc.collect()