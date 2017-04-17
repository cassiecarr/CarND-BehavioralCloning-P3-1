import csv
import os
import cv2
import numpy as np

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

# Define a generator to save images and add augmented images to the dataset
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			correction = 0.2
			count_zero_measurement = 0
			for batch_sample in batch_samples:
				# Remove every 4th zero
				if float(batch_sample[3]) > 0.001:
					count_zero_measurement += 1
				else:
					count_zero_measurement = 0
				if count_zero_measurement > 3:
					continue
				else:
					count_zero_measurement = 0
				for i in range (3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = 'data/IMG/' + filename
					# Original image
					image = cv2.imread(current_path)
					# # Apply resize, crop, and apply histogram equalization to the image
					image = cv2.resize(image, (235,118))
					image = image[34:100, 17:217]
					image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
					# Add image
					images.append(image)
					# Steering angle
					measurement = float(batch_sample[3])
					# If left image, adjust steering angle
					if i == 1 and abs(measurement) > 0.2: 
						measurement = measurement + correction
					# If right image, adjust steering angle
					if i == 2 and abs(measurement) > 0.2: 
						measurement = measurement - correction
					# Add steering angle
					measurements.append(measurement)
					# Add horizontally flipped image to dataset, adjust steering angle
					augmented_image = np.fliplr(image)
					images.append(augmented_image)
					augmented_measurement = -measurement
					measurements.append(augmented_measurement)
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Call generator for training and validation data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Appy NVIDIA Architecture for Keras model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (66,200,3)))
# model.add(Cropping2D(cropping=((60,25), (0,0))))
# model.add(Convolution2D(6,5,5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5, activation="relu"))
# model.add(MaxPooling2D())
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
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
model.add(Dense(1))

# Load weights
# model.load_weights("weights.best.hdf5")

adam = optimizers.Adam(lr=0.00001)
model.compile(loss = 'mse', optimizer = adam, metrics=['mse', 'accuracy'])

# Checkpoint best model weights
from keras.callbacks import ModelCheckpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Generate the model
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, \
	validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=2, \
	verbose=1)

# Save the model
model.save('model.h5')
import gc; gc.collect()

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss')
