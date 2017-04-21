import csv
import os
import cv2
import numpy as np

def preprocess(samples):
	images = []
	measurements = []
	correction = 0.2
	count_zero_measurement = 0

	for batch_sample in samples:
		# Remove every other zero
		if abs(float(batch_sample[3])) < 0.1:
			count_zero_measurement += 1
		else:
			count_zero_measurement = 0
		if count_zero_measurement > 1:
			if count_zero_measurement > 2:
				count_zero_measurement = 0
			continue

		# Apply preprocessing to left, right and center images
		for i in range (3):

			# Read image
			source_path = batch_sample[i]
			filename = source_path.split('/')[-1]
			current_path = 'data/IMG/' + filename
			image = cv2.imread(current_path)

			# Apply resize
			image = cv2.resize(image, (235,118))

			# Apply crop
			image = image[34:100, 17:217]

			# Convert to HSV color space
			image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

			# Add image
			images.append(image)

			# Get steering angle
			measurement = float(batch_sample[3])

			# If left image, apply positive correction
			if i == 1 and abs(measurement) > 0.2: 
				measurement = measurement + correction

			# If right image, apply negative correction
			if i == 2 and abs(measurement) > 0.2: 
				measurement = measurement - correction

			# Add steering angle
			measurements.append(measurement)

			# Add horizontally flipped image to dataset, adjust steering angle
			augmented_image = np.fliplr(image)
			images.append(augmented_image)
			augmented_measurement = -measurement
			measurements.append(augmented_measurement)

			# Add additional images when steering angle is greater than 0.4
			for i in range(4):
				if (abs(measurement)) > 0.4:
					images.append(image)
					measurements.append(measurement)
					images.append(augmented_image)
					measurements.append(augmented_measurement)
				# if (abs(measurement)) > 0.55:
				# 	images.append(image)
				# 	measurements.append(measurement)
				# 	images.append(augmented_image)
				# 	measurements.append(augmented_measurement)
				if (abs(measurement)) > 0.7:
					images.append(image)
					measurements.append(measurement)
					images.append(augmented_image)
					measurements.append(augmented_measurement)

	return measurements, images