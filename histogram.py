import csv
import os
import cv2
import numpy as np
import utils

def getMeasurements(samples):
	measurements, images = utils.preprocess(samples)
	return measurements, images

# Read the csv file for image path
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		samples.append(line)

# Get measurement data
measurement_data, images = getMeasurements(samples)
images = np.array(images)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cv2.imwrite('processed1.png',images[100])
cv2.imwrite('processed2.png',images[1500])
cv2.imwrite('processed3.png',images[3500])


# Plot measurements
plt.hist(measurement_data)
plt.xticks((np.arange(-1.0, 1.0, 0.2)))
plt.title('Steering Angle Histogram')
plt.xlabel('Steering Angle')
plt.ylabel('Occurrences')
plt.savefig('measurement histogram')
