import csv
import os
import cv2
import numpy as np
import utils

def getMeasurements(samples):
	measurements, images = utils.preprocess(samples)
	return measurements



# Read the csv file for image path
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		samples.append(line)

#print (np.array(samples)[:,3])

measurement_data = getMeasurements(samples)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#print (np.array(measurement_data))

# Plot measurements
plt.hist(measurement_data)
plt.xticks((np.arange(-1.0, 1.0, 0.2)))
plt.savefig('measurement histogram')
