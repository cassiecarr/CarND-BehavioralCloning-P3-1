# **Behavioral Cloning - Project Writeup**
---

**Project Goals:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./examples/smooth_curve_driving.jpg "Smooth Curve Driving"
[image3]: ./examples/recovery_driving.jpg "Recovery Driving"
[image4]: ./examples/unflipped_image.jpg "Unflipped"
[image5]: ./examples/flipped_image.jpg "Flipped"
[image6]: ./examples/Histogram_OriginalDataset.png "Original Dataset"
[image7]: ./examples/Histogram_ModifiedDataset.png "Modified Dataset"
[image8]: ./examples/cnn-architecture.png "NVIDIA Architecture" 


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [utils.py](utils.py) containing the preprocessing steps for the dataset
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h) containing a trained convolution neural network 
* writeup.md (this file) explains the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip) and my [drive.py](drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the [NVIDIA network architecture for self driving cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) (model.py lines 49-66) 

The data is also normalized in the model using a Keras lambda layer (code line 51). See below image for layers used in the NVIDIA model:

![alt text][image8]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 53). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but a learning rate was manually selected at a reduced rate. Originally the model validation loss stopped improving after the first epoch, so a reduced learning rate was added to fine tune the stopping point.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving along smooth curves, and driving the track in the other direction. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose something appropriate for the application. I decided to use the convolutional neural network architecture designed by NVIDIA because it was used in a similar application for a self driving car. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My original dataset had low training and validation loss, with the validation loss sometimes being lower than the training loss. However, when I ran the model in the simulator, the car went off the track at the first curve. 

I also noticed that on the second epoch, the model stopped improving. In order to ensure no overfitting, dropout layers were added after each of the (5) convolutional 2D layers. In addition, the learning rate was manually set to a reduced rate in order to better fine tune the stopping point. Unfortunatly, after running this model in the simulator, the car still went off the track at the first curve. 

Then I added additional pre-processing, including resizing the image to the NVIDIA model input size of 66x200, changing the colorspace to HSV, and cropping the top and bottom portions of the image that included the hood of the car and the sky and trees. 

After this, I looked at histograms of the dataset and found the data was highly skewed to low steering angles. Code was added to the pre-processing to more evenly represent the data. In addition augmented images were added for each image flipped horizontally. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle went outside the lines. In order to improve this, more training data was collected in these areas. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. [See my video](https://www.youtube.com/embed/KLlb4TpVvCY)!

#### 2. Final Model Architecture

The final model architecture (model.py lines 49-66) consisted of a convolution neural network with the following layers and layer sizes, see image diagram above:
* Normalized input 66x200x3
* (3) Convolutional 2D layers with a 5x5 filter (dropout between)
* (2) Convolutional 2D layers with a 3x3 filter (dropout between)
* Flatten
* (3) Fully connected layers 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first utilized the Udacity sample dataset. Here is an example image of good driving behavior:

![alt text][image1]

I then added additional training data myself. The track was driven backwards, additional curves were recorded, and recovery data was added. For the recovery data, snippets were recorded showing the car going from the outer edge of the track back to the center.

![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images horizontally to even out the left and right turns for a more representative dataset. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had X number of data points. I then preprocessed this data by:
* Changing the colorspace to HSV
* Cropping the image of the hood of the car at the bottom and sky / trees at the top
* Resizing the image to 66x200 for the NVIDIA CNN Network Architecture

In addition, to develop a more represented datset, I removed every other image that had a steering angle lower than 0.1. I also added copies of the data for images with higher steering angles. See the histograms below to see how the dataset was more representative before and after applying this technique:

![alt text][image6]
![alt text][image7]

I finally randomly shuffled the data set into training and validation sets. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Originally, the atom optimizer was used, where the learning rate was set automatically. However, after 1 epoch, the model would stop improving. In order to reduce overfitting and improve the stopping point, a smaller learning rate was selected. With this learning rate, 5 epochs were used. 

