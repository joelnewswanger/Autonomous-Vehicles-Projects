# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./distribution.jpg "Visualization"
[image2]: .signs/1.jpeg "1"
[image3]: .signs/2.jpeg "2"
[image4]: .signs/3.jpeg "3"
[image5]: .signs/4.jpeg "4"
[image6]: .signs/5.jp3g "5"

---

### Data Set Summary & Exploration

I used the python len() function and the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 24 X 24
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between the 43 classes.

![Distribution of data between classes][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I one-hot encoded the label data for all the data to allow for using softmax outputs for prediction.

Next, I converted the images to grayscale, because this is more accurate and efficient.

As a last step, I centered and normalized the pixel values which is necessary for a model to function.

I decided to use a keras imagedatagenerator() to implement image augmentation on the training data to help force the model to learn valuable features rather then memorizing images.

The data generator implements shear, zoom and rotational modifications to the data as it 'flows' batches to the model during training. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x8  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x8 		    		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x16	|
| RELU                  |                                               |
| Max pooling           | outputs 8x8x16                                |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x32      |
| RELU                  |                                               |
| Flatten               | Outputs 2048                                  |
| Fully connected		| 64 units     									|
| Dropout               | 25%                                           |
| Fully connected       | 64 units                                      |
| Dropout               | 25%
| Fully connected       | 43 units
| Softmax				|            									|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I fit the model to the data generator using a batch size of 20, for 20 epochs, validating against the unaugmented validation data, and using a callback to a ModelCheckpoint set to save best only.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.32%
* validation set accuracy of 95.92
* test set accuracy of 95.02

An iterative approach:
* The first arcitecture that was tried was just the three conv layers with the one dense 43 layer, because this has been a successful starting place in the past.
* The highest accuracy I could get from just the conv layers was in the 80s.
* More dense layers were added, with the result of increasing the accuracy into the 90% range.
* This model had more overfitting problems even with the data augmentation.
* 25% Dropout layers were added between the Dense layers, which solved this problem.
* I experimented with many different numbers of filters for the conv layers, and numbers of nodes in the dense layers, to find the most optimal arrangement.
* The convolutional network architecture is suitable for this problem because convolutional layers preserve the spatial relationships of the data which is important for features of an image. Three layers is optimal because it allows for low medium and high level features, and the dense layers help to make the right decisions based on the information provided by the conv network.
