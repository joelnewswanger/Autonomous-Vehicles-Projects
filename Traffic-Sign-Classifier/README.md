# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./distribution.png "Visualization"

[image2]: ./grayscale.jpg "Visualization"

---

### Data Set Summary & Exploration

The data set was comprised of close-up images of german traffic signs.

I used the python len() function and the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 24 X 24
* The number of unique classes/labels in the data set is 43

---
 ##### The distribution of the data between the 43 classes.

![Distribution of data between classes][image1]

### Design and Test a Model Architecture

##### Preprocessing techniques

As a first step, I one-hot encoded the label data for all the data to allow for using softmax outputs for prediction.

Next, I converted the images to grayscale, because this is more accurate and efficient.

![Grayscale][image2]


As a last step, I centered and normalized the pixel values which is necessary for a model to function.

I decided to use a keras imagedatagenerator() to implement image augmentation on the training data to help force the model to learn valuable features rather then memorizing images.

The data generator implements shear, zoom and rotational modifications to the data as it 'flows' batches to the model during training. 


##### My Neural Network consisted of the following layers:

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
 
##### Training:
To train the model, I fit the model to the data generator using a batch size of 20, for 20 epochs, validating against the unaugmented validation data, and using a callback to a ModelCheckpoint set to save best only.

#### An iterative approach:
* The first arcitecture that was tried was just the three conv layers with the one dense 43 layer, because this has been a successful starting place in the past.
* The highest accuracy I could get from just the conv layers was in the 80s.
* More dense layers were added, with the result of increasing the accuracy into the 90% range.
* This model had more overfitting problems even with the data augmentation.
* 25% Dropout layers were added between the Dense layers, which solved this problem.
* I experimented with many different numbers of filters for the conv layers, and numbers of nodes in the dense layers, to find the most optimal arrangement.
* The convolutional network architecture is suitable for this problem because convolutional layers preserve the spatial relationships of the data which is important for features of an image. Three layers is optimal because it allows for low medium and high level features, and the dense layers help to make the right decisions based on the information provided by the conv network.


##### My final model results were:
* training set accuracy of **96.32%**
* validation set accuracy of **95.92**
* test set accuracy of **95.02**

