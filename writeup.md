# **Traffic Sign Recognition** 

[//]: # (Image References)

[image0]: ./capture-with-labels.png "Web images"
[image1]: ./data-vis.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/aaliani/CarND-Traffic-Sign-Classifier-Project-P2/blob/master/P2_Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 118466
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot showing the number of samples per class in the dataset.

![number of samples per class][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The original dataset consisted of 34799 traning, 12630 testing and 4410 validation images of 3 32x32 RGB channels. The training set consisted of 43 classes with number of samples within each class ranging from 180 to 2010. That is the differnce of approximately 11 times between class with least representation against the class with most representation in the dataset.

As a first step, I decided to convert the images to grayscale because it reduces the size of image 3 folds, hence significantly lower training time on same number of samples. Less processing woud also be vital if the model is applied on any realtime applications where latency is critical. Moreover, traffic signs are just as recognizable in grayscale as in color, except traffic signals sign, beacause it is just the shape of the sign where all the meaningful infomation lies. Color and texture add little information of use to be considered against the additional processing time colored images require for processing.

Next step was to increase the amount of training data. More data a neural network has, better it trains. In order to increase the data, I used the original data and applied various transformations to it to create new samples. My algorithm loops through the original dataset and randomly decide if it wants to create further samples from that one. If it chooses to augment that sample, it then creates between 2 and 5, number again randomly selected, rotated images to between -30 and 30 degrees out of it, between 2 and 5 images that have shearing transformation of factor between -25 and 25 applied to it, and between 2 and 6 images that have perspective distortion of randomly generated tranformation matrix for each applied to it. This generates on average 12 new images out of the original sample if the algorithm decides to use that original sample. To also address the disparity between representation of each class within the dataset, I decided to assign a higher probability of generating more samples from an image if it belonged to a class with less representation and vice versa. This probability was defined based on the normal distribution of the amount of samples for each class. This approach reduced the disparity from 11.2x in the original training set to 3.8x in the augmented training set with 1500 samples for class with least representation and 5674 samples for one with largest.

I then also converted the images in test and validation sets to grayscale, without any augmentation however.

Controvertially though, I decided not to normalize the image data. My reasoning is that normalization is aimed at confining the values within a specifc range to make the dataset is homogeneous so that sensible weight values can work for all data points within the set. Image data, however, is within a set range of 0 and 255, so it is certainly homogeneous. I understand, though, that is not zero mean which is also one reason to do normalization. But in my observation, the model trained very well on the non-normalized data with better validation accuracy and convergence than when I did normalize it. Although the value of learning rate might have something to do with that too. Perhaps it would have performed better had I reduced the learning by a factor of 255 as well since that is also the difference between the scale of normalized vs non-normalized values. I also observed that non-normalized data was certainly more effiecient and quicker while training. It was certainly quicker for processing for augmentation etc since the datatype of non-normalized data is uint8, which is of size one byte, vs normalized dataset of type float32, which is of size 4, hence the size of normalized data is 4x that of non-normalized. So the efficiency can be explained by the reduction size of the dataset, and hence significant performance improvement in loading. But I do understand, of course, that when fed to the tensorflow graph the data is converted to tf.float32 type anyways, which suggests there might be some latency tradeoffs when converting from np.float32 to tf.float32 vs np.uin8 to tf.float32. Could be studied further. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU					|	
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|	
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				|
| Flatten   | outputs 2304 |
| Fully connected		| outputs 1024       									|
| RELU					|	
| Fully connected		| outputs 512       									|
| RELU					|	
| Fully connected		| outputs 128       									|
| RELU					|	
| Fully connected		| outputs 128       									|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. I used the learning rate of 0.001 initially but realized during training that the accuracy seems to have plateued. And sure enough, when I used the learning rate of 0.0005, the model started with lower accuracy as expected but after few epochs produnced much better accuracy than with the learning rate of 0.001. To test the  performance of different hyperparamaeter configurations, I tranined to up to 20 epocs first. With the learning rate of 0.005, it already acheived the desired accuracy of greater than 93% within 20 epochs. But I tranined my final model to 200 epochs, which resulted in 98% accuracy in the end. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 98%
* test set accuracy of 94.5%

I started with simply the LeNet architecture, hoping that with the data augmentation and preprocessing I applied to the dataset could already give the 93% target accuracy. But the architecture indeed still acheived the maximum 89.5% accuracy. Upon traning over many epochs, the accuracy also started to decrease significantly. So I had to change the architecture.

After doing some research, I applied the approach of [Microsoft's ResNet](https://arxiv.org/pdf/1512.03385.pdf) and added Residual Blocks to the LeNet architecture to tackle the problem of loss divergence. Instead of two convolutional layers flowing into fully connected layers in LeNet, I made two Residual Blocks with two convolutional layers each that are passed onto fully connected layers. Outputs of the first convolutional layer in each bloack are appended to the output of the second output layer in each block. Second convolutional layer is padded to have the same output shape as the first in each block so that the outputs can be concatenated. I removed max pooling between the layers within each block so that the output shape of can be as close as possible. But I also decided to avoid the max pooling between the two residual blocks in order to preserve the number of features flowing into the second block. So I just apply max pooling once after the second residual block before flattening the output and passing it onto the fully connected layers. These layers stayed the same as in LeNet, except the input and output shapes needed to be adjusted according to the data flow in rest of the architecture. The final layer outputs 43 logits, no softmax is applied.     

This small adjustment, inspired by ResNet, did indeed do the trick. The loss divergence disappeared and model acheived the eventual accuracy of 98% on the validation set.

I did not do any further ammendments to the architecture. However, it can certainly be improved. The test accuracy of 94.5%, albeit acceptable for this project, suggests that the model is probably over fitted. This suspicion is further boosted by the mere 83.3% accuracy it acheived with the images from the web. So to improve the model, at least one dropout layer needs to be added. The net can be made deeper to enhance the accuracy as well. 

### Test a Model on New Images

#### 1. Choose few German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are German traffic signs that I found on the web:

![Traffic Signs from web and their predicted labels][image0]

Some of then are tilted or taken from different perspectives. Such were chosen to see how the model performs on them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The results of the predictions can be seen in the image above. The model incorrectly classifies two out of the twelve images. Both of them regarding the speed limit (hence this model would be extremely dangerous in the real world!) It classifies one 50 km/h speed limit sign as 20 km/h, and one 120 km/h sign as 20 km/h. But it does classify one 70 km/h speed limit sign correctly.

The model with this result gives an accuracy of 83.3%. This is acceptable for this project, but inconsistent with the validation accuracy of 98% that the model trained on. With 94.5% accuracy on the test set, this result of the web images can be considered passable outlier.

