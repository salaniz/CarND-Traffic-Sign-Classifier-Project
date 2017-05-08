# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dist]: ./figures/dist.png "Data Set Distribution of Classes"
[examplesigns]: ./figures/example_signs.png "Example Signs Organized by Class"
[examplesigns2]: ./figures/example_signs2.png "Example Signs Organized by Class"
[newdist]: ./figures/new_dist.png "Training Set Distribution of Classes after Augmentation"
[augmentationexamples]: ./figures/augmentation_examples.png "Examples of Data Augmentation"
[grayscaleexamples]: ./figures/grayscale_examples.png "Examples of Grayscaling"
[ownsignsresults]: ./figures/own_signs_results.png "Results of Gathered Traffic Signs"
[examplefeaturevis]: ./figures/example_feature_vis.png "Example Image for Feature Visualization"
[featurevis1]: ./figures/feature_vis1.png "Feature Visualization"
[featurevis2]: ./figures/feature_vis2.png "Feature Visualization"
[featurevis3]: ./figures/feature_vis3.png "Feature Visualization"
[featurevis4]: ./figures/feature_vis4.png "Feature Visualization"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated the following summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The four histograms show the data distribution per class in the data set. The distribution is the same accross training, validation and test set. The fourth plot shows the data I gathered on my own and is used in the 3rd part of this project.

![alt text][dist]

Most importantly, the distribution is not uniform, but rather imbalanced towards certain classes. This can make it hard to learn a model that classifies classes with few training samples correctly. Thus, I decided to include a data augmentation step to level out the distribution.

The second plot shows three random samples per class, which mainly help in understanding how the actual data looks like and that classes are assigned correctly. Some images seem to be very dark and hard to recognize by a human so adjusting the contrast might also help.


![alt text][examplesigns]
![alt text][examplesigns2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing can be summerized in three steps:
1. Data augmentation
2. Addition of a grayscaled and histogram equalized channel
3. Normalization

**Data augmentation** is done as suggested by Sermanet et al.[2] Images of the training set are scaled, shifted and rotated to generate new training samples. Hyperparameters are taken from the paper and only used as discrete values: 
* scaling factor [0.89, 0, 1.11], 
* shift by [-2, 0, 2] pixels, 
* rotation of [-15, -10, -5, 0, 5, 10, 15] degrees.

For each individual augmentation, a random configuration of these three parameters is sampled and then applied to generate a new image. The goal is to get a uniform distribution of 2000 samples per class.
Augmentation is done on a per class basis with each individual image being augmented the same amount of times (with some exceptions of augmenting a subset of images one extra time to match the targeted number of class samples).
After augmentation the class distribution in the training set looks like this:

![alt text][newdist]

The following 20 examples images (top) were augmented with a random configuration (bottom):

![alt text][augmentationexamples]

In order to help the network in better distinguishing edges inside dark pictures, the images are **grayscaled** and their **histogram of pixel values equalized** to get a input feature with a good contrast accross the whole range of values. This newly created image is appended to the RGB image as an additional channel so that the new image shape is (32, 32, 4).
Here are 20 examples images (top) and their corresponding grayscaled image (bottom):

![alt text][grayscaleexamples]

Finally, **normalization** of the data is done by substracting the mean and then dividing by the standard deviation on a per image basis.

**What didn't work that well:**  
Sermanet et al.[2] used the image in YUV format instead of RGB. My experiments with both formats showed that RGB seems to work better, so I discarded this preprocessing step.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is identical to the LeNet architecture[1] with added dropout layers and consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16      				|
| Fully connected		| 120 units        								|
| RELU					|												|
| Dropout               | 20% during training							|
| Fully connected		| 84 units        								|
| RELU					|												|
| Dropout               | 20% during training							|
| Fully connected		| 43 units (number of classes), output layer	|
| Softmax				|             									|
 
This network is small enough to be trainable on a laptop and achieves a very good classification performance with the chosen preprocessing steps. Small adjustments didn't improve performance. Using a considerably larger network with dropout could be considered, but needs more time to tune and is not necessarily needed.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using the Adam optimizer, a batch size of 128, 30 epochs, a learning rate of 0.001 and a dropout of 20% at each dropout layer. The hyperparameters were empirically chosen to give the best validation peformance by using the base configuration of the LeNet lab project as a starting point.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.4%
* validation set accuracy of 98.4%
* test set accuracy of 95.7%

I chose a well known architecture:
* What architecture was chosen?  
LeNet network architecture[1].
* Why did you believe it would be relevant to the traffic sign application?  
The course already showed that the LeNet architecture works suprisingly well even though it was designed for digit/letter recognition. Nonetheless, both tasks are image classification tasks, which is why it seems to be a good initial fit for the problem.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
At first, the network saturated quickly on training accuracy with validation accuracy not quite matching up. This can be a sign of overfitting. Adding the two dropout layer after the two hidden fully connected layers considerably improved the network performance on the validation set to a good level so that I sticked to the network. Since the accuracies for validation and test set are over 95% it shows that this is a good architecture for the given problem.
 

### Test a Model on New Images

This visualization includes all results and plots from the tasks of this segment:
![alt text][ownsignsresults]

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose to explore my neighborhood and take pictures of traffic signs myself. From the gathered data, I labeled a total of 50 traffic signs that were used for evaluation (see plot above).
Some of the images were deliberately chosen because they seem to have properties that make them difficult to classify:
* Sign No. 1: This speed limit sign has the abbreviation "km" written on it next to the "20". These additional features could be irritating to the classifier since they do not occur in the training data (as far as I know).
* Sign No. 2: This sign is composed of two different signs. The classifier is not trained on an image like this at all, but still it includes two different signs that are part of the data set's classes.
* Sign No. 3: A slightly different variation of the speed limit 30 sign having the work "ZONE" written underneath.
* Sign No. 4, 22, 34: Sign classes from the data set, but cropped out from a bigger sign containing them. Therefore, they occured in a different surrounding as in the data set.
* Sign No. 8 and 29: The colors of this sign are a bit faded from sun light.
* Sign No. 12, 30, 33 and 50: Small stickers cover parts of this signs that could make it harder to classify.
* Sign No. 31: Pedestrians sign of different variation than in the data set . Below the pedestrian on the sign there is a symbol for the crosswalk that is not found in any of the samples from the data set (to the best of my knowledge).



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly classify 48 of the 50 traffic signs, which gives an accuracy of 96%. This result is in line with the accuracy of original the test set. It is interesting that the model is able to generalize to the difficulties presented with some of the chosen test images. Even signs that haven't been seen before in this specific variation could be classified accurately (cf. sign no. 31).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model shows very high prediction probabilities when correctly predicting the class of the sign even when presented with more challenging images. The two signs that were classified incorrectly, are part of the more challenging test samples.
* Sign no. 2 does not conform the image requirements of the data since it shows two signs instead of one. Unsurprisingly the model is neither very certain nor does it classify any of the two signs correctly. In contrast the two individual sign extracted from the same image (no. 4 and no. 34) are correctly classified.
* Sign no. 3 resembles a speed limit 30 sign in a slightly different variation (with "ZONE" written on it). The model classifies it as speed limit 20 with roughly 2/3 confidence and as speed limit 30 only with 1/3 confidence (according to the softmax values). It is unclear though, if the misclassification is cause by this variation of the sign or by other features.

### (Optional) Visualizing the Neural Network 
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following figures visualize
* the output feature activations (denoted as FeatureMapX) and
* the area of the original image causing this feature activations (denoted as AreaOriginX).

I analyzed the learned features of the two convolutional layers for a given example image. The first layer seems to just learn low level features, since the original image is still visible in the activations. Edges of different angles seem to be detected by the different filters. 
The seconds layer learns more complex features, that are not so easily to clearly make out. However, some still seem to activate on certain edges.

Example Image used for Visualization:  
![alt text][examplefeaturevis]

1st Convolutional Layer:  
![alt text][featurevis1]
![alt text][featurevis3]

2nd Convolutional Layer:  
![alt text][featurevis2]
![alt text][featurevis4]


## References
1. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. In Proceedings of the IEEE, pages 2278-2324, 1998.
2. P. Sermanet and Y. LeCun. Traffic sign recognition with multi-scale convolutional networks. In 2011 International Joint Conference on Neural Networks, pages 2809-2813, 2011.


