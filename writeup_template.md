# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/image_distribution_analysis.png "Visualization"
[image2]: ./examples/gray_image.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_online/image_1.jpg "Traffic Sign 1"
[image5]: ./test_online/image_2.jpg "Traffic Sign 2"
[image6]: ./test_online/image_3.jpg "Traffic Sign 3"
[image7]: ./test_online/image_4.jpg "Traffic Sign 4"
[image8]: ./test_online/image_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/govinsprabhu/Traffic_signs_detector/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs dataset:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will introduce little bit color invariant. I.e, our image, even it is coming with a slighter change in color (due to the camera or whatever), our model will be able to predict 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As the last step, I normalized the image data because normalized images have zero mean and small variance (generally 1). Dataset  which is having the small constant mean tend to perform well because of gradient descent can converge faster in these datasets. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have used the LeNet convolutional network but added a lot of modification.
My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Gray Image                        | 
| Convolution 5x5         | 1x1 stride, same padding, outputs 28x28x8     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x8                 |
| Convolution 5x5        | 1x1 stride, same padding, outputs 10x10x32     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x32                 |
| Convolution 3x3        | 1x1 stride, same padding, outputs 3x3x64     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride, 'SAME' padding, outputs 3x3x64                |
| Flattening    | output 576                                            |
| Fully connected        | output  120                                           |
| RELU                    |                                                |
| Dropout                    |            Keep prob of 0.6                                    |
| Fully connected        | output  84                                           |
| RELU                    |                                                |
| Dropout                    |            Keep prob of 0.6                                    |
| Fully connected        | output  43                                           |
| SoftMax                |                                                |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Below are the description of my model training hyperparameters and optimizers

* I have used `softmax_cross_entropy_with_logits` function from tf, then take mean of it to get the loss
* Used adam optmizers, with learning rate 0.0015
* Epochs is 50 and batch_size = 64

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.943
* test set accuracy of 0.93

I have initially chosen LeNet but modified it iteratively for this particular dataset. I have given the details below

* What was the first architecture that was tried and why was it chosen?
   * I have started with LeNet architecture because I have already done MNIST data classification on the same. So I was getting around 89% accuracy
* What were some problems with the initial architecture?
   * The architecture LeNet was designed for MNIST dataset, which has only 10 images to classify. But for traffic Sign, we need classify 43 images. So the number of layers and number of filters(weights) were insufficient
   
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * As I mentioned earlier, the LeNet was architecture was insufficient for detecting the 43 traffic signs. So I have added one more convolutional layer and also increased the number of filters in each layers.
    * Now, from training and validation error, I was come to know network was overfitting on the training data. So I have introduced the dropout with keep_prob = 0.6. in fully connected layers.
    * I have used 50 epochs, which is greater than the one which I used in LeNet

* Which parameters were tuned? How were they adjusted and why?
    * the Main parameter I tried to tune was learning rate. Started from 0.001, I have tried different learning rates like 0.01, 0.0001, 0.005, 0.002, etc, to see the effect of the learning rate having on the model training. Higher learning rate most of the time was resulting underfitting, so I reduced it. Lower learning rates were too slow while training. After trying a lot of combinations, I have settled to 0.0015, which was giving me a good result
    * Batch size I have reduced from 128 to 64, which was giving a slightly better result
    * No of epochs increased to 50
    * Increased the number of filters in each convolution layer for underfitting
    * Added one more convolution layer at the end of convolution layers for the same reason (underfitting)
    * Tried different keep probs for drop out. Settled with 0.6. Added dropouts in the fully connected layers only.
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * I have started with LeNetModel, but modified it significantly for detecting 43 different traffic signals. I have chosen Convolution layers because of following reasons
      * Unlike Multi-Layer Perceptron (MLP) model, CNN architecture does not have dedicated weights, weights (filter) is being shared across the image patches. This reduces the overfitting significantly. 
      * Number of the parameter to train would have been significantly high if I have used the same number of weights in MLP due to many to many connection dedicated weight structure of MLP
      * MaxPooling in CNN helps to reduce the size, without adding any parameter for training
      * Convolution and max-pooling architecture introduces location, translation, rotation invariance to the model.
    * I have used dropout with keep-prob of 0.6, which has helped me to reduce the difference between training and validation error. If dropout was not there, I was getting 99.5 training accuracy. I was able to reduce it to 98 % with dropout, without reducing the validation accuracy


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. All of them having different sizes. Some of them adding text in the bottom. Some of them comes with hyper link also.:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it had a green background. I selected this one to check how the model is performing if the background is not uniform, it has very low resolution
The second one had a blue background and taken from below. It had a triangular border. 
The third one was having a white background and round figure. The sign was indicated by white in the blue background
The fourth image was not a square, but a rectangle, indicating speed limit 60 km/ph, I selected this because I wanted to know whether it will be able to classify this numbers properly. 
The last one was a circular sign with a green and white background. Was one among the most difficult one to classify. Due to because of non-uniform background and its sign (roundabout mandatory) was getting confused with other sign.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)           | Speed limit (30km/h)                                       | 
| Right-of-way at the next intersection                 | Right-of-way at the next intersection                                         |
| Keep right                    | Keep right                                            |
| Speed limit (60km/h)              | Speed limit (60km/h)                                     |
| Roundabout mandatory        | Priority Road                                  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. For the test set, we got an accuracy of 93%. If we increase the number of images from the web, accuracy will reach approximately to that of the test set. 

#### 3. Describe how certain the model is when predicting each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For the first image, the model is almost 100% sure that this is a speed limit of 30km/h (probability of 0.99), and the image does contain a stop sign. The top five softmax probabilities were

| Probability                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)           | 0.99                                       | 
| Speed limit (20km/h)                 | 0.000006                                         |
| Speed limit (50km/h)                    | negligible                                            |
| Speed limit (70km/h)              | negligible                                     |
| Turn right ahead        | negligible                                  |


For the second image (Right-of-way at the next intersection), again model was almost 100% sure, it was the same.   

| Probability                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection           | 1                                       | 
| General caution                 | 2.7e-28                                         |
| Double curve                    | negligible                                            |
| Speed limit (20km/h)              | negligible                                     |
| Speed limit (30km/h)        | negligible                                  |

Again for the third image (Keep right), the model was almost 100% sure, it was the same.   

| Probability                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection           | 1                                       | 
| Speed limit (20km/h)                 | 0                                         |
| Speed limit (30km/h)                    | 0                                            |
| Speed limit (50km/h)              | 0                                     |
| Speed limit (60km/h)        | 0                                  |

For the fourth image, the model was almost sure it was Speed limit (60km/h) (0.6)

| Probability                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)          | 0.6                                       | 
| Speed limit (80km/h)                 | .23640                                         |
| Speed limit (50km/h)                    | .0839537531                                            |
| Keep right              | .00192461314                                     |
| Wild animals crossing        | Negligible                                  |

For 5th image, the model was not able to predict its right value. Right answer Roundabout mandatory, which came on the third spot with negligible probability

| Probability                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Priority road          | 0.99                                       | 
| Right-of-way at the next intersection                 | 3.48782305e-05                                         |
| Roundabout mandatory                    | 1.81179348e-05                                            |
| Beware of ice/snow              | .00192461314                                     |
| Speed limit (100km/h)        | Negligible                                  |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


