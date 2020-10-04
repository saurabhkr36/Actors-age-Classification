# Actors-age-Classification
Link to Competition -> https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/#About
Link to Leaderboard -> https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/#LeaderBoard
## Introduction
Facial feature analysis has always been a topic of interest mainly due to its applicability. Deep Learning techniques are now making it possible for face analysis to be not just a dream but a reality.
Dataset->Indian Movie Face database (IMFDB) is a large unconstrained face database consisting of 34512 images of 100 Indian actors collected from more than 100 videos. All the images are manually selected and cropped from the video frames resulting in a high degree of variability interms of scale, pose, expression, illumination, age, resolution, occlusion, and makeup. IMFDB is the first face database that provides a detailed annotation of every image in terms of age, pose, gender, expression and type of occlusion that may help other face related applications.
It consists of a total of 26742 images with 19906 images in train and 6636 images in test.
## Objective
The objective is to classify the age of a person from his or her facial attributes as Young, Middle and Old.
## Step-wise Approach
### Image Preprocessing
The images are read using scipy and processed for resizing, squeezing and normalizing as preprocessing step.
#### Resizing
All the images are resized to 32 by 32 using OpenCV for the model.
#### Squeezing
Squeezing is used to remove single-dimensional entries from the shape of the image array and prepare for further processing. It is a feature in numpy.
#### Normalizing
Normalization is used to standardize all feature vectors in the same range. In this dataset, after squeezing, the data features are normalized by dividing each feature vectors by 255.
### Model Training
The model consists of multiple Conv2d layers, batch normalization layers, MaxPooling2D layers, dropout layers, flatten layers and dense layers with input dimension as (32,32,3).
Convolutional neural networks apply a filter to an input to create a feature map that summarizes the presence of detected features in the input.
#### Layers
There are 4 convolutional 2d layers with gradually increasing number of filters and padding is 'same' inorder to keep the dimensions same for each layer. The first two convolutional layers are followed by batch normalization layer and dropout layer with dropout probability 0.25. The last two convolutional layers are followed by maxpoolings and dropouts with 0.5 probabilities. 
There is a flattening and dropout layer with probability 0.5 followed by 3 dense layers with 512, 128 and finally 3 nodes as output. 
->Convolutional layer: It will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
->Maxpooling layer: It is used to extract the maximum features in a convolutional layer and is used in many examples.
->Padding: In this model, I've used same padding which pads the image such that output size is the same as the input size.
->Flattening:Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. And it is connected to the final classification model, which is called a fully-connected layer.
->Dense layer:A dense layer is just a regular layer of neurons in a neural network. Each neuron recieves input from all the neurons in the previous layer, thus densely connected. The layer has a weight matrix W, a bias vector b, and the activations of previous layer a.
#### Optimizer
Adam optimization algorithm is used for this model and categorical crossentropy loss function is used. The metric used is accuracy. 
#### Hyperparameter tuning (Overfitting/Underfitting)
->BatchNormalization: Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
->Dropout: A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during training. This is called dropout and offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks of all kinds.
->Activation function: 
1) Relu: RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged.
2) Sigmoid: It is used for models where we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice. 
## End Notes
The first step is to reshape and apply normalization to make the feature space best for use. 
In this model, we've used multiple convolutional, maxpooling, batch normalization, dropout and Fully connected layers with hyperparameter tuning as batchnormalization, dropout and activation functions to get the best prediction classifications.
