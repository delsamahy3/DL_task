# Part 2
# Requirement: Building a classifier for clothing articles

## Classification problem 

Classification is the process of recognition, understanding, and grouping of objects into preset categories a.k.a “sub-populations.” With the help of pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories. Classification algorithms utilize input training data for the purpose of predicting the probability that the data that follows will fall into one of the predetermined categories. 

## Dataset used: Fashion-MNIST

For the purpose of this project, I have used a simple, moderate-sized dataset to train the classifier, visualize and test the training results.

Fashion-MNIST is a dataset of Zalando’s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Data labels in the table below.

<img width="250" alt="Screen Shot 2022-12-26 at 6 12 27 PM" src="https://user-images.githubusercontent.com/114371118/209566670-2836714a-5cb9-4854-bca6-260eafb94b92.png">

Another available dataset is DeepFashion dataset: DeepFashion: Powering Robust Clothes Recognition and Retrieval With Rich Annotations, with 289,222 examples. This would require longer training times and therefore I settled to using FashionMNIST dataset.

## Data Handling and Loading

Using PyTorch utils, we can load the data from the disk using pandas read_csv() method then inherit Dataset class to build my own dataset class. However, I chose to download FashionMNIST class from torchvision module.

Training and testing sets (60,000 and 10,000 respectively) are then made into batches of size 100, and data is transformed into tensors that has a range from 0 to 1.

The value of each pixel in the image data is an integer in the range [0,255]. We need to normalize these values to the range [0,1].

## Proposed Solutions

## Solution of choice

Creating a model class (FashionCNN)
which inherits nn.Module class that is a super class for all the neural networks in Pytorch.

It consists of the following layers:

Two Sequential layers each consisting of:


*   Convolution layer that has kernel size of 3 * 3, padding = 1 (zero_padding) in 1st layer and padding = 0 in second one. Stride of 1 in both layer.
*   Batch Normalization layer.

*   Acitvation function: ReLU.

*   Max Pooling layer with kernel size of 2 * 2 and stride 2.

*   Flatten out the output for fcn.

*   2 Fully connected layer with different in/out features.

*   1 Dropout layer that has class probability p = 0.25.

All the functionaltiy is given in forward method that defines the forward pass of CNN.

The input image is changing in a following way:

**First Convulation layer** : input: 28 * 28 * 3, output: 28 * 28 * 32

**First Max Pooling layer** : input: 28 * 28 * 32, output: 14 * 14 * 32

**Second Conv layer** : input : 14 * 14 * 32, output: 12 * 12 * 64

**Second Max Pooling layer** : 12 * 12 * 64, output: 6 * 6 * 64

Final fully connected layer has 10 output features for 10 types of clothes.

<img width="792" alt="Screen Shot 2022-12-26 at 10 10 14 PM" src="https://user-images.githubusercontent.com/114371118/209580585-cc3295e6-78c0-480f-a544-fcb8ab231fff.png">

<img width="565" alt="Screen Shot 2022-12-26 at 10 46 28 PM" src="https://user-images.githubusercontent.com/114371118/209582431-b17a7efb-23de-42ff-81fd-fe4e5d9bdb09.png">


## Choice of Metrics

For a Loss function, I am using CrossEntropyLoss as it is suitable for multi-label classification. We compute the binary cross-entropy for each class separately and then sum them up for the complete loss.

Using Adam algorithm for optimization purpose as it has faster computation time, and requires fewer parameters for tuning.

Classification accuracy metric is used together with loss to judge the classifier progress.

Precision, recall and a confusion matrix analysis is done to be able to visualize per class performance.

## Test Results

Over the course of 5 epochs, the testing accuracy reaches 90.29000091552734% and loss drops down from 0.5451227426528931 to 0.1914568841457367, which is considered a good accuracy with respect to the model depth, dataset size and type of images (low features)

<img width="642" alt="Screen Shot 2022-12-26 at 10 10 56 PM" src="https://user-images.githubusercontent.com/114371118/209580631-dbaad802-1ebf-42d4-9f12-798e4b2664bf.png">

<img width="383" alt="Screen Shot 2022-12-26 at 10 33 36 PM" src="https://user-images.githubusercontent.com/114371118/209581760-7cf9c6dc-0527-4ea1-8633-13cd2e9cdc77.png">

Per-label accuracies are shown

<img width="311" alt="Screen Shot 2022-12-26 at 10 33 57 PM" src="https://user-images.githubusercontent.com/114371118/209581773-3daaeb87-1a1f-4d68-8b48-55f406f7ea95.png">

Precision, recall and F1-score

<img width="512" alt="Screen Shot 2022-12-26 at 10 34 13 PM" src="https://user-images.githubusercontent.com/114371118/209581785-ea3f4626-4fe2-4ec4-a2a9-606a4e20451c.png">

Normalized Confusion Matrix

<img width="311" alt="Screen Shot 2022-12-26 at 10 34 49 PM" src="https://user-images.githubusercontent.com/114371118/209581824-c690672b-80e7-42b7-9acb-891abf987158.png">


## Overall Model Receptive Field

## FLOPs and MACs

## Computationally Expensive Layers

## Optimization

Quantization is a technique that converts 32-bit floating numbers in the model parameters to 8-bit integers. With quantization, the model size and memory footprint can be reduced to 1/4 of its original size, and the inference can be made about 2-4 times faster, while the accuracy stays about the same. Here I used Dynamic Quantization, which converts all the weights in a model from 32-bit floating numbers to 8-bit integers but doesn’t convert the activations to int8 till just before performing the computation on the activations. This is done through using: torch.quantization.quantize_dynamic.

Quantization Accuracy is at : 90.16999816894531% compared with 90.29000091552734% from the un-quantized model at epoch 5.
