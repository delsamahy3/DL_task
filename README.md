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

## Test Results

## Choice of Metrics

## Overall Model Receptive Field

## FLOPs and MACs

## Computationally Expensive Layers

## Optimization

Quantization is a technique that converts 32-bit floating numbers in the model parameters to 8-bit integers. With quantization, the model size and memory footprint can be reduced to 1/4 of its original size, and the inference can be made about 2-4 times faster, while the accuracy stays about the same.
