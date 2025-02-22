# MNISTNeuralNetwork
A multi-layer (3 layer) fully connected,  feed forward network consisting of sigmoidal neurons, trained with stochastic gradient descent and back  propagation.

# Documentation

## Overview
The `MNISTNeuralNetwork` program is a simple artificial neural network implemented in Java to classify handwritten digits from the MNIST dataset. It includes functionalities to train the network, load pre-trained weights, test its accuracy, and visualize misclassified images.

## Features
- Train the neural network using mini-batches and backpropagation.
- Load and save trained weights to a file.
- Display network accuracy on both training and testing datasets.
- Visualize misclassified testing images.
- User-interactive console menu for easy navigation.

## Neural Network Architecture
- **Input Layer:** 784 neurons (corresponding to 28x28 pixel grayscale images).
- **Hidden Layer:** 100 neurons.
- **Output Layer:** 10 neurons (one for each digit from 0 to 9).
- **Activation Function:** ReLU for hidden layers, Softmax for the output layer.
- **Learning Rate:** 0.01.
- **Mini-batch Size:** 32.
- **Epochs:** 50.

## Program Execution
Upon running the program, the user is prompted to choose between:
1. Training the network from scratch.
2. Loading a pre-trained network.
3. Exiting the program.

After choosing an option, additional functionalities become available, including:
- Retraining the network.
- Displaying accuracy.
- Running the network on test data.
- Viewing misclassified images.
- Saving the network state.

## Core Functions
### `feedforward(double[] input)`
Computes the output of the network given an input image.

### `backpropagation(double[] input, double[] target)`
Performs the backpropagation algorithm to adjust weights and biases based on error.

### `updateweightsandbiases(double[] input)`
Updates network weights and biases after computing errors.

### `train(String trainingDataPath, int trainingSize)`
Trains the network using the specified dataset.

### `accuracy(String dataPath, int dataSize)`
Calculates the accuracy of the network on a given dataset.

### `saveWeightsToFile(String filename)`
Saves the trained weights and biases to a file for future use.

### `loadWeightsFromFile(String filename)`
Loads saved weights and biases from a file.

### `runOnTestDataAndDisplay(String dataPath, int dataSize)`
Runs the network on testing data and displays the results, including misclassified images.

## File Handling
- Training and testing data should be in CSV format.
- Weight sets are stored in `weightset.csv`.

## Dependencies
- Java Standard Library (java.io, java.util)

## Usage Example
```sh
javac MNISTNeuralNetwork.java
java MNISTNeuralNetwork
```
Follow on-screen prompts to train, load, and test the network.
