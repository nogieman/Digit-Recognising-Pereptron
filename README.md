# **Handcrafted Digit Recognizer with Numpy**

This project is a custom implementation of a digit recognizer using only NumPy. It includes algorithms for gradient descent, backpropagation, and activation functions like ReLU and softmax. The project focuses on building a neural network from scratch in Python, without relying on external libraries for deep learning.

## Dataset
This model uses the MNIST Dataset for training. It contains grayscale images of handwritten digits (0-9) and their corresponding labels.

## Implementation and Usage
This project can be implemented locally by mentioning the dataset's exact location. 
The project consists of several code cells that perform the following tasks:

1. **Data Preprocessing**: Reading and preprocessing the dataset, including shuffling and normalization.
2. **Initialization**: Initializing the parameters (weights and biases) of the neural network.
3. **Forward Propagation**: Implementing forward propagation to compute the outputs of the neural network.
4. **Backward Propagation**: Implementing backward propagation to compute the gradients of the parameters.
5. **Gradient Descent**: Implementing gradient descent to update the parameters based on the gradients.
6. **Prediction and Testing**: Making predictions using the trained model and testing the accuracy on the training set.

## Dependencies

- NumPy
- Pandas
- Matplotlib
