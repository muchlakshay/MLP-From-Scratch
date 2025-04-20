# MLP-From-Scratch
A C++ implementation of a Multilayer Perceptron (MLP) neural network using Eigen, supporting multiple activation and loss functions, mini-batch gradient descent, and backpropagation for training.

## Features

- **Customizable Architecture**: You can define the number of layers and the number of neurons per layer.
- **Multiple Activation Functions**: Supports ReLU, Sigmoid, Tanh, Leaky Relu and Softmax for non-linearity.
- **Loss Functions**: Supports Mean Squared Error (MSE), Binary Cross Entropy and Cross-Entropy loss functions for regression and classification tasks.
- **Backpropagation**: Implements the backpropagation algorithm for training the network.
- **Mini-batch Gradient Descent**: Optimizes the network using mini-batch gradient descent, allowing for more efficient training on large datasets.
- **Modular Design**: The code is organized in a way that allows for easy extension and modification (e.g., adding new activation functions, loss functions, etc.).

## Prerequisites

- C++11 or later
- [Eigen](https://eigen.tuxfamily.org/) (header-only C++ library for linear algebra)

## Getting Started

### Clone the repository

```bash
git clone https://github.com/muchlakshay/MLP-From-Scratch.git
cd MLP-From-Scratch
```
### Compile and run
To compile and run the project, use a C++ compiler that supports C++11 or later (e.g., g++, clang++):

```bash
g++  main.cpp NeuralNetwork.cpp -o NN -I/path/to/eigen
./NN
```
Replace /path/to/eigen with the actual path where Eigen is installed or located.

## Training
- The network can be trained on your own dataset. You can define the input data (features) and target data (labels) in the form of Eigen MatrixXd, specify the number of epochs, mini-batch size, and learning rate.
```cpp
NeuralNetwork mlp({784, 16, 16, 10}, {"relu", "relu", "softmax"});  // 784 input neurons, 2 hidden layer with 16 neurons and 10 output neuron
mlp.learn(input_data, labels, 1000, 32, 0.01, "cross_entropy_loss", true); // input data, labels, batch size, learning rate, loss function, verbose (true by default)
```



