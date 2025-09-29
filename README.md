# Neural Network From Scratch

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-NumPy-orange.svg)](https://numpy.org/)

This project is a hands-on implementation of a complete neural network framework from the ground up. Using only **Python** and **NumPy**, this project demystifies the inner workings of deep learning by avoiding high-level libraries like TensorFlow or PyTorch.

The primary goal is to build a reusable, object-oriented Neural Network class that showcases a fundamental mastery of core deep learning mechanics, including forward propagation, backpropagation, and gradient descent, using the classic MNIST dataset for training and validation.

---

## Features Implemented

This framework was built by implementing the core components of a neural network layer by layer:

* **Dense Layer:** A fully connected layer with trainable weights and biases.
* **Activation Layer:** An abstract base class for activation functions, with concrete implementations for:
    * **ReLU** (Rectified Linear Unit)
    * **Sigmoid**
    * **Softmax** (for output layer in classification)
* **Loss Functions:** Implemented common loss functions and their derivatives:
    * **Mean Squared Error (MSE)**
    * **Categorical Cross-Entropy**
* **Forward Propagation:** A complete forward pass mechanism that processes input data through the network layers.
* **Backpropagation Algorithm:** A full implementation of the backpropagation algorithm to calculate gradients with respect to the loss.
* **Gradient Descent:** A simple optimizer to update the network's weights and biases based on the calculated gradients.

---

## Tech Stack

* **Core Language:** Python
* **Numerical Computation:** NumPy
* **File Management:** Pandas
* **Environment:** Google Colab

---

## Project Structure

The neural network is built using an object-oriented approach to ensure modularity and reusability.

* **`Layer` (Base Class):** Defines the common interface for all layers (`forward` and `backward` methods).
* **`Dense` (Class):** Inherits from `Layer` and manages the weights, biases, and their gradients for a fully connected layer.
* **`Activation` (Base Class):** Defines the structure for activation functions.
* **`ReLU`, `Sigmoid`, `Softmax` (Classes):** Inherit from `Activation` and implement the specific activation logic and their derivatives.
* **Loss Functions:** Separate functions are defined for `mse` and `categorical_crossentropy`, along with their respective derivative functions.

This structure allows for easily constructing different network architectures by stacking various layers.

---

## Learning Outcomes

* A deep, practical understanding of the mathematical foundations of neural networks.
* Hands-on experience implementing the backpropagation algorithm from first principles.
* Insight into the roles of different layers, activation functions, and loss functions in a deep learning model.
