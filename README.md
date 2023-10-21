This repository contains a Python implementation of a Multilayer Perceptron (MLP) from scratch. The code is contained in a Jupyter notebook, `main.ipynb`.

## Description

The `NeuralNetMLP` class in the notebook implements a basic MLP with one hidden layer. The class includes methods for forward propagation (`_forward`), computing the loss (`_compute_loss`), making predictions (`predict`), and training the network (`fit`). The class also includes methods for saving and loading the model weights (`save_weights` and `load_weights`).

## Usage

To use the `NeuralNetMLP` class, you need to provide the following parameters:

- `n_hidden`: The number of hidden units.
- `l2`: The lambda value for L2 regularization.
- `epochs`: The number of epochs for training.
- `eta`: The learning rate.
- `shuffle`: Whether to shuffle the training data every epoch.
- `minibatch_size`: The size of the minibatches for stochastic gradient descent.
- `seed`: The random seed for weight initialization and shuffling.

After creating an instance of the class, you can train the network using the `fit` method, which requires the training and validation data as inputs. The `predict` method can be used to make predictions on new data.

## Example

```python
nn = NeuralNetMLP(n_hidden=30, l2=0.01, epochs=100, eta=0.0005, shuffle=True, minibatch_size=1, seed=1)
nn.fit(X_train, y_train, X_valid, y_valid)
predictions = nn.predict(X_test)
```

## Dependencies

This code requires the following libraries:

- numpy
- keras
