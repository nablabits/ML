"""
Create from the scratch a neural network that predicts the outcome for the
logical OR operation.

This version gets rid of pandas dataframe in the previous one.
"""

import numpy as np


class Layer:
    """Create a new hidden layer."""

    def __init__(self, x, neurons=3, layer_type='regular'):
        """Set up a hidden layer inside the network.

        The regular hidden layers get one input and spit out one output for
        each neuron.
        """
        # Check args
        self.not_np(x, 'The input should be a numpy array')

        if not isinstance(neurons, int):
            raise TypeError('Neurons should be a int.')

        if layer_type not in ('regular', 'gateway', 'output'):
            raise ValueError(
                'Only regular, gateway & output layer types are allowed.')

        # Set layer name
        self.name = None

        # Prepare inputs for the neuron
        self.dim = neurons

        if len(x) != neurons and layer_type == 'regular':
            raise ValueError(
                'In hidden layers input dimension should match neurons.')

        self.w = 2 * np.random.random_sample(len(x)) - 1

        """
        Values computed on forward & backward pass. These are:
        1) The layer's output (s): a vector with the output for each neuron.
        2) The Accumulated error (e): the value passed back through the chain
           to be computed for previous layers.
        3) The delta (delta_w): the value that must be added to the weight to
           reduce the error (gradient descent).
        """
        self.s = self.solve_fwd(x)
        self.e = np.zeros(neurons)
        self.delta_w = np.zeros(neurons)

    def __str__(self):
        """Return the custom name for the layer."""
        return self.name

    @staticmethod
    def not_np(array, msg=None):
        """Test if the value is actually a numpy array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(msg)

    def neuron_input(self):
        """Define the input for activation function."""
        return self.x * self.w

    def solve_fwd(self, x):
        """Compute the regular layer values in the forward pass."""
        self.x = x
        self.z = self.neuron_input()
        return 1 / (1 + np.exp(-self.z))

    def solve_bwd(self, acc_error, lr=1):
        """Compute the layer values in the backward pass."""
        # Check args
        self.not_np(acc_error, 'The accumulated error should be a numpy array')
        if len(acc_error) != self.dim:
            raise ValueError('The accumulated error dimension doesn\'t match!')

        if not isinstance(lr, int):
            raise TypeError('Learning rate should be an integer')

        # Accumulate error to be passed back in the chain --rule--
        self.partial_s = self.s * (1 - self.s)
        self.e = acc_error * self.partial_s * self.w

        # Gradient descent
        self.delta_w = -lr * acc_error * self.partial_s * self.x

    def update_weights(self):
        """Update weights for the layer."""
        self.w = self.w + self.delta_w


class Gateway(Layer):
    """The first layer in the hidden layers.

    Gateway layers are special because they clone every input so each neuron
    receives a copy of them. It also creates the weights accordingly.
    """

    def __init__(self, x, neurons=3):
        """Set up a gateway layer."""
        super().__init__(x, neurons=neurons, layer_type='gateway')

        # now redefine the terms to this special layer
        self.w = 2 * np.random.random_sample((self.dim, len(x))) - 1
        self.s = self.solve_fwd(x)

    def neuron_input(self):
        """Define the input for activation function."""
        return (self.x * self.w).sum(axis=1)

    def solve_fwd(self, x):
        """Compute the regular layer values in the forward pass."""
        self.x = np.tile(x, [self.dim, 1])  # clone input to match neurons
        self.z = self.neuron_input()
        return 1 / (1 + np.exp(-self.z))

    def solve_bwd(self, acc_error, lr=1):
        """Compute the layer values in the backward pass."""
        # check args
        super().not_np(
            acc_error, 'The accumulated error should be a numpy array')
        if not isinstance(lr, int):
            raise TypeError('Learning rate should be an integer')

        # Now compute it
        self.partial_s = self.s * (1 - self.s)
        delta0 = -lr * self.partial_s * acc_error
        delta0 = np.repeat(delta0, self.x.shape[1]).reshape(self.x.shape)
        self.delta_w = delta0 * self.x
        return self.delta_w


class Output(Layer):
    """The last layer in the network.

    Output layer combines the outputs in the last hidden layer to spit out a
    single value as the result of the network.
    """

    def __init__(self, x):
        """Set up a output layer."""
        # first call the base layer
        super().__init__(x, neurons=1, layer_type='output')

        # now redefine the terms to this special layer
        self.s = self.solve_fwd(x)

    def neuron_input(self):
        """Define the input for activation function."""
        return np.dot(self.x, self.w)

    def solve_fwd(self, x):
        """Compute the regular layer values in the forward pass."""
        self.x = x
        self.z = self.neuron_input()
        return 1 / (1 + np.exp(-self.z))

    def solve_bwd(self, net_error, lr=1):
        """Compute the layer values in the backward pass."""
        # check args
        super().not_np(
            net_error, 'The network error should be a numpy array')
        if net_error.shape != (1, ):
            raise ValueError('The shape for net error should be 1')
        if not isinstance(lr, int):
            raise TypeError('Learning rate should be an integer')

        # Now, solve partial s and replicate to match weights & inputs
        partial_s = self.s * (1 - self.s)
        self.partial_s = np.repeat(partial_s, self.x.shape[0])
        net_error = np.repeat(net_error, self.x.shape[0])

        # Accumulate error to be passed back in the chain --rule--
        self.e = -lr * net_error * self.partial_s * self.w

        # Gradient descent
        self.delta_w = -lr * net_error * self.partial_s * self.x


