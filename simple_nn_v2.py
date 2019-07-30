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

        # Prepare inputs for the neuron
        self.dim = neurons

        if len(x) != neurons and layer_type == 'regular':
            raise ValueError(
                'In hidden layers input dimension should match neurons.')

        self.w = 2 * np.random.random_sample(len(x)) - 1
        self.x = x

        """
        Values computed on forward & backward pass. These are:
        1) The layer's output (s): a vector with the output for each neuron.
        2) The Accumulated error (e): the value passed back through the chain
           to be computed for previous layers.
        3) The delta (delta_w): the value that must be added to the weight to
           reduce the error (gradient descent).
        """
        if layer_type == 'regular':
            self.z = self.x * self.w
            self.s = self.solve_fwd()
        self.e = np.zeros(neurons)
        self.delta_w = np.zeros(neurons)

    @staticmethod
    def not_np(array, msg=None):
        """Test if the value is actually a numpy array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(msg)

    def solve_fwd(self):
        """Compute the regular layer values in the forward pass."""
        return 1 / (1 + np.exp(-self.z))

    def solve_bwd(self, acc_error, lr=1):
        """Compute the layer values in the backward pass."""
        self.not_np(acc_error, 'The accumulated error should be a numpy array')
        if len(acc_error) != self.dim:
            raise ValueError('The accumulated error dimension doesn\'t match!')

        if not isinstance(lr, int):
            raise TypeError('Learning rate should be an integer')

        for n in range(self.dim):  # Loop over neurons
            # Calculate the accumulated error
            partial_s = self.s[n] * (1 - self.s[n])
            mid = acc_error[n] * partial_s  # intermediate value proxy
            self.e[n] = mid * self.w[n]

            # Calculate the gradient descent
            self.delta_w[n] = -lr * mid * self.x[n]

            # Finally update the weight
            self.w[n] = self.w[n] + self.delta_w[n]

