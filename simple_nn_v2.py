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

        # now redefine the attr for this special layer
        self.w = 2 * np.random.random_sample((self.dim, len(x))) - 1
        self.s = self.solve_fwd(x)

    def neuron_input(self):
        """Define the input for activation function."""
        return (self.x * self.w).sum(axis=1)

    def solve_fwd(self, x):
        """Compute the regular layer values in the forward pass."""
        # check args
        super().not_np(
            x, 'The input should be a numpy array')
        self.x = np.tile(x, [self.dim, 1])  # clone input to match neurons
        self.z = self.neuron_input()
        return 1 / (1 + np.exp(-self.z))

    def solve_bwd(self, acc_error, lr=1):
        """Compute the layer values in the backward pass."""
        # check args
        super().not_np(
            acc_error, 'The accumulated error should be a numpy array')

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

        # Now, solve partial s and replicate to match weights & inputs
        partial_s = self.s * (1 - self.s)
        self.partial_s = np.repeat(partial_s, self.x.shape[0])
        net_error = np.repeat(net_error, self.x.shape[0])

        # Accumulate error to be passed back in the chain --rule--
        common = net_error * self.partial_s
        self.e = common * self.w

        # Gradient descent
        self.delta_w = -lr * common * self.x


class Network:
    """Create a new neural network."""

    def __init__(self, layers=1, neurons=3, lr=1, cli=False):
        """Set up the network.

        Builds the first forward pass leaving it prepared for the backprop pass
        just where the trainig starts out.

        Layers arg stands for hidden layers (excluding gateway & output ones).

        Notice that, for the sake of simplicity, neurons are connected one to
        one across hidden layers which, therefore, must have the same number
        of neurons.
        """
        if cli:
            print('Starting Network...')
        # check args
        if not isinstance(layers, int):
            raise TypeError('Layers should be an int.')
        if not isinstance(neurons, int):
            raise TypeError('Neurons should be an int.')
        if layers < 1:
            raise ValueError('Layers value should be greater than one.')
        if neurons < 3:
            raise ValueError('Inner layer must have more than 2 neurons.')

        # Initial pass #
        ################

        # First, generate the input & expected value
        self.i = self._gen_input()
        self.y_hat = self.expected()

        self.layer_track = list()  # Track the layers

        # Now create the network, first the gateway
        gtway = Gateway(self.i, neurons=neurons)
        gtway.name = 'gateway'
        self.layer_track.append(gtway)

        # now the hidden layers
        x, n = gtway.s, 1
        for i in range(layers):
            layer = Layer(x, neurons=neurons)
            layer.name = str('hidden_%s' % n)
            self.layer_track.append(layer)
            x = layer.s
            n += 1

        # Process output layer
        self.output = Output(x)
        self.output.name = 'output'
        self.layer_track.append(self.output)

        # Network outcome
        self.Op = np.array([self.output.s, ])
        self.E = self.least_squares()

        self.train_loss = np.array([self.E, ])

        # add learning rate attribute
        self.lr = lr

    @staticmethod
    def _gen_input():
        """Generate a random input to be used."""
        a, b = np.random.randint(0, 2), np.random.randint(0, 2)
        return np.array([a, b])

    def expected(self):
        """Compute the expected result for the network."""
        if not isinstance(self.i, np.ndarray):
            raise TypeError('A numpy ndarray was expected.')
        if len(self.i) != 2:
            raise ValueError('The length should be 2.')
        if not ((self.i == 0) | (self.i == 1)).all():
            raise ValueError('The values should be either 0 or 1')
        val = self.i[0] | self.i[1]
        return np.array([val, ])

    def least_squares(self):
        """Compute the error for the network."""
        sq_error = 0.5 * (self.y_hat - self.Op)**2
        return sq_error

    def partial_e(self):
        """Compute the error with respect to last layer output."""
        return (self.Op - self.y_hat)

    def fwd(self, i=None):
        """Perform the forward pass in the training."""
        if i is None:
            self.i = self._gen_input()
        else:
            self.i = i
        self.y_hat = self.expected()
        x = self.i
        for layer in self.layer_track:
            x = layer.solve_fwd(x)
        self.Op = np.array([x, ])

    def backprop(self):
        """Perform the backpropagation pass."""
        acc_error = self.partial_e()
        for layer in self.layer_track[::-1]:
            layer.solve_bwd(acc_error, lr=self.lr)
            acc_error = layer.e
            layer.update_weights()

    def train(self, cycles=100, epochs=20, cli=False):
        """Train the network."""
        if cli:
            print('Training...', end='')
        # store some useful values for outcome
        self.cycles, self.epochs = cycles, epochs

        # Start out performing a backprop to start the cycles on fwd pass
        self.backprop()

        for cycle in range(cycles * epochs):
            self.fwd()
            self.E = self.least_squares()  # update error
            self.train_loss = np.append(self.train_loss, [self.E, ])
            self.backprop()
        print('done')

    def outcome(self):
        """Print some outcomes for the training."""
        print('\nNETWORK OUTCOME')
        print(20 * '#' + '\n')
        print(('-> Built a Neural network with {} layers having the inner ' +
              'layer {} neurons').format(
                len(self.layer_track), self.layer_track[1].dim))
        print('-> After {} epoch(s) of {} cycles per epoch:'.format(
            self.epochs, self.cycles))
        for n, layer in enumerate(self.layer_track):
            print('-> Layer {} final weights are:\n {}'.format(n, layer.w))

        print('Final error for the network (loss) is:', self.train_loss[-1])
        print('[initial was: {}]'.format(self.train_loss[0]))

        i = np.array([0, 0])
        print('\nTesting the value {}|{} = {}'.format(
            i[0], i[1], i[0] | i[1]
        ))
        nt.fwd(i=i)
        print('Network result:', nt.Op)


if __name__ == '__main__':
    nt = Network(lr=5, layers=1, neurons=5, cli=True)
    nt.train(cli=True)
    nt.outcome()

#
#
#
#
#
#
#
#
