"""Create from scratch a simple neural network to find out logical OR.

It will take two values either, 0 or 1, and try to predict the OR clause
between them.
"""

import numpy as np
import pandas as pd

# Set the data for training
DATA = [[0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        ]


def SetUp(initial=DATA[1]):
    """Prepare the neccesary elements."""
    if initial not in DATA:
        raise ValueError('The inital data is not in DATA')

    # Set random initial weights
    mlw = np.random.random((3, 2))
    olw = np.random.random(3)

    # Build the initial vector
    v = [[initial[0], initial[1], ],
         [initial[0], initial[1], ],
         [initial[0], initial[1], ],
         ]

    """Understanding the columns:
        init: th initial vector prepared to to be injected in the three neurons
        of the medium layer.
        MLW: medium layer weights.
        MS: medium layer sigma, output for the first layer.
        OWL: output layer weights.
        OS: output sigma, final output
        Expected: expected value, yhat.
        Error: deviation from the expected value.
        Pass: informs about the direction of the training.
    """

    data = {'init': [v, ],  # initial input
            'MLW': [mlw, ], 'MS': [np.zeros(3)],  # Middle layer
            'OLW': [olw, ], 'OS': [np.zeros(1)],  # Output layer
            'Expected': initial[2],
            'Error': np.nan, }

    return pd.DataFrame(data=data)


class Train:
    """Run the network forward to compute the outputs and the error."""

    def __init__(self, df):
        """Require the weights df."""
        self.df = df

    def z(self, x, w):
        """Compute the value for z."""
        return np.dot(x, w)

    def sigma(self, z):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def E(self, yhat, sigma):
        """Compute the error function."""
        return 0.5 * (yhat - sigma)**2

    def forward(self, df):
        """Fill forward the missing values in the df."""
        # fill the mid layer outputs
        ms = list()
        for c, val in enumerate(df.iloc[-1, 0]):
            z = self.z(val, df.iloc[-1, 1][c])
            ms.append(self.sigma(z))
        df.at[len(df) - 1, 'MS'] = ms

        # Compute final output
        z = self.z(df.iloc[-1, 2], df.iloc[-1, 3])
        df.at[len(df) - 1, 'OS'] = self.sigma(z)

        # And the error
        yhat, sigma = df.iloc[-1, 5], df.iloc[-1, 4]
        df.at[len(df) - 1, 'Error'] = self.E(yhat, sigma)

        return df

    def go(self):
        """Launch the training."""
        f = self.forward(self.df)

        return f
