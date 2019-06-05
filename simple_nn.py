"""Create from scratch a simple neural network to find out logical or.

It will take two values either, 0 or 1, and try to predict the or clause
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


def SetUp(initial):
    """Prepare the neccesary elements."""
    # Initial should be either 0 or 1 and contain 2 elements
    if len(initial) != 2:
        raise ValueError('For this network only two inputs are allowed')
    if initial[0] not in (0, 1) or initial[1] not in (0, 1):
        raise ValueError('The inputs should be either 0 or 1')

    # Set random initial weights
    mlw = np.random.random((3, 2))
    olw = np.random.random((3, 2))

    # Build the initial vector
    initial = [[initial[0], initial[1], ],
               [initial[0], initial[1], ],
               [initial[0], initial[1], ],
               ]

    data = {'init': [initial, ],  # initial input
            'MLW': [mlw, ], 'MO': [np.zeros(3)],  # Middle layer
            'OWL': [olw, ], 'OS': [np.zeros(1)],  # Output layer
            'Error': np.nan, }

    return pd.DataFrame(data=data)


class Train:
    """Run the network forward to compute the outputs and the error."""

    def __init__(self, df):
        """Require the weights df."""
        self.df = df

    def sigma(self, z):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def E(self, yhat, sigma):
        """Compute the error function."""
        return 0.5 * (yhat - sigma)

    def forward(self, df):
        """Fill the nan values in the df."""
        # fill the mid layer outputs
        ms = list()
        for c, val in enumerate(df.iloc[-1, 0]):
            z = np.dot(val, df.iloc[-1, 1][c])
            ms.append(self.sigma(z))
        df.at[len(df) - 1, 'MS'] = ms

        # Compute final output
        z = np.dot(df.iloc[-1, 2], df.iloc[-1, 3])
        df.at[len(df) - 1, 'OS'] = self.sigma(z)

        # And the error
        yhat, sigma = df.iloc[-1, 5], df.iloc[-1, 4]
        df.at[len(df) - 1, 'Error'] = self.E(yhat, sigma)

        return df

    def go(self):
        """Launch the training."""
        f = self.forward(self.df)

        return f
