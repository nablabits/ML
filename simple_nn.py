"""Create from scratch a simple neural network to find out logical or.

It will take two values either, 0 or 1, and try to predict the or clause
between them.
"""

import numpy as np
import pandas as pd


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


