"""Create from scratch a simple neural network to find out logical OR.

It will take two values either, 0 or 1, and try to predict the OR clause
between them.
"""

import numpy as np
import pandas as pd

# Set the data for training. Inputs 1 & 2 and expected result
DATA = [[0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        ]


def SetUp(initial=DATA[1]):
    """Prepare the neccesary elements."""
    if initial not in DATA:
        raise ValueError('The inital data is not in DATA')

    # Build the initial vector
    v = [[initial[0], initial[1], ],
         [initial[0], initial[1], ],
         [initial[0], initial[1], ],
         ]

    """Forward propagation:
        raw: the initial data as 2 element list.

        # Mid Layer
        MLX: medium layer x, the initial data vectorized ready to be injected.
        MLW: medium layer weights.
        MLZ: The input for the neuron in the mid layer
        MLS: medium layer sigma, output for the first layer.

        # Output Layer
        OLW: output layer weights.
        OLZ: The input for the neuron in the output layer.
        OLS: output sigma, final output
        Expected: expected value, yhat.
        Error: deviation from the expected value.

       Backpropagation:
        dE_dOLS: change in the error w/ respect to the last neuron output.
        dOLS_dOLZ: change in last neuron output w/ respect to its input.
        dE_dOLW: change in the error w/ respect to the output layer weights.
        dOLZ_dMLS: change in the output layer input w/ respect to its input.
        dOLX_dMLZ: change in the middle layer output w/ respect to its input.
        dE_dMLW: change in the error w/ respect to the mid layer weights.

    """

    data = {'raw': [np.array([initial[0], initial[1]])],
            # Mid layer
            'MLX': [np.array(v)],
            'MLW': [np.random.random((3, 2))],
            'MLZ': [np.zeros(3)],
            'MLS': [np.zeros(3)],

            # Output layer
            'OLW': [np.random.random(3)],
            'OLZ': np.zeros(1),
            'OLS': np.zeros(1),
            'Expected': initial[2],
            'E': np.zeros(1),

            # Backpropagation
            'dE_dOLS': np.zeros(1),
            'dOLS_dOLZ': np.zeros(1),
            'dE_dOLW': [np.zeros(3)],
            'dOLZ_dMLS': [np.zeros(3)],
            'dOLX_dMLZ': [np.zeros(3)],
            'dE_dMLW': [np.zeros((3, 2))]
            }

    return pd.DataFrame(data)


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

    def partial_E(self, yhat, sigma):
        """Compute the partial of E with respect to sigma."""
        return sigma - yhat

    def partial_sigma(self, sigma):
        """Compute the partial of sigma with respect to z."""
        return sigma * (1 - sigma)

    def partial_w(self, x):
        """Compute the partial of w with repect to z."""
        return x

    def partial_x(self, w):
        """Compute the partial of x with respect to z."""
        return w

    def forward(self, df):
        """Fill forward the missing values in the df."""
        # fill the mid layer outputs
        c_row = len(df) - 1
        mlz, mls = list(), list()
        for c, val in enumerate(df.loc[c_row, 'MLX']):
            z = self.z(val, df.loc[c_row, 'MLW'][c])
            mlz.append(z)
            mls.append(self.sigma(z))
        df.at[c_row, 'MLZ'], df.at[c_row, 'MLS'] = mlz, mls

        # Fill out the output layer
        z = self.z(df.loc[c_row, 'MLS'], df.loc[c_row, 'OLW'])
        df.at[c_row, 'OLS'] = self.sigma(z)

        # And the error
        yhat, sigma = df.loc[c_row, 'Expected'], df.loc[c_row, 'OLS']
        df.at[c_row, 'E'] = self.E(yhat, sigma)

        return df
        yhat, sigma = df.iloc[-1, 5], df.iloc[-1, 4]
        df.at[len(df) - 1, 'Error'] = self.E(yhat, sigma)

        return df

    def backpropagation(self, df):
        """Fill out backpropagation values."""
        c_row = len(df) - 1  # current row

        # fill the error w/ rspct to outer layer output
        yhat, sigma = df.loc[c_row, ['Expected', 'OLS']]
        df.at[c_row, 'dE_dOLS'] = self.partial_E(yhat, sigma)

        # outer output w/ rspct to outer input
        df.at[c_row, 'dOLS_dOLZ'] = self.partial_sigma(sigma)

        # error w/ respect to the outer weight (chain rule)
        de_dolw = list()
        de_dols, dols_dolz = df.loc[c_row, ['dE_dOLS', 'dOLS_dOLZ']]
        c_rule = de_dols * dols_dolz
        for val in df.loc[c_row, 'MLS']:
            de_dolw.append(self.partial_w(val) * c_rule)
        df.at[c_row, 'dE_dOLW'] = de_dolw

        # outer input w/ rspct to mid output
        dolz_dmls = list()
        for val in df.loc[c_row, 'OLW']:
            dolz_dmls.append(self.partial_x(val))
        df.at[c_row, 'dOLZ_dMLS'] = dolz_dmls

        # mid output w/ rspct to mid input
        dmls_dmlz = list()
        for val in df.loc[c_row, 'MLS']:
            dmls_dmlz.append(self.partial_sigma(val))
        df.at[c_row, 'dMLS_dMLZ'] = dmls_dmlz

        # error w/ rspct to mid weight (chain rule)
        de_dmlw = list()
        dolz_dmls, dmls_dmlz = df.loc[c_row, ['dOLZ_dMLS', 'dMLS_dMLZ']]

        # multiply element wise to get a chain rule vector for the last step
        c_rule_v = list()
        for c, val in enumerate(dolz_dmls):
            c_rule_v.append(c_rule * val * dmls_dmlz[c])

        # finally compute the error
        de_dmlw_comp = list()
        for c, vector in enumerate(df.loc[c_row, 'MLX']):
            c_rule_vc = c_rule_v[c]
            for comp in vector:
                de_dmlw_comp.append(self.partial_w(comp) * c_rule_vc)
            de_dmlw.append(de_dmlw_comp)
            de_dmlw_comp = list()

        df.at[c_row, 'dE_dMLW'] = de_dmlw

        return df
    def go(self):
        """Launch the training."""
        # First pass
        df = self.forward(self.df)
        df = self.backpropagation(df)

