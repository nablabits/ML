# Simple NN
***
## About
This project is the implementation of a neural network that predicts the result of an `OR` operation between two elements, either 1 or 0.

## Content
There are two files with two different approaches to the problem and the test suite for the second one.

## Brief description of how a neural network works
A neural network is an computational object that makes predictions about the numerical inputs it receives. Even more, it's capable of self-improve these predictions, so they approach to the real value, thereby a process called training.

A neural network is made by a set of layers that contains elements called neurons which receive several (raw) inputs. Each one of this inputs has also a weight --randomly assigned in the beginning-- associated to it. These neurons perform two operations:
  1. They transform the inputs/weights into a weighted input by taking the dot product between them. Notice that they are vectors.

  2. They flatten the weighted input (now a scalar) into values between -1 and 1 using a sigmoid function, sometimes called activation function, although some other fancy functions can be used.

The first layer is fed by the initial input whereas next ones are fed by the outputs of the previous one. That is, outputs in one layer become inputs for the next one.

The last layer (output layer) contains a single neuron that produces the final output for the network. This output is then compared with the expected result with an error function.

The key process is to adjust the weights so the output approaches to the expected value, that is, the **error function is minimized**.

That is done through a process called **backpropagation** that determines how each weight affects the error. In other words, the goal is to determine how slightly changes (differential changes, understand derivatives) in the weights affect the final error.

Once known how these weights affect the final error they could be tweaked so the error gets lower.

***

### V1, simple_nn.py
It uses a pandas dataframe to track all the changes along the cycles adding a new row for each one.

It can be run simply by calling this file from the shell (remember to install a virtual environment with all the dependencies in the requirements file `pip install -r requirements.txt`.).


#### V1.a, Set up
It creates first the dataframe (df) calling `SetUp()` method. This df has several cols for the data meaning three layers --input, mid and output, with 2-3-1 neurons respectively--.

This is an excerpt of the *docstring* inside `SetUp()` that references the cols found in the df:

```
  Forward propagation:
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
  dMLS_dMLZ: change in the middle layer output w/ respect to its input.
  dE_dMLW: change in the error w/ respect to the mid layer weights.
```

#### v1.b, Train class and computation functions
This df is passed to `Train` class which will fill up and/or update all the values in the network throughout the cycle. `Train` first defines all the functions that are going to be used in the process to change values in the df. These are:
  * `z()`: get weighted input.
  * `sigma()`: the activation function (here a sigmoid).
  * `E()`: the error function (loss) for the last output.
  * `partial_E()`: the change in the error with respect to the network's output.
  * `partial_sigma()`: the change in the neuron's output with respect to its weighted input `z`.
  * `partial_w()`: the change in the neuron's weighted input with respect to its weigth.
  * `partial_x()`: the change in the neuron's weighted input with respect to its raw input.

Next are defined the two core processes in the cycle: forward and back propagation.

#### v1.c, Forward propagation.
Computes a guess for the network using current weights (initial or updated in the last pass) and evaluates the error function.

Notice that for the mid layer, each neuron receives the two initial inputs and, remember, it takes the dot product between them and the weights that will be the argument for the activation function. That's why there's a for loop that iterates over neurons.

Once got the output for the network, it computes the error.

#### v1.d, Backpropagation.
The key process, where the network really *learns*. It computes the chain rule backwards so it can be known how exactly each weight has contributed to the final error.

As with forward propagation, the mid layer has two weights for each neuron and, therefore, it must be computed how each one has contributed to the error, so again we have to iterate over neurons.

Also notice that for the neuron in the output layer there are two computations:
  * The error that should be passed back in the chain. (derivativative of z wrt to the raw input, that is, the weight)

  * The current weight's contribution to the final error (derivative of z wrt to the weigth, that is, the raw input.).


#### v1.e, Update weights.
Now it's time to update the weights so the error can be reduced effectively. Since the gradient (we were calculating gradients with those chain rules) is the steepest ascent we should go against it in order to decrease the error, that is:

  * (+) Positive gradients mean that the error increases when we increase the weight, therefore, decrease it (-)

  * (-) Negative gradients mean that the error decreases as we increase the weight, therefore, increase it (+)

Also, a learning rate, a factor that multiplies the effect of the gradient, must be chosen.

There's a lot to be said about learning rates, but quickly:
  * If it's too high we won't reach the minimum value.

  * Alternatively, if it's too low we can reach a minimum value but it could be a local minimum rather than the global one.

#### v1.f, Launch the training.
Chains all the processes in the network for several loops creating on each one a new input for network using `inject()` method, that adds a new row on the dataframe, and updating weights, passing forward and computing backpropagation.
