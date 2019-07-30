import unittest

from simple_nn_v2 import Layer, Gateway, Output, Network

import numpy as np


class TestLayer(unittest.TestCase):
    """ Test the layer class."""

    def test_layer_requires_an_input(self):
        regex = 'missing 1 required positional argument: \'x\''
        with self.assertRaisesRegex(TypeError, regex):
            Layer()

    def test_layer_input_should_be_a_numpy_array(self):
        i = [1, 2, 3]
        regex = 'The input should be a numpy array'
        with self.assertRaisesRegex(TypeError, regex):
            Layer(i)

    def test_neurons_should_be_an_int(self):
        i = np.array([1, 2, 3])
        regex = 'Neurons should be a int.'
        with self.assertRaisesRegex(TypeError, regex):
            Layer(i, neurons='void')

    def test_only_certain_types_of_layer_are_allowed(self):
        i = np.array([1, 2, 3])
        regex = 'Only regular, gateway & output layer types are allowed.'
        with self.assertRaisesRegex(ValueError, regex):
            Layer(i, layer_type='void')

    def test_dim_is_the_number_of_neurons(self):
        i = np.array([1, 2, 3])
        layer = Layer(i)
        self.assertEqual(layer.dim, 3)

    def test_input_should_match_neurons_in_regular_layers(self):
        i = np.array([1, 2, 3])
        regex = 'In hidden layers input dimension should match neurons.'
        with self.assertRaisesRegex(ValueError, regex):
            Layer(i, neurons=2)

    def test_auto_weigths_match_shape_on_regular_layer(self):
        layer = Layer(np.array([1, 2, 3]))
        self.assertEqual(layer.w.shape, (3, ))

    def test_auto_weights_are_greater_than_minus_one_in_regular_layers(self):
        layer = Layer(np.zeros(300), neurons=300)  # big enough
        self.assertTrue((layer.w > -1).all())

    def test_auto_weights_are_smaller_than_one_in_regular_layers(self):
        layer = Layer(np.zeros(300), neurons=300)  # big enough
        self.assertTrue((layer.w < 1).all())

    def test_input_attribute_in_regular_layer(self):
        layer = Layer(np.array([1, 2, 3]))
        self.assertTrue((layer.x == np.array([1, 2, 3])).all())

    def test_z_is_the_product_between_input_and_weight(self):
        i = np.array([1, 2, 3])
        layer = Layer(i)
        self.assertTrue((layer.z == (i * layer.w)).all())

    def test_output_for_regular_layer(self):
        layer = Layer(np.array([1, 2, 3]))
        expected = 1 / (1 + np.exp(-layer.x * layer.w))
        self.assertTrue((layer.s == expected).all())

    def test_output_for_regular_layer_matches_the_no_of_neurons(self):
        layer = Layer(np.array([1, 2, 3]))
        self.assertEqual(len(layer.s), layer.dim)

    def test_init_value_for_accumulated_error_is_an_np_zeros(self):
        layer = Layer(np.zeros(3))
        self.assertIsInstance(layer.e, np.ndarray)
        self.assertTrue((layer.e == 0).all())

    def test_init_value_for_delta_w_is_an_np_zeros(self):
        layer = Layer(np.zeros(3))
        self.assertIsInstance(layer.delta_w, np.ndarray)
        self.assertTrue((layer.delta_w == 0).all())

