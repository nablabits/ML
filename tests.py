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

    def test_layer_name_is_set_to_none(self):
        self.assertTrue(Layer(np.array([1, 2, 3])).name is None)

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

    def test_solve_fwd_requires_one_argument(self):
        layer = Layer(np.array([1, 2, 3]))
        with self.assertRaises(TypeError):
            layer.solve_fwd()

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

    def test_str_returns_name(self):
        layer = Layer(np.zeros(3))
        layer.name = 'test'
        self.assertEqual(layer.__str__(), 'test')

    def test_solve_bwd_requires_one_arg(self):
        layer = Layer(np.array([1, 1, 2]))
        regex = 'missing 1 required positional argument: \'acc_error\''
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd()

    def test_solve_bwd_acc_error_should_be_np_array(self):
        layer = Layer(np.array([1, 1, 2]))
        regex = 'The accumulated error should be a numpy array'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd('void')

    def test_solve_bwd_acc_error_matches_dimension(self):
        layer = Layer(np.array([1, 1, 2]))
        regex = 'The accumulated error dimension doesn\'t match!'
        with self.assertRaisesRegex(ValueError, regex):
            layer.solve_bwd(np.zeros(5))

    def test_solve_bwd_lr_is_int(self):
        layer = Layer(np.array([1, 1, 2]))
        regex = 'Learning rate should be an integer'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd(np.zeros(3), 'str')

    def test_partial_s_is_the_derivative_of_sigmoid_function(self):
        layer = Layer(np.array([1, 1, 2]))
        part_s = layer.s * (1 - layer.s)
        layer.solve_bwd(np.zeros(3), 5)
        self.assertTrue((part_s == layer.partial_s).all())

    def test_partial_s_dimension_matches_neuron_qty(self):
        layer = Layer(np.array([1, 1, 2]))
        layer.solve_bwd(np.zeros(3), 5)
        self.assertEqual(len(layer.partial_s), layer.dim)

    def test_error_passed_back_meets_chain_rule(self):
        layer = Layer(np.array([1, 1, 2]))
        acc_error = np.array([0.5, 0.5, 0.5])
        layer.solve_bwd(acc_error, 5)
        chain_rule = acc_error * layer.partial_s * layer.w
        self.assertTrue((layer.e == chain_rule).all())

    def test_error_passed_back_matches_neuron_dimension(self):
        layer = Layer(np.array([1, 1, 2]))
        layer.solve_bwd(np.zeros(3))
        self.assertEqual(len(layer.e), layer.dim)

    def test_delta_w_value(self):
        layer = Layer(np.array([1, 1, 2]))
        acc_error, lr = np.array([0.5, 0.5, 0.5]), 5
        layer.solve_bwd(acc_error, lr)
        expected = -lr * acc_error * layer.partial_s * layer.x
        self.assertTrue((layer.delta_w == expected).all())

    def test_delta_w_matches_neuron_dimension(self):
        layer = Layer(np.array([1, 1, 2]))
        layer.solve_bwd(np.zeros(3))
        self.assertEqual(len(layer.delta_w), layer.dim)

    def test_update_weights(self):
        layer = Layer(np.array([1, 1, 2]))
        w0 = layer.w
        acc_error, lr = np.array([0.5, 0.5, 0.5]), 5
        layer.solve_bwd(acc_error, lr)
        layer.update_weights()
        self.assertTrue((layer.w == (w0 + layer.delta_w)).all())


class TestGateway(unittest.TestCase):
    """Test the special gateway layer."""

    def test_gateway_is_a_Layer_subclass(self):
        self.assertIsInstance(Gateway(np.array([1, 2])), Layer)

    def test_x_is_cloned_on_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        self.assertTrue((layer.x == np.array([[1, 2], [1, 2], [1, 2]])).all())

    def test_auto_weigths_are_reshaped_on_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        self.assertEqual(layer.w.shape, (3, 2))

    def test_auto_weights_are_greater_than_minus_one(self):
        layer = Gateway(np.zeros(300))  # big enough
        self.assertTrue((layer.w > -1).all())

    def test_auto_weights_are_smaller_than_one(self):
        layer = Gateway(np.zeros(300))  # big enough
        self.assertTrue((layer.w < 1).all())

    def test_z_is_a_dot_product_element_wise_between_input_and_weight(self):
        i = np.array([1, 2])
        layer = Gateway(i)
        z = (layer.x * layer.w).sum(axis=1)
        self.assertTrue((layer.z == z).all())

    def test_output_for_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        expected = 1 / (1 + np.exp(-(layer.x * layer.w).sum(axis=1)))
        self.assertTrue((layer.s == expected).all())

    def test_output_for_gateway_layer_matches_dim(self):
        layer = Gateway(np.array([1, 2]))
        self.assertEqual(len(layer.s), layer.dim)

    def test_gateway_solve_bwd_check_args_np(self):
        layer = Gateway(np.array([1, 2]))
        regex = 'The accumulated error should be a numpy array'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd('void')

    def test_gateway_solve_bwd_check_args_learning_rate(self):
        layer = Gateway(np.array([1, 2]))
        regex = 'Learning rate should be an integer'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd(np.zeros(3), 'str')

    def test_gateway_partial_s(self):
        layer = Gateway(np.array([1, 2]))
        layer.solve_bwd(np.repeat(1, 3))

    def test_delta_w_dimension_matches_w_dimension(self):
        layer = Gateway(np.array([1, 2]))
        delta_w = layer.solve_bwd(np.repeat(1, 3))
        self.assertTrue(delta_w.shape == layer.w.shape)


class TestOutputLayer(unittest.TestCase):
    """Test the special output layer."""

    def test_output_is_a_Layer_subclass(self):
        self.assertIsInstance(Output(np.array([1, 2])), Layer)

    def test_z_is_a_dot_product_between_the_input_and_weight(self):
        i = np.array([1, 2, 3])
        layer = Output(i)
        self.assertTrue((layer.z == np.dot(i, layer.w)).all())

    def test_output_for_output_layer_is_one(self):
        layer = Output(np.array([1, 2, 3]))
        self.assertEqual(layer.s.size, 1)

    def test_output_solve_bwd_check_args_np(self):
        layer = Output(np.array([1, 2]))
        regex = 'The network error should be a numpy array'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd('void')

    def test_output_solve_bwd_check_args_shape(self):
        layer = Output(np.array([1, 2]))
        regex = 'The shape for net error should be 1'
        with self.assertRaisesRegex(ValueError, regex):
            layer.solve_bwd(np.array([1, 2]))

    def test_output_solve_bwd_check_args_learning_rate(self):
        layer = Output(np.array([1, 2]))
        regex = 'Learning rate should be an integer'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd(np.zeros(1), 'str')

    def test_output_partial_s(self):
        x = np.array([1, 2, 3])
        layer = Output(x)
        layer.solve_bwd(np.array([0.5, ]))
        partial_s = layer.s * (1 - layer.s)
        self.assertTrue((layer.partial_s == partial_s).all())
        self.assertEqual(layer.partial_s.shape, x.shape)

    def test_output_acc_error_to_be_passed_back_in_the_chain(self):
        x, err, lr = np.array([1, 2, 3]), np.array([0.5, ]), 3
        layer = Output(x)
        layer.solve_bwd(err, lr)
        expected = -lr * np.repeat(err, x.shape[0]) * layer.partial_s * layer.w
        self.assertTrue((expected == layer.e).all())

    def test_output_gradient_descent(self):
        x, err, lr = np.array([1, 2, 3]), np.array([0.5, ]), 3
        layer = Output(x)
        layer.solve_bwd(err, lr)
        expected = -lr * np.repeat(err, x.shape[0]) * layer.partial_s * layer.x
        self.assertTrue((expected == layer.delta_w).all())


