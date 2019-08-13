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

    def test_auto_weigths_are_reshaped_on_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        self.assertEqual(layer.w.shape, (3, 2))

    def test_auto_weights_are_greater_than_minus_one(self):
        layer = Gateway(np.zeros(300))  # big enough
        self.assertTrue((layer.w > -1).all())

    def test_auto_weights_are_smaller_than_one(self):
        layer = Gateway(np.zeros(300))  # big enough
        self.assertTrue((layer.w < 1).all())

    def test_x_is_cloned_on_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        self.assertTrue((layer.x == np.array([[1, 2], [1, 2], [1, 2]])).all())

    def test_z_is_a_dot_product_element_wise_between_input_and_weight(self):
        i = np.array([1, 2])
        layer = Gateway(i)
        z = (layer.x * layer.w).sum(axis=1)
        self.assertTrue((layer.z == z).all())

    def test_output_for_gateway_layer(self):
        layer = Gateway(np.array([1, 2]))
        expected = 1 / (1 + np.exp(-layer.z))
        self.assertTrue((layer.s == expected).all())

    def test_output_for_gateway_layer_matches_dim(self):
        layer = Gateway(np.array([1, 2]))
        self.assertEqual(len(layer.s), layer.dim)

    def test_gateway_solve_bwd_check_args_np(self):
        layer = Gateway(np.array([1, 2]))
        regex = 'The accumulated error should be a numpy array'
        with self.assertRaisesRegex(TypeError, regex):
            layer.solve_bwd('void')

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

    def test_output_requires_an_arg(self):
        with self.assertRaises(TypeError):
            Output()

    def test_layer_x_is_input(self):
        i = np.array([1, 2, 3])
        layer = Output(i)
        self.assertTrue((layer.x == i).all())

    def test_z_is_a_dot_product_between_the_input_and_weight(self):
        i = np.array([1, 2, 3])
        layer = Output(i)
        self.assertTrue((layer.z == np.dot(i, layer.w)).all())

    def test_output_for_output_layer_is_one(self):
        layer = Output(np.array([1, 2, 3]))
        self.assertEqual(layer.s.size, 1)

    def test_output_value(self):
        layer = Output(np.array([1, 2, 3]))
        expected = 1 / (1 + np.exp(-layer.z))
        self.assertTrue((layer.s == expected).all())

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
        expected = np.repeat(err, x.shape[0]) * layer.partial_s * layer.w
        self.assertTrue((expected == layer.e).all())

    def test_output_error_passed_back_in_the_chain_matches_input_dim(self):
        x, err, lr = np.array([1, 2, 3]), np.array([0.5, ]), 3
        layer = Output(x)
        layer.solve_bwd(err, lr)
        self.assertEqual(layer.e.shape, layer.x.shape)

    def test_output_gradient_descent(self):
        x, err, lr = np.array([1, 2, 3]), np.array([0.5, ]), 3
        layer = Output(x)
        layer.solve_bwd(err, lr)
        expected = -lr * np.repeat(err, x.shape[0]) * layer.partial_s * layer.x
        self.assertTrue((expected == layer.delta_w).all())


class NetworkTests(unittest.TestCase):
    """Test the network class."""

    def test_layers_type(self):
        regex = 'Layers should be an int.'
        with self.assertRaisesRegex(TypeError, regex):
            Network(layers='a')

    def test_neurons_type(self):
        regex = 'Neurons should be an int.'
        with self.assertRaisesRegex(TypeError, regex):
            Network(neurons='void')

    def test_layers_value(self):
        regex = 'Layers value should be greater than one.'
        with self.assertRaisesRegex(ValueError, regex):
            Network(layers=0)

    def test_neurons_value(self):
        regex = 'Inner layer must have more than 2 neurons.'
        with self.assertRaisesRegex(ValueError, regex):
            Network(neurons=2)

    def test_network_gen_input_is_ndarray(self):
        nt = Network()
        self.assertIsInstance(nt.i, np.ndarray)

    def test_gen_input_output_values_either_0_or_1(self):
        for _ in range(500):  # Big enough
            ndarray = Network._gen_input()
            self.assertTrue(((ndarray == 0) | (ndarray == 1)).all())

    def test_expected_requires_ndarray_for_i(self):
        nt = Network()
        nt.i = 'void'
        regex = 'A numpy ndarray was expected.'
        with self.assertRaisesRegex(TypeError, regex):
            nt.expected()

    def test_expected_requires_ndarray_len_2_for_i(self):
        nt = Network()
        nt.i = np.array([1, 2, 3, ])
        regex = 'The length should be 2.'
        with self.assertRaisesRegex(ValueError, regex):
            nt.expected()

    def test_expected_requires_ndarray_with_0s_or_1s_for_i(self):
        nt = Network()
        nt.i = np.array([1, 2, ])
        regex = 'The values should be either 0 or 1'
        with self.assertRaisesRegex(ValueError, regex):
            nt.expected()

    def test_expected_returns_logic_or(self):
        nt = Network()
        nt.i = np.array([0, 0])
        self.assertEqual(nt.expected(), 0)
        nt.i = np.array([1, 0])
        self.assertEqual(nt.expected(), 1)
        nt.i = np.array([0, 1])
        self.assertEqual(nt.expected(), 1)
        nt.i = np.array([1, 1])
        self.assertEqual(nt.expected(), 1)

    def test_layer_track_is_a_list_of_layer_objects(self):
        nt = Network(layers=5)
        for layer in nt.layer_track:
            self.assertIsInstance(layer, Layer)

        # Five hidden layers + gateway + output
        self.assertEqual(len(nt.layer_track), 7)

    def test_layer_names(self):
        nt = Network(layers=2)
        ly = nt.layer_track
        self.assertEqual(ly[0].__str__(), 'gateway')
        self.assertEqual(ly[1].__str__(), 'hidden_1')
        self.assertEqual(ly[2].__str__(), 'hidden_2')
        self.assertEqual(ly[3].__str__(), 'output')

    def test_layer_input_is_previous_layer_output(self):
        nt = Network(layers=2)
        ly = nt.layer_track
        self.assertTrue((ly[0].x == nt.i).all())
        for i in range(2):
            self.assertTrue((ly[i+1].x == ly[i].s).all())

    def test_network_output_should_be_a_ndarray_len_one(self):
        nt = Network()
        self.assertIsInstance(nt.Op, np.ndarray)
        self.assertEqual(len(nt.Op), 1)

    def test_network_output_limits(self):
        for _ in range(100):
            nt = Network(layers=5)
            self.assertLess(nt.Op, 1)
            self.assertGreater(nt.Op, -1)

    def test_error_output_limits(self):
        for _ in range(100):
            nt = Network(layers=5, neurons=5)
            self.assertLessEqual(nt.E, 2)
            self.assertGreaterEqual(nt.E, 0)

    def test_a_whole_cycle(self):
        """
        Although individual tests have been written, test all together to stick
        knowledge and ensure everything is working as expected.
        """

        nt = Network(layers=1, neurons=3)
        gt, hd, op = nt.layer_track

        # Gateway, forward pass
        self.assertTrue((gt.x == np.tile(nt.i, [3, 1])).all())
        self.assertTrue(gt.x.shape == (3, 2))
        self.assertTrue(gt.w.shape == (3, 2))
        self.assertTrue((gt.z == (gt.x * gt.w).sum(axis=1)).all())
        self.assertTrue((gt.s == 1 / (1 + np.exp(-gt.z))).all())

        # Hidden, forward pass
        self.assertTrue((hd.x == gt.s).all())
        self.assertTrue(hd.x.shape == (3, ))
        self.assertTrue((hd.z == (hd.x * hd.w)).all())
        self.assertTrue((hd.s == 1 / (1 + np.exp(-hd.z))).all())

        # Output forward pass
        self.assertTrue((op.x == hd.s).all())
        self.assertTrue(op.x.shape == (3, ))
        self.assertTrue((op.z == np.dot(op.x, op.w)).all())
        self.assertTrue((op.s == 1 / (1 + np.exp(-op.z))).all())
        self.assertIsInstance(op.s, np.float64)

        # Network outcome
        self.assertTrue(nt.Op.shape == (1, ))
        self.assertTrue((nt.Op == op.s).all())
        self.assertTrue((nt.E == (.5 * (nt.y_hat - nt.Op)**2)))

        # Start backpropagation
        gtw0, hdw0, opw0 = gt.w, hd.w, op.w  # Keep original weights
        del_E = nt.partial_e()
        self.assertTrue((del_E == nt.Op - nt.y_hat).all())
        self.assertTrue(del_E.shape == (1, ))
        nt.backprop()

        # Output backprop, partial output with respect to input
        comp = op.s * (1 - op.s)
        partial_s = np.array([comp, comp, comp])
        self.assertTrue((op.partial_s == partial_s).all())
        self.assertEqual(op.partial_s.shape, (3, ))

        # Output backprop, error passed back in the chain
        net_error = np.array([del_E, del_E, del_E])
        expected = net_error * partial_s * opw0
        self.assertTrue((op.e == expected).all())
        self.assertEqual(op.e.shape, (3, ))

        # Output backprop, delta for weights
        expected = -1 * net_error * partial_s * op.x
        self.assertTrue((op.delta_w == expected).all())
        self.assertEqual(op.delta_w.shape, (3, ))

        # Output backprop, update weights
        self.assertTrue((op.w == opw0 + expected).all())
        self.assertEqual(op.w.shape, (3, ))

        # Hidden backprop, partial output w/ respect to input
        self.assertTrue((hd.partial_s == hd.s * (1 - hd.s)).all())
        self.assertEqual(hd.partial_s.shape, (3, ))

        # Hidden backprop, error passed back in the chain
        self.assertTrue((hd.e == op.e * hd.partial_s * hdw0).all())
        self.assertEqual(hd.e.shape, (3, ))

        # Hidden backprop, delta for weights
        self.assertTrue((hd.delta_w == -1 * op.e * hd.partial_s * hd.x).all())
        self.assertEqual(hd.delta_w.shape, (3, ))

        # Hidden backprop, update weights
        self.assertTrue((hd.w == hdw0 + hd.delta_w).all())
        self.assertEqual(hd.w.shape, (3, ))

        # Gateway backprop, patial output with respect to the input
        self.assertTrue((gt.partial_s == gt.s * (1 - gt.s)).all())
        self.assertEqual(gt.partial_s.shape, (3, ))

        # Gateway backprop, clone partial_s & acc error to match weights
        delta0 = -1 * gt.partial_s * hd.e
        delta_w = np.array(
            [[delta0[0], delta0[0]],
             [delta0[1], delta0[1]],
             [delta0[2], delta0[2]]])
        self.assertEqual(delta_w.shape, gt.x.shape)

        # Gateway backprop, calculate delta for weights
        self.assertTrue((gt.delta_w == delta_w * gt.x).all())
        self.assertEqual(gt.delta_w.shape, (3, 2))

        # Gateway backprop, update weights
        self.assertTrue((gt.w == gtw0 + gt.delta_w).all())
        self.assertEqual(gt.w.shape, (3, 2))

    def test_least_squares(self):
        nt = Network()
        nt.y_hat, nt.Op = 5, 2
        self.assertTrue((nt.least_squares() == 4.5))

    def test_backprop_partial_e_value(self):
        nt = Network()
        nt.y_hat, nt.Op = 5, 2
        self.assertTrue((nt.partial_e() == -3))


if __name__ == '__main__':
    unittest.main()
