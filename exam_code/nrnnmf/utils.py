from __future__ import absolute_import, print_function
# Standard modules
import math
# Third party modules
import tensorflow as tf

def _weight_init_range(n_in, n_out):
    """Calculates range for picking initial weight values from a uniform distribution."""
    range = 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)
    return {
        'minval': -range,
        'maxval': range,
    }

def build_mlp(f_input_layer, hidden_units_per_layer):
    """Builds a feed-forward NN (MLP) with 2 hidden layers."""
    # Note: tf.contrib.layers could likely be used instead, but total control allows for easier debugging in this case
    # TODO make number of hidden layers a parameter, if needed
    num_f_inputs = f_input_layer.get_shape().as_list()[1]

    # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
    mlp_weights = {
        'h1': tf.Variable(tf.random_uniform([num_f_inputs, hidden_units_per_layer],
                                            **_weight_init_range(num_f_inputs, hidden_units_per_layer))),
        'h2': tf.Variable(tf.random_uniform([hidden_units_per_layer, hidden_units_per_layer],
                                            **_weight_init_range(hidden_units_per_layer, hidden_units_per_layer))),
        'out': tf.Variable(tf.random_uniform([hidden_units_per_layer, 1],
                                            **_weight_init_range(hidden_units_per_layer, 1))),
    }
    out = mlp_out(f_input_layer, mlp_weights)

    return out, mlp_weights


def mlp_out(f_input_layer, mlp_weights):
    mlp_layer_1 = tf.nn.sigmoid(tf.matmul(f_input_layer, mlp_weights['h1']))
    mlp_layer_2 = tf.nn.sigmoid(tf.matmul(mlp_layer_1, mlp_weights['h2']))
    out = tf.nn.sigmoid(tf.matmul(mlp_layer_2, mlp_weights['out']))
    return out
