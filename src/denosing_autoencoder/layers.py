import tensorflow as tf


def fc(input, name, out_dim, non_linear_fn=None):
    """
    Function that takes the input to a layer and applies a non linear function, reducing the
    input to an encoded representation
    Args:
        input: the input dataset that feeds into layer
        name: name of the variable where the current layer is stored
        out_dim: dimension of layer
        non_linear_fn: the function applied to the layer
    Returns:
        layer with applied function
    """
    assert(type(out_dim) == int)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        input_dims = input.get_shape().as_list()
        if len(input_dims) == 4:
            batch_size, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [batch_size, in_dim])
        else:
            in_dim = input_dims[-1]
            batch_size=input_dims[0]
            flat_input = tf.reshape(input, [batch_size, in_dim])
        weights_fc = tf.get_variable('weights', [in_dim, out_dim])
        biases = tf.get_variable('biases', [out_dim])
        out = tf.nn.xw_plus_b(flat_input, weights_fc, biases)
        if non_linear_fn:
            return non_linear_fn(out, name=scope.name)
        else:
            return out

