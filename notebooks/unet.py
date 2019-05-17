import tensorflow as tf

def conv_block(x, layout, filters=None, kernel_size=1,
               transpose=False, rate=0.1, activation=None, is_training=True):
    ci = 0
    try:
        iter(filters)
    except TypeError:
        filters = list([filters] * layout.count('c'))

    for s in layout:
        if s == 'c':
            if transpose:
                x = tf.expand_dims(x, -2)
                x = tf.layers.conv2d_transpose(x, filters[ci], (kernel_size,) + (1,),
                                               strides=(2, 1), padding='same')
                x = tf.squeeze(x, axis=-2)
            else:
                x = tf.layers.conv1d(x, filters[ci], kernel_size, strides=1,
                                     padding='same')
            ci += 1
        elif s == 'a':
            if activation is not None:
                x = getattr(tf.nn, activation)(x)
        elif s == 'p':
            x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='same')
        elif s == 'n':
            x = tf.layers.batch_normalization(x, training=is_training, momentum=0.9)
        elif s == 'd':
            x = tf.layers.dropout(x, rate=rate, training=is_training)
        else:
            raise KeyError('unknown letter {0}'.format(s))
    return x

def u_net(x, depth, filters, kernel_size, activation=None, is_training=True):
    conv_d = []

    conv = x
    print('input', conv.get_shape())

    for d in range(depth):
        conv = conv_block(conv, layout='caca', filters=filters * (2 ** d),
                          kernel_size=kernel_size, activation=activation,
                          is_training=is_training)
        print('conv_block_{0}'.format(d), conv.get_shape())
        conv_d.append(conv)
        conv = conv_block(conv, layout='pd', is_training=is_training)
        print('pool_{0}'.format(d), conv.get_shape())

    conv = conv_block(conv, layout='cacad', filters=filters * (2 ** depth),
                      kernel_size=kernel_size, activation=activation,
                      is_training=is_training)
    print('bottom_conv_block_{0}'.format(depth), conv.get_shape())

    up = conv

    for d in range(depth, 0, -1):
        up = conv_block(up, 'cad', filters=filters * (2 ** d),
                        activation=activation, kernel_size=kernel_size,
                        transpose=True, is_training=is_training)
        
        concat_shape = conv_d[d - 1].get_shape()
        up = tf.cond(tf.less(concat_shape[1], tf.shape(up)[1]),
                     true_fn=lambda: tf.slice(up, [0, 0, 0], [-1, concat_shape[1], -1]),
                     false_fn=lambda: up)

        print('up_{0}'.format(d - 1), up.get_shape())
        up = tf.concat([up, conv_d[d - 1]], axis=-1)
        print('concat_{0}'.format(d), up.get_shape())
        up = conv_block(up, 'cacad', filters=filters * (2 ** (d - 1)), 
                        kernel_size=kernel_size, activation=activation,
                        is_training=is_training)
        print('up_conv_block_{0}'.format(d), up.get_shape())

    return up