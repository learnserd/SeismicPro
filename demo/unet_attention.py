import numpy as np
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

    for d in range(depth):
        conv = conv_block(conv, layout='caca', filters=filters * (2 ** d),
                          kernel_size=kernel_size, activation=activation,
                          is_training=is_training)
        conv_d.append(conv)
        conv = conv_block(conv, layout='pd', is_training=is_training)

    conv = conv_block(conv, layout='cacad', filters=filters * (2 ** depth),
                      kernel_size=kernel_size, activation=activation,
                      is_training=is_training)

    up = conv

    for d in range(depth, 0, -1):
        up = conv_block(up, 'cad', filters=filters * (2 ** d),
                        activation=activation, kernel_size=kernel_size,
                        transpose=True, is_training=is_training)
        
        concat_shape = conv_d[d - 1].get_shape()
        up = tf.cond(tf.less(concat_shape[1], tf.shape(up)[1]),
                     true_fn=lambda: tf.slice(up, [0, 0, 0], [-1, concat_shape[1], -1]),
                     false_fn=lambda: up)

        up = tf.concat([up, conv_d[d - 1]], axis=-1)
        up = conv_block(up, 'cacad', filters=filters * (2 ** (d - 1)), 
                        kernel_size=kernel_size, activation=activation,
                        is_training=is_training)

    return up

class UNetAttention:  
    def __init__(self, restore=None):
        config = tf.ConfigProto()
        graph = tf.Graph()
        with graph.as_default():
            self.trace_in = tf.placeholder('float', shape=(None, 3000, 1), name='trace_in')
            self.trace_offset = tf.placeholder('float', shape=(None, 1), name='trace_offset')
            self.target = tf.placeholder('float', shape=(None, 3000, 1), name='target')
            self.balance = tf.placeholder('float', name='balance')
            self.learning_rate = tf.placeholder('float', name='learning_rate')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.variable_scope("attention_scope"):
                attention = u_net(self.trace_in, depth=3, filters=8, kernel_size=3,
                                  activation='elu', is_training=self.is_training)
                attention = conv_block(attention, 'c', filters=1, kernel_size=3)
                attention = conv_block(attention, 'ca', filters=1, kernel_size=3,
                                       activation='sigmoid')

                attention_sum = tf.reduce_sum(attention, axis=1)

                sigm_x = tf.fill(tf.shape(attention)[:2], 0.0)
                sigm_x = tf.add(sigm_x, -tf.range(0, tf.cast(tf.shape(attention)[1], 'float'), dtype='float'))

                shift_in = tf.concat([attention_sum, self.trace_offset], axis=1)
                shift_in = tf.layers.dense(shift_in, 16, activation=tf.nn.elu)
                shift_out = tf.layers.dense(shift_in, 2, activation=tf.nn.relu)

            sigm_x = tf.divide(tf.add(sigm_x, shift_out[:, :1]), 1 + shift_out[:, 1:2])

            self.attention_sigmoid = tf.expand_dims(tf.sigmoid(sigm_x), -1)

            with tf.variable_scope("lift_scope"):
                lift_trace = u_net(self.trace_in, depth=5, filters=16, kernel_size=7,
                                   activation='elu', is_training=self.is_training)
                lift_trace = conv_block(lift_trace, 'c', filters=1, kernel_size=3)

            self.predict = (tf.multiply(self.trace_in, self.attention_sigmoid) +
                            tf.multiply(lift_trace, 1 - self.attention_sigmoid))
            self.loss = (tf.losses.absolute_difference(self.target, self.predict) +
                         self.balance * tf.reduce_mean(1 - self.attention_sigmoid))

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            lift_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='lift_scope')
            attention_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope='attention_scope')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.step_attention = optimizer.minimize(self.loss, var_list=attention_vars)
                self.step_lift = optimizer.minimize(self.loss, var_list=lift_vars)
               
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if restore is not None:
                saver.restore(self.sess, restore)
    
    def train_on_batch(self, trace_in, target, trace_offset, balance,
                       is_training=True, learning_rate=0.001):
        res = self.sess.run([self.loss, self.step_attention, self.step_lift],
                             feed_dict={self.trace_in: trace_in,
                                        self.target: target,
                                        self.trace_offset: trace_offset,
                                        self.is_training: is_training,
                                        self.balance: balance,
                                        self.learning_rate: learning_rate})
        return res[0]

    def predict_on_batch(self, trace_in, trace_offset, is_training=False):
        res = self.sess.run([self.predict, self.attention_sigmoid],
                             feed_dict={self.trace_in: trace_in,
                                        self.trace_offset: trace_offset,
                                        self.is_training: is_training})
        return res
    
def make_data_train(batch):
    x = np.expand_dims(np.vstack(batch.raw), -1)
    y = np.expand_dims(np.vstack(batch.lift), -1)
    offset = batch.index.get_df()['offset'].values[:, np.newaxis]
    
    return {'trace_in': x,
            'target': y,
            'trace_offset': offset,
            'is_training': True,
            'balance': 0.05,
            'learning_rate': 0.0001}

def make_data_predict(batch):
    x = np.expand_dims(np.vstack(batch.raw), -1)
    offset = batch.index.get_df()['offset'].values[:, np.newaxis]    
    return {'trace_in': x, 'trace_offset': offset}