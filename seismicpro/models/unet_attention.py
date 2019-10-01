""" UnetAttention model """
import tensorflow as tf

from ..batchflow.batchflow.models.tf import EncoderDecoder
from ..batchflow.batchflow.models.tf.layers import conv_block, combine
from ..batchflow.batchflow.models.utils import unpack_args


class EncoderDecoderWithBranch(EncoderDecoder):
    """ EncoderDecoderWithBranch """
    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and decoder. """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        embeddings = kwargs.get('embedding')
        decoder = kwargs.pop('decoder')

        with tf.variable_scope(name):
            # Encoder: transition down
            if encoder is not None:
                encoder_args = {**kwargs, **encoder}
                encoder_outputs = cls.encoder(inputs, name='encoder', **encoder_args)
            else:
                encoder_outputs = [inputs]
            x = encoder_outputs[-1]

            # Bottleneck: working with compressed representation via multiple steps of processing
            if embeddings is not None:
                embeddings = embeddings if isinstance(embeddings, (tuple, list)) else [embeddings]

                for i, embedding in enumerate(embeddings):
                    embedding_args = {**kwargs, **embedding}
                    x = cls.embedding(x, name='embedding-'+str(i), **embedding_args)
            encoder_outputs[-1] = x

            # Decoder: transition up
            if decoder is not None:
                decoder_args = {**kwargs, **decoder}
                x = cls.decoder(encoder_outputs[::-1], name='decoder', **decoder_args)
        return x

    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        steps = kwargs.pop('num_stages') or len(inputs)-1
        factor = kwargs.pop('factor') or [2]*steps
        skip, order, upsample, block_args = cls.pop(['skip', 'order', 'upsample', 'blocks'], kwargs)
        order = ''.join([item[0] for item in order])
        base_block = block_args.get('base')

        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor] * steps
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        with tf.variable_scope(name):
            x = inputs[0]

            for i in range(steps):
                with tf.variable_scope('decoder-'+str(i)):
                    # Skip some of the steps
                    if factor[i] == 1:
                        continue

                    # Make all the args
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}
                    upsample_args = {'factor': factor[i],
                                     **kwargs, **upsample, **unpack_args(upsample, i, steps)}

                    combine_op = args.get('combine_op')
                    combine_args = {'op': combine_op if isinstance(combine_op, str) else '',
                                    'data_format': args.get('data_format'),
                                    **(combine_op if isinstance(combine_op, dict) else {}),
                                    **(skip if isinstance(skip, dict) else {})}

                    for letter in order:
                        if letter == 'b':
                            x = base_block(x, name='block', **args)
                        elif letter in ['u']:
                            if upsample.get('layout') is not None:
                                x = cls.upsample(x, name='upsample', **upsample_args)
                        elif letter == 'c':
                            # Combine result with the stored encoding of the ~same shape
                            if (skip or isinstance(skip, dict)) and (i < steps):
                                x = cls.crop(x, inputs[i+1], data_format=kwargs.get('data_format'))
                                x = combine([x, inputs[i+1]], **combine_args)
                        else:
                            raise ValueError('Unknown letter in order {}, use one of ("b", "u", "c")'.format(letter))

        return x

    @classmethod
    def branch(cls, inputs, name='branch', **kwargs):
        """ Branch, consisting of blocks """
        steps = kwargs.pop('num_stages') or len(inputs) - 1

        order, downsample, block_args = cls.pop(['order', 'downsample', 'blocks'], kwargs, default=None)
        skip, upsample = cls.pop(['skip', 'upsample'], kwargs, default=None)
        order = ''.join([item[0] for item in order])

        x = inputs[0]
        branch_outputs = []

        with tf.variable_scope(name):
            for i in range(steps):
                with tf.variable_scope('block-' + str(i)):
                    # Make all the args
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}

                    for letter in order:
                        if letter == 'b':

                            base_block = block_args.get('base')
                            x = base_block(x, name='block', **args)
                        elif letter == 's':
                            branch_outputs.append(x)
                        elif letter in ['d', 'p']:
                            downsample_args = {**kwargs, **downsample, **unpack_args(downsample, i, steps)}
                            if downsample.get('layout') is not None:
                                x = conv_block(x, name='downsample', **downsample_args)
                        elif letter in ['u']:
                            factor = kwargs.get('factor')
                            if factor is None:
                                factor = 2
                            elif isinstance(factor, int):
                                factor = int(factor ** (1 / steps))
                            elif isinstance(factor, list):
                                factor = factor[i]
                            else:
                                raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

                            upsample_args = {**kwargs, 'factor': factor,
                                             **upsample, **unpack_args(upsample, i, steps)}
                            if upsample.get('layout') is not None:
                                x = cls.upsample(x, name='upsample', **upsample_args)
                        elif letter == 'c':
                            combine_op = args.get('combine_op')
                            combine_args = {'op': combine_op if isinstance(combine_op, str) else '',
                                            'data_format': args.get('data_format'),
                                            **(combine_op if isinstance(combine_op, dict) else {}),
                                            **(skip if isinstance(skip, dict) else {})}
                            # Combine result with the stored encoding of the ~same shape
                            if (skip or isinstance(skip, dict)) and (i < steps):
                                x = cls.crop(x, inputs[i + 1], data_format=kwargs.get('data_format'))
                                x = combine([x, inputs[i + 1]], **combine_args)
                        else:
                            raise ValueError('Unknown letter in order {}, use one of "b", "d", "p", "s"'
                                             .format(letter))

        branch_outputs.append(x)
        return branch_outputs


class UnetAtt(EncoderDecoderWithBranch):
    """Class for Unet Attention model."""

    @classmethod
    def default_config(cls):
        config = super().default_config()

        body_config = config['body']

        config['body'] = None
        config['body/main'] = body_config
        config['body/attn'] = body_config

        return config

    def initial_block(self, inputs, *args, **kwargs):
        _ = args, kwargs
        return inputs

    def body(self, inputs, name='body', *args, **kwargs):
        _ = args
        raw, offset = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('attn')

        main = super().body(raw, name='main', **{**kwargs, **main_config}) # pylint: disable=not-a-mapping
        att = super().body(raw, name='attention', **{**kwargs, **attn_config}) # pylint: disable=not-a-mapping
        return main, att, raw, offset

    def head(self, inputs, *args, **kwargs):
        _ = args, kwargs
        main, att, raw, offset = inputs

        #Get a single channel with sigmoid activation for the attention branch
        att = conv_block(att, layout='ca', kernel_size=3, filters=1, units=1,
                         activation=tf.nn.sigmoid, name='head_att')

        #Quick estimation of sigmoid center location
        att_sum = tf.reduce_sum(att, axis=1, keepdims=True)

        #Define a domain for sigmoid function
        sigm_x = tf.fill(tf.shape(att), 0.0)
        arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
        arange = tf.expand_dims(arange, axis=-1)
        sigm_x = sigm_x - arange

        #Shallow network that estimates sigmoid center location and shoothness
        #based on its quick estimation and offset
        shift_in = tf.concat([tf.squeeze(att_sum, axis=1), offset], axis=1)
        shift_in = tf.layers.dense(shift_in, 16, activation=tf.nn.elu)
        shift_in = tf.layers.dense(shift_in, 16, activation=tf.nn.elu)
        sigmoid_center = tf.layers.dense(shift_in, 1, activation=tf.nn.relu)
        self.store_to_attr("sigmoid_center", sigmoid_center)

        #Shift and stretch sigmoid domain based on network estimations
        sigmoid_center = tf.expand_dims(sigmoid_center, axis=-1)
        sigm_x = sigm_x + sigmoid_center[:, :1]

        #Apply sigmoid function to the above obtained domain
        attention_sigmoid = tf.sigmoid(sigm_x)
        self.store_to_attr("attention_sigmoid", attention_sigmoid)

        #Get a single channel with linear activation for the main branch
        main = conv_block(main, layout='c', filters=1, units=1, name='head_main')
        self.store_to_attr("out_main", main)

        #Get a model output that is a superposition of raw input and main branches
        #according to attention mask
        out_lift = raw * attention_sigmoid + main * (1 - attention_sigmoid)
        self.store_to_attr("out_lift", out_lift)

        return tf.stack([out_lift, attention_sigmoid], axis=0)


def attention_loss(targets, predictions, balance, **kwargs):
    """Loss function for Unet Attention model

    Parameters
    ----------
    targets : tensor
        Target values.
    predictions : tensor
        Predicted values.
    balance : tensor
        Balance coeffitient between L1 loss and attention mask area.

    Returns
    -------
    loss : tensor
        Computed loss.
    """
    _ = kwargs
    out_lift = predictions[0]
    attention_sigmoid = predictions[1]
    loss = (tf.losses.absolute_difference(targets, out_lift) +
            balance * tf.reduce_mean(1 - attention_sigmoid))
    tf.losses.add_loss(loss)
    return loss
