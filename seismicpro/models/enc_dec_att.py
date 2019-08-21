""" Att model """
import tensorflow as tf

from ..batchflow.batchflow.models.tf.layers import conv_block
from ..batchflow.batchflow.models.tf import UNet

class UnetAttEmbOut(UNet):
    """attention model_V2"""
    @classmethod
    def initial_block(cls, inputs, *args, **kwargs):
        x = super().initial_block(inputs, *args, **kwargs)
        return [x, inputs]

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        inputs, raw = inputs
        x = super().body(inputs, name=name, **kwargs)
        return [*x, raw]

    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        x = super().decoder(inputs, name, **kwargs)
        return [x, inputs[-1]]

    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        """ Linear convolutions. """
        inputs, skip, raw = inputs
        with tf.variable_scope(name):
            kwargs = cls.fill_params('head', **kwargs)
            # attention
            mask = conv_block(skip, layout='XcXcca', factor=[10, 15],
                              filters=[512, 128, 1], kernel_size=3,
                              activation=tf.nn.sigmoid, name='mask_branch')

            main = cls.crop(inputs, targets, kwargs['data_format'])
            mask = cls.crop(mask, main, kwargs['data_format'])

            #without training. Predict sigmoid center by mask brach.
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            sigm_x = tf.fill(tf.shape(mask), 0.0)
            arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
            arange = tf.expand_dims(arange, axis=-1)
            sigm_x = sigm_x - arange
            sigm_x += mask_sum
            attention_sigmoid = tf.sigmoid(sigm_x)

            main = conv_block(main, layout='c', filters=1, units=1, name='head_main')

            out_lift = raw * attention_sigmoid + main * (1 - attention_sigmoid)
        return out_lift
