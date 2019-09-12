""" UnetAttention model """
import tensorflow as tf

from ..batchflow.batchflow.models.tf import EncoderDecoder
from ..batchflow.batchflow.models.tf.layers import conv_block

class Unet(EncoderDecoder):
    """Class for Unet Attention model."""

    @classmethod
    def default_config(cls):
        config = super().default_config()

        body_config = config['body']

        config['body'] = None
        config['body/main'] = body_config

        return config

    def initial_block(self, inputs, *args, **kwargs):
        _ = args, kwargs
        return inputs

    def body(self, inputs, *args, **kwargs):
        _ = args
        raw = inputs

        main_config = kwargs.pop('main')

        main = super().body(raw, name='main', **{**kwargs, **main_config}) # pylint: disable=not-a-mapping
        return main

    def head(self, inputs, *args, **kwargs):
        _ = args, kwargs
        main = inputs

        #Get a single channel with linear activation for the main branch
        main = conv_block(main, layout='c', filters=1, units=1, name='head_main')
        self.store_to_attr("out_main", main)

        return main

class UnetAtt(EncoderDecoder):
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

    def body(self, inputs, *args, **kwargs):
        _ = args
        raw = inputs

        main_config = kwargs.pop('main')
        attn_config = kwargs.pop('attn')

        main = super().body(raw, name='main', **{**kwargs, **main_config}) # pylint: disable=not-a-mapping
        att = super().body(raw, name='attention', **{**kwargs, **attn_config}) # pylint: disable=not-a-mapping
        return main, att, raw

    def head(self, inputs, *args, **kwargs):
        _ = args, kwargs
        main, att, raw = inputs

        #Get a single channel with sigmoid activation for the attention branch
        att = conv_block(att, layout='ca', kernel_size=3, filters=1, units=1,
                         activation=tf.nn.sigmoid, name='head_att')
        self.store_to_attr("out_att", att)

        #Estimation of sigmoid center location
        att_sum = tf.reduce_sum(att, axis=1, keepdims=True)
        self.store_to_attr("att_sum", att_sum)

        #Define a domain for sigmoid function
        sigm_x = tf.fill(tf.shape(att), 0.0)
        arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
        arange = tf.expand_dims(arange, axis=-1)
        sigm_x = sigm_x - arange

        #Shift sigmoid domain
        sigm_x = sigm_x + att_sum

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

        self.store_to_attr("cone_loss", tf.reduce_mean(1 - attention_sigmoid))
        self.store_to_attr("sigm_loss", tf.reduce_mean(tf.abs(attention_sigmoid - att)))

        return tf.stack([out_lift, att, attention_sigmoid], axis=0)

def attention_loss(targets, predictions, alpha, beta, **kwargs):
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
    att = predictions[1]
    attention_sigmoid = predictions[2]

    loss = (tf.reduce_mean(tf.abs(targets - out_lift)) +
            alpha * tf.reduce_mean(1 - attention_sigmoid) +
            beta * tf.reduce_mean(tf.abs(attention_sigmoid - att))
           )
    tf.losses.add_loss(loss)
    return loss
