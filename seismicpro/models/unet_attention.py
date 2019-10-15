""" UnetAttention model """
import tensorflow as tf

from ..batchflow.batchflow.models.tf import EncoderDecoder
from ..batchflow.batchflow.models.tf.layers import conv_block

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
        
        with tf.variable_scope('main_branch'):
            main = super().body(raw, **{**kwargs, **main_config}) # pylint: disable=not-a-mapping
            
            #Get a single channel with linear activation for the main branch
            main = conv_block(main, layout='c', filters=1, units=1, name='head_main')
            
        with tf.variable_scope('attention_branch'):    
            att = super().body(raw, **{**kwargs, **attn_config}) # pylint: disable=not-a-mapping
            
            #Get a single channel with sigmoid activation for the attention branch
            att = conv_block(att, layout='ca', kernel_size=3, filters=1, units=1,
                             activation=tf.nn.sigmoid, name='head_att')
            
        return main, att, raw

    def head(self, inputs, *args, **kwargs):
        _ = args
        main, att, raw = inputs


        #Quick estimation of sigmoid center location
        att_sum = tf.reduce_sum(att, axis=1, keepdims=True)

        #Define a domain for sigmoid function
        sigm_x = tf.fill(tf.shape(att), 0.0)
        arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
        arange = tf.expand_dims(arange, axis=-1)
        sigm_x = sigm_x - arange

        #Shallow network that estimates sigmoid center location and shoothness
        #based on its quick estimation and offset
#         shift_in = tf.concat([tf.squeeze(att_sum, axis=1), offset], axis=1)
        shift_in = tf.squeeze(att_sum, axis=1)
        with tf.variable_scope('attention_dense'):
            sigmoid_center = conv_block(shift_in, **kwargs)
        self.store_to_attr("sigmoid_center", sigmoid_center)

        #Shift and stretch sigmoid domain based on network estimations
        sigmoid_center = tf.expand_dims(sigmoid_center, axis=-1)
        sigm_x = sigm_x + sigmoid_center[:, :1]

        #Apply sigmoid function to the above obtained domain
        attention_sigmoid = tf.sigmoid(sigm_x)
        self.store_to_attr("attention_sigmoid", attention_sigmoid)

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
