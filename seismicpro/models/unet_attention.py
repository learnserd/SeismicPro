""" HMModel """
import tensorflow as tf

from ..batchflow.batchflow.models.tf import UNet

class UnetAtt(UNet):
    def body(self, inputs, *args, **kwargs):
        main_config = self.config['main_config']
        attn_config = self.config['attn_config']
        raw, offset = inputs
        main = super().body(raw, name='main', **main_config)
        att = super().body(raw, name='attention', **attn_config)
        return main, att, raw, offset

    def head(self, inputs, *args, **kwargs):
        main, att, raw, offset = inputs

        att = conv_block(att, layout='ca', kernel_size=3, filters=1, units=1,
                         activation=tf.nn.sigmoid, name='head_att')
        main = conv_block(main, layout='c', filters=1, units=1, name='head_main')
        self.store_to_attr("out_main", main)

        att_sum = tf.reduce_sum(att, axis=1, keepdims=True)

        sigm_x = tf.fill(tf.shape(att), 0.0)
        arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
        arange = tf.expand_dims(arange, axis=-1)
        sigm_x = sigm_x - arange

        shift_in = tf.concat([tf.squeeze(att_sum, axis=1), offset], axis=1)
        shift_in = tf.layers.dense(shift_in, 16, activation=tf.nn.elu)
        sigmoid_params = tf.layers.dense(shift_in, 2, activation=tf.nn.relu)
        self.store_to_attr("sigmoid_params", sigmoid_params)

        sigmoid_params = tf.expand_dims(sigmoid_params, axis=-1)
        sigm_x = (sigm_x + sigmoid_params[:, :1]) / (1 + sigmoid_params[:, 1:2])

        attention_sigmoid = tf.sigmoid(sigm_x)
        self.store_to_attr("attention_sigmoid", attention_sigmoid)
        out_lift = raw * attention_sigmoid + main * (1 - attention_sigmoid)
        self.store_to_attr("out_lift", out_lift)
        return tf.stack([out_lift, attention_sigmoid], axis=0)
