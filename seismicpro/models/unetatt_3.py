""" Unet Attention iss3: common encoder and skips """

import tensorflow as tf

from . import UnetAtt
from ..batchflow.batchflow.models.tf import EncoderDecoder


class UnetAttention3(UnetAtt):
    """ Unet Attention iss3: common encoder and skips """

    @classmethod
    def default_config(cls):
        config = EncoderDecoder.default_config()

        config['body/encoder/num_stages'] = 4
        config['body/encoder/order'] = ['block', 'skip', 'downsampling']
        config['body/encoder/blocks'] += dict(layout='cna cna', kernel_size=3, filters=[64, 128, 256, 512])
        config['body/embedding'] += dict(layout='cna cna', kernel_size=3, filters=1024)
        config['body/decoder/order'] = ['upsampling', 'combine', 'block']
        config['body/decoder/blocks'] += dict(layout='cna cna', kernel_size=3, filters=[512, 256, 128, 64])

        config['body/attention'] = config['body/decoder']

        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and 2 decoder branches: main and attention """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        embeddings = kwargs.get('embedding')
        decoder = kwargs.pop('decoder')
        attention = kwargs.pop('attention')

        raw, offset = inputs

        with tf.variable_scope(name):
            # Encoder: transition down
            if encoder is not None:
                encoder_args = {**kwargs, **encoder}
                encoder_outputs = cls.encoder(raw, name='encoder', **encoder_args)
            else:
                encoder_outputs = [raw]
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
                main = cls.decoder(encoder_outputs[::-1], name='decoder', **decoder_args)

            # Attention: transition up
            if attention is not None:
                attention_args = {**kwargs, **attention}
                att = cls.decoder(encoder_outputs[::-1], name='attention', **attention_args)

        return main, att, raw, offset


class UnetAttention4(UnetAtt):
    """ Unet Attention iss3: common encoder and skips """

    @classmethod
    def default_config(cls):
        config = EncoderDecoder.default_config()

        config['body/encoder/num_stages'] = 4
        config['body/encoder/order'] = ['block', 'skip', 'downsampling']
        config['body/encoder/blocks'] += dict(layout='cna cna', kernel_size=3,
                                              filters=[64, 128, 256, 512])

        config['body/embedding'] += dict(layout='cna cna', kernel_size=3, filters=1024)

        config['body/decoder/order'] = ['upsampling', 'combine', 'block']
        config['body/decoder/blocks'] += dict(layout='cna cna', kernel_size=3,
                                              filters=[512, 256, 128, 64])

        config['body/attention'] = config['body/decoder']

        config['body/attention/order'] = ['upsampling', 'combine', 'skip', 'block']
        config['body/attention/blocks'] += dict(layout='cna cna', kernel_size=3,
                                                filters=[512, 256, 128, 64])

        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and 2 decoder branches: main and attention """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        embeddings = kwargs.get('embedding')
        decoder = kwargs.pop('decoder')
        attention = kwargs.pop('attention')

        raw, offset = inputs

        with tf.variable_scope(name):
            # Encoder: transition down
            if encoder is not None:
                encoder_args = {**kwargs, **encoder}
                encoder_outputs = cls.branch([raw], name='encoder', **encoder_args)
            else:
                encoder_outputs = [raw]
            x = encoder_outputs[-1]

            # Bottleneck: working with compressed representation via multiple steps of processing
            if embeddings is not None:
                embeddings = embeddings if isinstance(embeddings, (tuple, list)) else [embeddings]

                for i, embedding in enumerate(embeddings):
                    embedding_args = {**kwargs, **embedding}
                    emb = cls.embedding(x, name='embedding-'+str(i), **embedding_args)
            encoder_outputs[-1] = emb

            # Attention: transition up
            if attention is not None:
                attention_args = {**kwargs, **attention}
                att = cls.branch(encoder_outputs[::-1], name='attention', **attention_args)

            # Decoder: transition up
            if decoder is not None:
                decoder_args = {**kwargs, **decoder}
                main = cls.branch([emb] + att[:-1], name='decoder', **decoder_args)

        return main[-1], att[-1], raw, offset
