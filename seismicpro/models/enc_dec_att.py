""" Att model """
import tensorflow as tf

from ..batchflow.batchflow.models.tf.layers import conv_block
from ..batchflow.batchflow.models.tf import UNet

class UnetAttEmbOut(UNet):
    """attention model_V2"""
    @classmethod
    def initial_block(cls, inputs, *args, **kwargs):
        x = super().initial_block(inputs, *args, **kwargs)
        # Итоговый сингал состоит из преобразованного и входного сигналов
        # поэтому через все блоки пробрасываем исходный сигнал (inputs)
        return [x, inputs]

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        inputs, raw = inputs
        x = super().body(inputs, name=name, **kwargs)
        # тут исходный сингал - raw. Из боди нам возвращается
        # вектор из двух элементов - выход декодера и выход из
        # эмбеддинга (на самом деле это выход энкодера, тк эмбеддинг по дефолту None)
        # Это нужно потому что аттеншен ветку мы строим от эмбеддинга до выхода модели
        return [*x, raw]

    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        x = super().decoder(inputs, name, **kwargs)
        return [x, inputs[-1]]

    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        """ Linear convolutions. """
        inputs, skip, raw = inputs
        #боди нам возвращает 3 элемента: выход декодера, выход эмбеддинга и исходный сигнал
        with tf.variable_scope(name):
            kwargs = cls.fill_params('head', **kwargs)
            # Увеличиваем маску в 16 раз и уменьшаем число каналов до 1.
            mask = conv_block(skip, layout='Xccca', factor=16,
                              filters=[512, 128, 1], kernel_size=3,
                              data_format=kwargs['data_format'],
                              activation=tf.nn.sigmoid, name='mask_branch')
            # кропаем маску и выход декодера
            main = cls.crop(inputs, targets, kwargs['data_format'])
            mask = cls.crop(mask, main, kwargs['data_format'])

            # предсказание маски - количество отсчетов на которые надо ее подвинуть.
            # генерим массив от 0 до -len(mask) и прибавляем к ней mask_sum
            # берем сигмойду и получаем исходную маску
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            sigm_x = tf.fill(tf.shape(mask), 0.0)
            arange = tf.range(0, tf.cast(tf.shape(sigm_x)[1], 'float'), dtype='float')
            arange = tf.expand_dims(arange, axis=-1)
            sigm_x = sigm_x - arange
            sigm_x += mask_sum
            attention_sigmoid = tf.sigmoid(sigm_x)

            main = conv_block(main, layout='c', filters=1, units=1, name='head_main')
            # Выходой сингал состоит из входного сингала с исправленной частью
            out_lift = raw * attention_sigmoid + main * (1 - attention_sigmoid)
        return out_lift
