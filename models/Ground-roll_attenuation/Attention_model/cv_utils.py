""" Utilities for cross-validation"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sh

sys.path.append('../../..')

from seismicpro.batchflow import Dataset, DatasetIndex, Pipeline, V, B, F

from seismicpro.src import SeismicBatch, FieldIndex, TraceIndex, merge_segy_files
from seismicpro.models import UnetAtt, attention_loss, FieldMetrics


def exp_stack(x):
    """Data stacking."""
    return np.expand_dims(np.vstack(x), -1)

def train_model(path_raw, path_lift, train_fields_number=10, batch_size=64, n_epochs=3,
                offset_lim=None, tsize=3000, scale_coef=1):
    """Train attention model."""

    index = (FieldIndex(name='raw', extra_headers=['offset'], path=path_raw)
             .merge(FieldIndex(name='lift', path=path_lift)))

    tindex = TraceIndex(index.create_subset(index.indices[:train_fields_number]))
    if offset_lim is not None:
        tindex = tindex.filter('offset', lambda x: x < offset_lim)

    train_set = Dataset(tindex, SeismicBatch)

    def make_data(batch, **kwagrs):
        """Make data train."""
        return {"feed_dict": {'trace_raw': exp_stack(batch.raw) * scale_coef,
                              'offset': np.vstack(batch.trace_headers('offset')),
                              'lift': exp_stack(batch.lift) * scale_coef}}

    model_config = {
        'initial_block/inputs': ['trace_raw', 'offset'],
        'inputs': dict(trace_raw={'shape': (None, 1)},
                       offset={'shape': (1, )},
                       lift={'name': 'targets', 'shape': (None, 1)}),

        'loss': (attention_loss, {'balance': 0.05}),
        'optimizer': ('Adam', {'learning_rate': 0.001}),
        'main_config': {'filters': 2 * np.array([8, 16, 32, 64, 128]),
                        'data_format': "channels_last",
                        'encoder': dict(layout='caca', kernel_size=7, activation=tf.nn.elu),
                        'downsample': dict(layout='pd', pool_size=2, pool_strides=2, dropout_rate=0.05),
                        'decoder': dict(layout='caca', kernel_size=7, activation=tf.nn.elu),
                        'upsample': dict(layout='tad', kernel_size=7, strides=2,
                                         dropout_rate=0.05, activation=tf.nn.elu),},
        'attn_config': {'filters': [8, 16, 32, 64],
                        'data_format': "channels_last",
                        'encoder': dict(layout='caca', kernel_size=3, activation=tf.nn.elu),
                        'downsample': dict(layout='pd', pool_size=2, pool_strides=2, dropout_rate=0.05),
                        'decoder': dict(layout='caca', kernel_size=3, activation=tf.nn.elu),
                        'upsample': dict(layout='tad', kernel_size=3, strides=2,
                                         dropout_rate=0.05, activation=tf.nn.elu),},
    }

    train_pipeline = (Pipeline()
                      .init_model('dynamic', UnetAtt, name='unet', config=model_config)
                      .init_variable('loss', init_on_each_run=list)
                      .load(components=('raw', 'lift'), fmt='segy', tslice=np.arange(tsize))
                      .train_model('unet', make_data=make_data, fetches='loss', save_to=V('loss', 'a'))
                     )

    train_pipeline = train_pipeline << train_set
    train_pipeline = train_pipeline.run(batch_size=batch_size, n_epochs=n_epochs, drop_last=True,
                                        shuffle=True, bar=True)

    return train_pipeline


def inference_model(path_raw, train_pipeline, tmp_dump_path, output_path,
                    batch_size=2500, tsize=3000, scale_coef=1):
    """Make predictions with attention model and dump result."""

    inference_index = TraceIndex(name='raw', extra_headers='all', path=path_raw)

    def make_data_inference(batch, **kwagrs):
        """Make data for inference."""
        return {"feed_dict": {'trace_raw': exp_stack(batch.raw) * scale_coef,
                              'offset': np.vstack(batch.trace_headers('offset'))}}

    inference_ppl = (Pipeline()
                     .import_model('unet', train_pipeline)
                     .init_variable('res')
                     .init_variable('count', init_on_each_run=0)
                     .load(components='raw', fmt='segy', tslice=np.arange(tsize))
                     .predict_model('unet', make_data=make_data_inference,
                                    fetches=['out_lift'], save_to=B('raw'))
                     .dump(path=F(lambda _, x, **kwargs: os.path.join(tmp_dump_path, str(x) + '.sgy'))(V('count')),
                           src='raw', fmt='segy', split=False)
                     .update_variable('count', F(lambda _, x, **kwargs: x + 1)(V('count'))))

    inference_set = Dataset(inference_index,  SeismicBatch)
    inference_ppl = inference_ppl << inference_set
    inference_ppl.run(batch_size, n_epochs=1, drop_last=False, shuffle=False, bar=True)
    print("Merging files")
    merge_segy_files(output_path=output_path, extra_headers='all',
                     path=os.path.join(tmp_dump_path, '*.sgy'))

    sh.rm(sh.glob(os.path.join(tmp_dump_path, '*')))


def evaluate(path_ml, path_lift, output_filename, tsize=3000, scale_coef=1, model_path=None):
    """Evaluate model across various metrics."""

    m_index = (FieldIndex(name='ml', extra_headers=['SourceX', 'SourceY', 'offset'], path=path_ml)
               .merge(FieldIndex(name='lift', path=path_lift)))

    dset = Dataset(m_index, SeismicBatch)

    def get_pos(batch, *args):
        """Keep source positions."""
        sx = batch.trace_headers('SourceX', flatten=True)[0]
        sy = batch.trace_headers('SourceY', flatten=True)[0]
        return sx, sy

    def eval_metrics(batch, *args, ratio=0.1):
        """Evaluate metrics on arrays."""
        ntraces = batch.lift[0].shape[0]
        tmax = int(ratio * ntraces)
        mt = FieldMetrics(batch.lift[0][:tmax] * scale_coef, batch.ml[0][:tmax])
        return mt.mae(), mt.corrcoef(), mt.wspec()

    metr_pipeline = (Pipeline()
                     .init_variable('metrics', init_on_each_run=list())
                     .init_variable('pos', init_on_each_run=list())
                     .load(components=('ml', 'lift'), fmt='segy', tslice=np.arange(tsize))
                     .call(get_pos, save_to=V('pos', mode='a'))
                     .call(eval_metrics, save_to=V('metrics', mode='a')))

    metr_pipeline = metr_pipeline << dset
    metr_pipeline = metr_pipeline.run(batch_size=1, n_epochs=1, drop_last=False,
                                      shuffle=False, bar=True)

    pos = np.vstack(metr_pipeline.get_variable('pos'))
    metrics = np.vstack(metr_pipeline.get_variable('metrics'))

    np.savez(output_filename, pos=pos, metrics=metrics)


def show_cv(keys, path='./'):
    """Plot cross-validation matrix."""
    cv = np.zeros((len(keys), len(keys), 3))
    for i in range(len(keys)):
        for j in range(len(keys)):
            arrs = np.load('metrics_{}_{}.npz'.format(keys[i], keys[j]))
            cv[i, j] = np.mean(arrs['metrics'], axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    titles = ['L1', 'Corrcoef', 'Spectral']
    for i in range(3):
        im = ax[i].matshow(cv[:, :, i])
        ax[i].set_xticks(range(len(keys)))
        ax[i].set_xticklabels(keys, fontsize=14)
        ax[i].set_yticks(range(len(keys)))
        ax[i].set_yticklabels(keys, fontsize=14)
        ax[i].set_ylabel('Train', fontsize=14)
        ax[i].set_xlabel('Validation', fontsize=14)
        ax[i].set_title(titles[i], fontsize=18, y=1.1)
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
