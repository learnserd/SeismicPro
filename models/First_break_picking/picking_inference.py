""" Inference script that takes segy file and First break picking model,
predicts picking for traces and dump them to csv file.
"""
import sys

import argparse

sys.path.append('../..')

from seismicpro.batchflow import Pipeline, Dataset, B, V, F
from seismicpro.batchflow.models.torch import UNet
from seismicpro.src import SeismicBatch, FieldIndex, TraceIndex, seismic_plot, CustomIndex
from picking_batch import PickingBatch

def make_prediction():
    """ Read the model and data paths and run inference pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_raw', type=str, help="Path to SEGY file.",
                        required=True)
    parser.add_argument('-m', '--path_model', type=str, help="Path to saved model.",
                        required=True)
    parser.add_argument('-d', '--path_dump', type=str, help="Path to csv file where the results are stored.",
                        required=True)
    args = parser.parse_args()
    
    path_raw = args.path_raw
    model = args.path_model
    save_to = args.path_dump
    predict(path_raw, model, save_to)

def predict(path_raw, path_model, path_save_to):
    """Make predictions and dump results using loaded model and path to data.

    Parameters
    ----------
    path_raw: str
        path to SEGY file
    model_path: str
        path to the file with model
    csv_path: str
        path to csv file where the results will be stored
    """
    index = FieldIndex(name='raw', path=path_raw)
    data = Dataset(TraceIndex(index), PickingBatch)

    config_predict = {
        'build': False,
        'load': {'path': path_model},
        'device': 'cpu'
    }

    test_pipeline = (data.p
                        .init_model('dynamic', UNet, 'my_model', config=config_predict) 
                        .load(components='raw', fmt='segy')
                        .drop_zero_traces(num_zero=700, src='raw')
                        .normalize_traces(src='raw', dst='raw')
                        .add_components(components='unet_predictions')
                        .predict_model('my_model', B('raw'), fetches=['predictions'],
                                        save_to=[B('unet_predictions')], mode='a')
                        .mask_to_pick(src='unet_predictions', dst='unet_predictions', labels=False)
                        .dump(src='unet_predictions', fmt='picks', path=path_save_to, traces='raw', to_samples=True))

    test_pipeline.run(1000, n_epochs=1, drop_last=False, shuffle=False, bar=True)

if __name__ == "__main__":
    sys.exit(make_prediction())