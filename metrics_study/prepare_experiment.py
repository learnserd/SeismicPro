""" Utilities for preparing blind test"""
# pylint:disable=invalid-name

import os
import shutil
# import random
from functools import reduce
from collections import namedtuple

import numpy as np

from seismicpro.batchflow import Dataset, Pipeline
from seismicpro.src import SeismicBatch, FieldIndex

Parameters = namedtuple('Parameters', 'models_paths raw_path fields')

base_path = '/home/antonina/winhome/datasets/multiple_lift_4_metrics_validation/QC/'

test_params = Parameters(
    models_paths=[base_path + p for p in [
        '1_lift.sgy',
        '1_1dUnet_nomask.sgy',
    ]],
    raw_path=base_path + '1_raw.sgy',
    fields=[108840, 108844, 108846]
)


def split_folders(res_path, params):
    """
    split sgy filed by fields, encode model names
    resulting file tree:

    res_path
    |
    +- raw
    |    |
    |    +- i1.sgy
    |    +- i2.sgy
    |    +- ...
    +- M_1
    |    |
    ...  +- i1.sgy
         +- ...
    """
    num_models = len(params.models_paths)
    model_names = ['M_' + str(i) for i in range(num_models)]

    if os.path.exists(res_path):
        shutil.rmtree(res_path)

    field_index = reduce(lambda x, y: x.merge(y),
                         (FieldIndex(name=name, path=path, extra_headers=['offset'])
                          for name, path in zip(model_names, params.models_paths)))
    field_index = field_index.merge(FieldIndex(name='raw',
                                               path=params.raw_path, extra_headers=['offset']))
    field_index = field_index.create_subset(np.asarray(params.fields))

    components = tuple(model_names + ['raw'])
    p = (Pipeline(dataset=Dataset(field_index, SeismicBatch))
         .load(fmt='segy', components=components)
         .sort_traces(src=components, dst=components, sort_by='offset')
         )

    for c in components:
        os.makedirs(os.path.join(res_path, c))

    p = reduce(lambda x, comp: x.dump(src=comp, fmt='sgy', path=os.path.join(res_path, comp), split=True),
               components, p)

    p.run(batch_size=len(field_index), n_epochs=1)

    # print model paths and model aliases correspondence to file
    with open(os.path.join(res_path, 'description.txt'), 'w') as f:
        for name, path in zip(model_names, params.models_paths):
            f.write('{},"{}"\n'.format(name, path))




# def shuffle_folders(path_to_fields, path_to_res, num_experts, params):
#     mapping = {}
#     for i in range(num_experts):
#         expert_name = 'Expert_{}'.format(i)
#         mapping[expert_name] = {}
#
#         expert_path = path_to_res + expert_name
#         os.makedirs(expert_path)
#
#         for field_no in params.fields:
#             field_name = 'F_{}'.format(field_no)
#             field_path_new = expert_path + field_name
#             field_path_old = path_to_fields + field_name
#             os.makedirs(field_path_new)
#
#             mapping[expert_name][field_name] = {}
#
#             num_models = len(params.models_paths)
#             new_indices = random.sample(range(num_models), k=num_models)
#             for old_idx, new_idx in enumerate(new_indices):
#
#                 model_name_old = 'M_{}_{}.sgy'.format(field_no, old_idx)
#                 model_name_new = 'M_{}_{}.sgy'.format(field_no, new_idx)
#                 model_path_old = field_path_old + model_name_old
#                 model_path_new = field_path_new + model_name_new
#
#                 os.copy(model_path_old, model_path_new)
#                 os.copy(params.raw_path, field_path_new + 'raw.sgy')
#
#                 mapping[expert_name][field_name][new_idx] = params.models_paths[old_idx]
#
#     # print mapping to file
#
#     pass


if __name__ == '__main__':
    test_res_path = "test_res"
    split_folders(test_res_path, test_params)
