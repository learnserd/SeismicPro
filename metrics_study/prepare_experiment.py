""" Utilities for preparing blind test"""
# pylint:disable=invalid-name

import os
import shutil
import random
from functools import reduce
import csv

import numpy as np

from seismicpro.batchflow import Dataset, Pipeline
from seismicpro.src import SeismicBatch, FieldIndex


class Parameters:
    """ Input data for experiment """
    def __init__(self, models_paths, raw_path, fields):
        self.models_paths = models_paths
        self.raw_path = raw_path
        self.fields = fields
        self.num_models = len(self.models_paths)
        self.model_names = ['M_' + str(i) for i in range(self.num_models)]
        self.num_fields = len(self.fields)


base_path = '/datasets/multiple_lift_4_metrics_validation/QC/'
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
    if os.path.exists(res_path):
        shutil.rmtree(res_path)

    field_index = reduce(lambda x, y: x.merge(y),
                         (FieldIndex(name=name, path=path, extra_headers=['offset'])
                          for name, path in zip(params.model_names, params.models_paths)))
    field_index = field_index.merge(FieldIndex(name='raw',
                                               path=params.raw_path, extra_headers=['offset']))
    field_index = field_index.create_subset(np.asarray(params.fields))

    components = tuple(params.model_names + ['raw'])
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
        for name, path in zip(params.model_names, params.models_paths):
            f.write('{},"{}"\n'.format(name, path))


def shuffle_folders(path_to_fields, path_to_res, num_experts, params, archive=False):
    """
    prepare files for each expert individually
    model aliases are shuffled
    """
    if os.path.exists(path_to_res):
        shutil.rmtree(path_to_res)

    mapping = {}
    for i in range(num_experts):
        expert_name = 'Expert_{}'.format(i)
        mapping[expert_name] = {}

        expert_path = os.path.join(path_to_res, expert_name)
        os.makedirs(expert_path)

        for fi in params.fields:
            field_name = 'F_{}'.format(fi)
            field_dir = os.path.join(expert_path, field_name)
            os.makedirs(field_dir)
            mapping[expert_name][field_name] = {}

            raw_old_path = os.path.join(path_to_fields, 'raw', '{}.sgy'.format(fi))
            raw_new_path = os.path.join(field_dir, 'raw_{}.sgy'.format(fi))
            shutil.copyfile(raw_old_path, raw_new_path)

            new_indices = random.sample(range(params.num_models), k=params.num_models)
            for old_idx, new_idx in enumerate(new_indices):
                model_name = params.model_names[old_idx]
                field_path_old = os.path.join(path_to_fields, model_name, '{}.sgy'.format(fi))
                field_name_new = 'M_{}_{}.sgy'.format(fi, new_idx)
                field_path_new = os.path.join(field_dir, field_name_new)

                # debugging
                # print(field_path_old)
                # print("->")
                # print(field_path_new)
                # print()

                shutil.copyfile(field_path_old, field_path_new)

                mapping[expert_name][field_name][field_name_new] = params.models_paths[old_idx]

        if archive:
            shutil.make_archive(expert_path, 'zip', expert_path)

    # print mapping to file
    with open(os.path.join(path_to_res, 'description.csv'), 'w') as csv_file:
        csvwriter = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(['expert_name', 'field_name', 'model_name', 'true_model'])
        for expert_name in mapping:
            for field_name in mapping[expert_name]:
                for model_name in sorted(mapping[expert_name][field_name]):
                    csvwriter.writerow([expert_name,
                                        field_name,
                                        model_name,
                                        mapping[expert_name][field_name][model_name]
                                        ])


if __name__ == '__main__':
    test_res_path = "test_res"
    split_folders(test_res_path, test_params)
    shuffle_folders(test_res_path, "experts", num_experts=3, params=test_params, archive=True)
