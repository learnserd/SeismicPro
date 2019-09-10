[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://python.org)
[![Run Status](https://api.shippable.com/projects/5d5e601aed1bf40007051a93/badge?branch=master)](https://app.shippable.com/github/gazprom-neft/SeismicPro)

# SeismicPro

Machine learning for field seismic data processing.

Content
=================

* [About](#About)
* [Installation](#Installation)
* [Tutorials](#Tutorials)
* [File formats](#File-formats)
	* [Seismic data](#Seismic-data)
	* [SPS data](#SPS-data)
	* [Picking data](#Picking-data)
* [Datasets](#Datasets)
* [Models](#Models)
* [Installation](#Installation)
* [Literature](#Literature)
* [Citing](#Citing-SeismicPro)

## About

SeismicPro provides a framework for machine learning on field seismic data.

## Installation

```
git clone --recursive https://github.com/gazprom-neft/SeismicPro.git
```
## Tutorials

A set of IPython Notebooks introduces step-by-step the SeismicPro framework:

1. [Index](tutorials/1.Index.ipynb) explains how to index data with respect to traces, field records, shot points etc.
2. [Batch](tutorials/2.Batch.ipynb) shows how to load data, perform various actions with seismic traces and visualize them.
3. [Dataset](tutorials/3.Dataset.ipynb) describes how to calculate some parameters for all dataset.
4. [Models](tutorials/4.Models.ipynb) notebook shows how to build and run pipelines for model training, inference and evaluation with respect to ground-roll noise attenuation problem.

## File formats

### Seismic data

Seismic data are expected to be in SEG-Y format.

### SPS data

SPS data are expected as R, S, X text files in csv (comma-separated-values) format with required and optional headers:
* Required R file headers: **rline**, **rid**, **x**, **y**, **z**.
* Required S file headers: **sline**, **sid**, **x**, **y**, **z**.
* Required X file headers: **FieldRecord**, **sline**, **sid**, **from_channel**, **to_channel**, **from_recaiver**, **to_receiver**.

### Picking data

File with first-break picking data is expected to be in csv (comma-separated-values) format with columns **FieldRecord**, **TraceNumber**, **FIRST_BREAK_TIME**.

## Datasets

|Problem|Number of datasets|Number of fields|
|---|---|---|
|[Ground-roll attenuation](datasets/ground-roll_attenuation.ipynb)| 3 | 551, 991, 628 
|[First-break picking](datasets/first_break_picking.ipynb)| 3 | 1001, 1001, 460
|[Spherical divergence correction](datasets/spherical_divergence_correction.ipynb) | 1 | 10

## Models

|Model|Architecture|Metrics|
|---|---|---|
|[Ground-roll attenuation](models/Ground-roll_attenuation/Unet_1D_model/model_description.ipynb)| U-Net 1D| 0.004 MAE for dataset 1
|[Ground-roll attenuation](models/Ground-roll_attenuation/Attention_model/model_description.ipynb)| U-Net Attention 1D| 0.007 MAE for dataset 1
|[First-break picking](models/First_break_picking/1d_CNN/model_description.ipynb)| U-Net 1D | 0.06 MAE for dataset 1 <br/> 0.7 MAE for dataset 2 <br/> 15.9 MAE for dataset 3
|[First-break picking](models/First_break_picking/Coppen's_unsupervised_method/model_description.ipynb)| Coppen's analytical method | 7.57 MAE for dataset 1 <br/> 7.19 MAE for dataset 2 <br/> 12.6 MAE for dataset 3
|[First-break picking](models/First_break_picking/Hidden_Markov_model/model_description.ipynb)| Hidden Markov model | 2.6 MAE for dataset 1 <br/> 23.4 MAE for dataset 2 <br/> 8.0 MAE for dataset 3
|[Spherical divergence correction](models/Spherical_divergence_correction/model_description.ipynb) | Time and speed based method | 0.0017 Derivative metric

## Installation

> `SeismicPro` module is in the beta stage. Your suggestions and improvements are very welcome.

> `SeismicPro` supports python 3.5 or higher.

### Installation as a python package

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/SeismicPro.git#egg=SeismicPro

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/SeismicPro.git

After that just import `seismicpro`:
```python
import seismicpro
```

### Installation as a project repository

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``batchflow`` submodule is also cloned.

    git clone --recursive https://github.com/gazprom-neft/SeismicPro.git

## Literature

Some articles related to seismic data processing:
* [Deep learning tutorial for denoising](https://arxiv.org/pdf/1810.11614.pdf)
* [Seismic images construction](http://lserv.deg.gubkin.ru/file.php?file=../../1/dfwikidata/Voskresenskij.JU.N.Postroenie.sejsmicheskih.izobrazhenij.%28M,.RGUNG%29%282006%29%28T%29_GsPs_.pdf)
* [Difraction](https://mospolytech.ru/storage/43ec517d68b6edd3015b3edc9a11367b/files/LRNo93.pdf)
* [Automatic first-breaks picking: New strategies and algorithms](https://www.researchgate.net/publication/249866374_Automatic_first-breaks_picking_New_strategies_and_algorithms)

## Citing SeismicPro

Please cite SeismicPro in your publications if it helps your research.

    Khudorozhkov R., Illarionov E., Broilovskiy A., Kalashnikov N., Podvyaznikov D. SeismicPro library for seismic data processing and ML models training and inference. 2019.

```
@misc{seismicpro_2019,
  author       = {R. Khudorozhkov and E. Illarionov and A. Broilovskiy and N. Kalashnikov and D. Podvyaznikov},
  title        = {SeismicPro library for seismic data processing and ML models training and inference},
  year         = 2019
}
```
