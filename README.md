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
* [Literature](#Literature)

## About

SeismicPro provides a framework for machine learning on field seismic data.


## Installation

```
git clone --recursive https://github.com/analysiscenter/SeismicPro.git
```
## Tutorials

A set of IPython Notebooks introduces step-by-step the SeismicPro framework:

1. [Index](tutorials/1.Index.ipynb) explains how to index data with respect to traces, field records, shot points etc.
2. [Batch](tutorials/2.Batch.ipynb) shows how to load data, perform various actions with seismic traces and visualize them.
3. [Models](tutorials/3.Models.ipynb) notebook shows how to build and run pipelines for model training, inference and evaluation with respect to ground-roll noise attenuation problem.


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

|Problem|Number of datasets|Datasets description|
|---|---|---|
|Ground-roll attenuation| 3| In [notebook](datasets/ground-roll_attenuation.ipynb) 
|First-break picking| 3 | In [notebook](datasets/first_break_picking.ipynb)
|Spherical divergence correction | 1 | In [notebook](datasets/spherical_divergence_correction.ipynb)


## Models

|Model|Architecture|Dataset|Metrics|
|---|---|---|---|
|[Ground-roll attenuation](models/Ground-roll_attenuation/Unet_1D_model/model_description.ipynb)| U-Net 1D| Datasets 1 for NA| 0.004 MAE
|[Ground-roll attenuation](models/Ground-roll_attenuation/Attention_model/model_description.ipynb)| U-Net Attention 1D| Datasets 1 for NA | 0.007 MAE
|[First-break picking](models/First_break_picking/1d_CNN/model_description.ipynb)| U-Net 1D | Datasets 1, 2, 3 for FB picking <br/> | 0.06 MAE for dataset 1 <br/> 0.7 MAE for dataset 2 <br/> 15.88 MAE for dataset 3
|[First-break picking](models/First_break_picking/Coppen's_unsupervised_method/model_description.ipynb)| Coppen's analytical method | Datasets 1, 2, 3 for FB picking | 7.57 MAE for dataset 1 <br/> 7.19 MAE for dataset 2 <br/> 12.6 MAE for dataset 3
|[First-break picking](models/First_break_picking/Hidden_Markov_model/model_description.ipynb)| Hidden Markov model | Datasets 1, 2, 3 for FB picking | 2.6 MAE for dataset 1 <br/> 23.4 MAE for dataset 2 <br/> 8.0 MAE for dataset 3
|[Spherical divergence correction](models/Spherical_divergence_correction/model_description.ipynb) | Time and speed based method | Dataset 1 for SDC | 0.0017 Derivative metric


## Literature

Some articles related to seismic data processing:
* [Deep learning tutorial for denoising](https://arxiv.org/pdf/1810.11614.pdf)
* [Minimum weighted norm interpolation of seismic records](https://pdfs.semanticscholar.org/a742/67142fcd14c4c8d19992bd304a80e064d62c.pdf)
* [5D seismic data completion and denoising using a novel class of tensor decompositions](https://dspace.mit.edu/openaccess-disseminate/1721.1/98498)
* [Seismic images construction](http://lserv.deg.gubkin.ru/file.php?file=../../1/dfwikidata/Voskresenskij.JU.N.Postroenie.sejsmicheskih.izobrazhenij.%28M,.RGUNG%29%282006%29%28T%29_GsPs_.pdf)
* [Difraction](https://mospolytech.ru/storage/43ec517d68b6edd3015b3edc9a11367b/files/LRNo93.pdf)
* [Seismic facies recognition based on prestack data using deep convolutional autoencoder](https://arxiv.org/abs/1704.02446)
* [A comparison of classification techniques for seismic facies recognition](http://mcee.ou.edu/aaspi/publications/2015/Tao_Interpretation_1.pdf)
* [RESERVOIR CHARACTERIZATION: A MACHINE
LEARNING APPROACH](https://arxiv.org/pdf/1506.05070)
* [3D Seismic Attributes Enhancement and Detection by
Advanced Technology of Image Analysis](https://tel.archives-ouvertes.fr/tel-00731886/document)
* [CNN-based seismic facies clasification](https://cs230.stanford.edu/projects_spring_2018/reports/8291004.pdf)
* [Learning to Label Seismic Structures with Deconvolution Networks and Weak Labels](http://www.yalaudah.com/assets/files/seg2018.pdf)