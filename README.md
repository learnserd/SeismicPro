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
    * [Ground-roll attenuation](#Ground-roll-attenuation)
    * [First-break picking](#First--break-picking)
* [Models](#Models)
    * [Ground-roll attenuation](#Ground-roll-attenuation)
* [Literature](#Literature)

## About

SeismicPro provides a framework for machine learning on field seismic data.


## Installation

```
git clone --recursive https://github.com/analysiscenter/SeismicPro.git
```
## Tutorials

A set of IPython Notebooks introduces step-by-step the SeismicPro framework:

1. [Index](https://github.com/analysiscenter/SeismicPro/blob/master/tutorials/1.Index.ipynb) explains how to index data with respect to traces, field records, shot points etc.
2. [Batch](https://github.com/analysiscenter/SeismicPro/blob/master/tutorials/2.Batch.ipynb) shows how to load data, perform various actions with seismic traces and visualize them.
3. [Ground-roll attenuation](https://github.com/analysiscenter/SeismicPro/blob/master/tutorials/3.Noise_attenuation.ipynb) notebook shows how to build and run pipelines for model training, inference and evaluation with respect to ground-roll noise attenuation problem
4. [First-break picking](https://github.com/analysiscenter/SeismicPro/blob/master/tutorials/4.First-break_picking.ipynb) notebook shows model training and inference pipelines in a unsupervised first-break picking problem.


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

### Ground-roll attenuation

See the [notebook](https://github.com/analysiscenter/SeismicPro/blob/master/datasets/noise_attenuation.ipynb) for description of datasets.

### First-break picking

See the [notebook](https://github.com/analysiscenter/SeismicPro/blob/master/datasets/first_break_picking.ipynb) for description of datasets.

## Models

|Model|Architecture|Dataset|Metrics|
|---|---|---|---|
|[Ground-roll attenuation](https://github.com/analysiscenter/SeismicPro/blob/master/models/Ground-roll%20attenuation/model_description.ipynb)| U-Net 1D| Datasets 1, 2 for NA| 0.01 L1 
|[Masked ground-roll attenuation](https://github.com/analysiscenter/SeismicPro/blob/attention/notebooks/attention-demo.ipynb)| U-Net attention 1D| Datasets 1, 2 for NA | 0.01 L1, 0.02 L1 in GR area
|[First-break picking](https://github.com/analysiscenter/SeismicPro/blob/supervised_picking/models/First_break_picking/model_estimation.ipynb)| U-Net 1D | Datasets 1, 2, 3 for FB picking | 1.6 MAE, for 94% traces error is less than 3 samples
|[Trace Inversion Detection](https://github.com/analysiscenter/SeismicPro/blob/action_traces/models/Inverse_traces/find_inverse_traces.ipynb) | RandomForest | Dataset 1 for FB picking | 93% accuracy 

### Ground-roll attenuation

See the [notebook](https://github.com/analysiscenter/SeismicPro/blob/master/models/Ground-roll%20attenuation/model_description.ipynb) for description of the ground-roll attenuation model.


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