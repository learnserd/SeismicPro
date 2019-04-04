# geolog

Machine learning for field seismic data processing.

Content
=================

* [About](#About)
* [Installation](#Installation)
* [File formats](#File-formats)
	* [Seismic data](#Seismic-data)
	* [SPS data](#SPS-data)
	* [Picking data](#Picking-data)
* [Datasets](#Datasets)
    * [Noise attenuation](#Noise-attenuation)
    * [First-break picking](#First--break-picking)
* [Models](#Models)
    * [Noise attenuation](#Noise-attenuation)
    * [First-break picking](#First--break-picking)
* [Literature](#Literature)

## About
Geolog provides a framework for machine learning on field seismic data. Read [tutorial](https://github.com/analysiscenter/geolog/blob/master/tutorials/1.%20Index.ipynb) to learn how to index data with respect to traces, field records, shot points etc. Once the data are indexed, it can be loaded and processed. Read the next  [tutorial](https://github.com/analysiscenter/geolog/blob/master/tutorials/2.%20Batch.ipynb) to learn how to perform various actions.


## Installation

```
git clone --recursive https://github.com/analysiscenter/geolog.git
```

## File formats
### Seismic data

Seiemic data are expected to be in SEG-Y format.

### SPS data

SPS data are expected as R, S, X text files in csv (comma-separated-values) format with required and optional headers:
* Required R file headers: **rline**, **rid**, **x**, **y**, **z**.
* Required S file headers: **sline**, **sid**, **x**, **y**, **z**.
* Required X file headers: **FieldRecord**, **sline**, **sid**, **from_channel**, **to_channel**, **from_recaiver**, **to_receiver**.

### Picking data

File with first-break picking data is expected to be in csv (comma-separated-values) format with required and optional headers.

Required headers: **FieldRecord**, **TraceNumber**, **ShotPoint**, **timeOffset**.

## Datasets

### Noise attenuation

See the [notebook](https://github.com/analysiscenter/geolog/blob/master/datasets/noise_attenuation.ipynb) for description of datasets.

### First-break picking

## Models

### Noise attenuation

See the [tutorial](https://github.com/analysiscenter/geolog/blob/master/tutorials/3.%20Noise%20attenuation.ipynb) for model training and inference.

### First-break picking

See the [tutorial](https://github.com/analysiscenter/geolog/blob/master/tutorials/4.%20First-break%20picking.ipynb) for model training and inference.


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

