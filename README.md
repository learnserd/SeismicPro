# geolog

## Литература:
* [Построение сейсмических изображений](http://lserv.deg.gubkin.ru/file.php?file=../../1/dfwikidata/Voskresenskij.JU.N.Postroenie.sejsmicheskih.izobrazhenij.%28M,.RGUNG%29%282006%29%28T%29_GsPs_.pdf)
* [Явление дифракции](https://mospolytech.ru/storage/43ec517d68b6edd3015b3edc9a11367b/files/LRNo93.pdf)
* [Seismic facies recognition based on prestack data using deep convolutional autoencoder](https://arxiv.org/abs/1704.02446)
* [A comparison of classification techniques for seismic facies recognition](http://mcee.ou.edu/aaspi/publications/2015/Tao_Interpretation_1.pdf)
* [RESERVOIR CHARACTERIZATION: A MACHINE
LEARNING APPROACH](https://arxiv.org/pdf/1506.05070)
* [3D Seismic Attributes Enhancement and Detection by
Advanced Technology of Image Analysis](https://tel.archives-ouvertes.fr/tel-00731886/document)
* [Bayesian Inversion of Well Log Data
into Facies Units based on a Spatially
Coupled Model](http://daim.idi.ntnu.no/masteroppgaver/001/1371/tittelside.pdf)
* [CNN-based seismic facies clasification](https://cs230.stanford.edu/projects_spring_2018/reports/8291004.pdf)
* [Learning to Label Seismic Structures with Deconvolution Networks and Weak Labels](http://www.yalaudah.com/assets/files/seg2018.pdf)

## Датасеты:
* [2016-ml-contest](https://github.com/seg/2016-ml-contest)
* [LAS Wireline Logs in Kansas](http://www.kgs.ku.edu/Magellan/Logs/index.html)
* [Dutch F3](https://drive.google.com/drive/folders/0B7brcf-eGK8CRUhfRW9rSG91bW8)


## Процессы

### ГИС

<img src="https://yuml.me/diagram/usecase/%20[%D0%9F%D0%BE%D0%B4%D1%80%D1%8F%D0%B4%D1%87%D0%B8%D0%BA]-(%D0%9F%D1%80%D0%BE%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%93%D0%98%D0%A1),%20[%D0%9F%D0%BE%D0%B4%D1%80%D1%8F%D0%B4%D1%87%D0%B8%D0%BA]-(%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%B1%D1%83%D1%80%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D0%BA%D0%B2%D0%B0%D0%B6%D0%B8%D0%BD%D1%8B),%20(%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%B1%D1%83%D1%80%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D0%BA%D0%B2%D0%B0%D0%B6%D0%B8%D0%BD%D1%8B)%3E(C%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%84%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2%20%D0%B8%D0%BD%D0%BA%D0%BB%D0%B8%D0%BD%D0%BE%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%B8),%20(%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%B1%D1%83%D1%80%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%81%D0%BA%D0%B2%D0%B0%D0%B6%D0%B8%D0%BD%D1%8B)%3E(%D0%9F%D0%B5%D1%80%D0%B5%D0%B4%D0%B0%D1%87%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B2%20%D0%9A%D0%98%D0%9F),%20(%D0%9F%D1%80%D0%BE%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%93%D0%98%D0%A1)%3E(%D0%9F%D0%B5%D1%80%D0%B5%D0%B4%D0%B0%D1%87%D0%B0%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B2%20%D0%9A%D0%98%D0%9F),%20(%D0%9F%D1%80%D0%BE%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%93%D0%98%D0%A1)%3E(%D0%A3%D0%B2%D1%8F%D0%B7%D0%BA%D0%B0%20%D0%BA%D0%B0%D1%80%D0%BE%D1%82%D0%B0%D0%B6%D0%B5%D0%B9)">

<img src="https://yuml.me/diagram/usecase/%20[%D0%9F%D0%B5%D1%82%D1%80%D0%BE%D1%84%D0%B8%D0%B7%D0%B8%D0%BA]-(%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%D0%B0%20%D0%A0%D0%98%D0%93%D0%98%D0%A1),%20(%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%D0%B0%20%D0%A0%D0%98%D0%93%D0%98%D0%A1)%3E(%20%D0%9F%D1%80%D0%BE%D0%BF%D0%BB%D0%B0%D1%81%D1%82%D0%BA%D0%B0),%20(%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%D0%B0%20%D0%A0%D0%98%D0%93%D0%98%D0%A1)%3E(%D0%A0%D0%B0%D1%81%D1%87%D0%B5%D1%82%20%D0%B0%D1%82%D0%B8%D1%80%D0%B8%D0%B1%D1%83%D1%82%D0%BE%D0%B2)">

### Полевая сейсмика

<img src="https://yuml.me/diagram/usecase/%20[%D0%9F%D0%BE%D0%B4%D1%80%D1%8F%D0%B4%D1%87%D0%B8%D0%BA]-(%D0%A0%D0%B0%D1%81%D0%BA%D0%BB%D0%B0%D0%B4%D0%BA%D0%B0%20%D0%BF%D1%80%D0%BE%D1%84%D0%B8%D0%BB%D0%B5%D0%B9),%20[%D0%9F%D0%BE%D0%B4%D1%80%D1%8F%D0%B4%D1%87%D0%B8%D0%BA]-(%D0%92%D0%B7%D1%80%D1%8B%D0%B2%D0%BD%D1%8B%D0%B5%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B),%20(%D0%A0%D0%B0%D1%81%D0%BA%D0%BB%D0%B0%D0%B4%D0%BA%D0%B0%20%D0%BF%D1%80%D0%BE%D1%84%D0%B8%D0%BB%D0%B5%D0%B9)%3E(%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%84%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2%20SPS),%20(%D0%92%D0%B7%D1%80%D1%8B%D0%B2%D0%BD%D1%8B%D0%B5%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B)%3E(%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%84%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2%20SEGD/SEGY)">

<img src="https://yuml.me/diagram/usecase/%20[%D0%93%D0%B5%D0%BE%D0%BB%D0%BE%D0%B3]-(%D0%A1%D1%83%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%81%D0%B5%D0%B9%D1%81%D0%BC%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC),%20(%D0%A1%D1%83%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%81%D0%B5%D0%B9%D1%81%D0%BC%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC)%3E(%D0%9F%D1%80%D0%BE%D0%B2%D0%B5%D1%80%D0%BA%D0%B0%20%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BA%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%B8%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%B0%D1%86%D0%B8%D0%B8%20%D1%82%D1%80%D0%B0%D1%81%D1%81),%20(%D0%A1%D1%83%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%81%D0%B5%D0%B9%D1%81%D0%BC%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC)%3E(%D0%92%D1%8B%D0%B1%D0%BE%D1%80%20%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B1%D0%B8%D0%BD%D0%BE%D0%B2),%20(%D0%A1%D1%83%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%81%D0%B5%D0%B9%D1%81%D0%BC%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC)%3E(%D0%9C%D0%B8%D0%B3%D1%80%D0%B0%D1%86%D0%B8%D1%8F)">
