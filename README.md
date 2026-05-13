This code is for our paper **"MrsNet: Multi-Scale Dual-domain Reconstruction with Spatial-Spectral Masked Network for Hyperspectral Anomaly Detection"**

# Abstract

In recent years, reconstruction-based methods have achieved notable success in hyperspectral anomaly detection (HAD). However, accurately recovering high-quality background remains challenging due to background complexity and interference from anomalous pixels. Furthermore, spectral variations and underutilized frequency-domain information often limit detection performance. To address these issues, this paper proposes a multi-scale dual-domain reconstruction network with a spatial-spectral masking strategy for HAD, named MrsNet. Specifically, a spatial-spectral masking (SSM) module is first introduced to reduce the influence of anomalies and spectral variability during feature extraction. A multi-scale dual-domain feature interaction module (DFIM) is then designed to model complex backgrounds by extracting and integrating multi-scale features from both spatial and frequency domains. Specifically, multi-scale convolution blocks are used to extract spatial features, and multi-scale modified wavelet convolution is employed to extract features in the frequency domain. Additionally, feature interaction is used to fully integrate features from different domains. To compensate for potential background loss caused by masking, a smoothness prior-guided module (SPM) leverages local homogeneity to assist the decoder in background reconstruction. Anomalies are finally identified by computing the Mahalanobis distance between the input HSI and the reconstructed background. Experimental results on four hyperspectral datasets and two UAV datasets demonstrate the effectiveness of MrsNet compared with nine state-of-the-art approaches.


# Citation

~~~
@article{ZHAO2026113925,
title = {MrsNet: Multi-scale dual-domain reconstruction with spatial-spectral masked network for hyperspectral anomaly detection},
journal = {Pattern Recognition},
pages = {113925},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113925},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326008903},
author = {Zhe Zhao and Jiangluqi Song and Huixin Zhou and Yong Zhu},
}
~~~
