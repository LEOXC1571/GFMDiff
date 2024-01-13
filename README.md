<h1 align="center">Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2401.02683"><img src="https://img.shields.io/badge/arXiv-2401.02683-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/LEOXC1571/GFMDiff"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://arxiv.org/abs/2401.02683"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=AAAI%2724&color=orange"> </a>
    <a href="https://github.com/LEOXC1571/GFMDiff/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/CIGA?color=blue"> </a>

[//]: # (    <a href="https://neurips.cc/virtual/2022/poster/54643"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a>)

[//]: # (    <a href="https://lfhase.win/files/slides/CIGA.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>)

[//]: # (   <!--  <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->)
</p>

This repo contains the implementation of [Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation](https://arxiv.org/abs/2401.02683) for _de novo_ 3D molecule generation.

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2306.01103-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2306.01103&#41;)
## Table of contents

* [Overview](#overview)
* [Introduction](#introduction)
* [Citation](#citation)

[//]: # (* [Installation]&#40;#installation&#41;)
[//]: # (* [Run LECI]&#40;#run-leci&#41;)
[//]: # (* [Citing LECI]&#40;#citing-leci&#41;)
[//]: # (* [License]&#40;#license&#41;)
[//]: # (* [Contact]&#40;#contact&#41;)

## Overview
Denoising diffusion models have shown great potential in multiple research areas. Existing diffusion-based generative methods on _de novo_ 3D molecule generation face two major challenges. Since majority heavy atoms in molecules allow connections to multiple atoms through single bonds, solely using pair-wise distance to model molecule geometries is insufficient. Therefore, the first one involves proposing an effective neural network as the denoising kernel that is capable to capture complex multi-body interatomic relationships and learn high-quality features. Due to the discrete nature of graphs, mainstream diffusion-based methods for molecules heavily rely on predefined rules and generate edges in an indirect manner. The second challenge involves accommodating molecule generation to diffusion and accurately predicting the existence of bonds. In our research, we view the iterative way of updating molecule conformations in diffusion process is consistent with molecular dynamics and introduce a novel molecule generation method named Geometric-Facilitated Molecular Diffusion (GFMDiff). For the first challenge, we introduce a Dual-Track Transformer Network (DTN) to fully excevate global spatial relationships and learn high quality representations which contribute to accurate predictions of features and geometries. As for the second challenge, we design Geometric-Facilitated Loss (GFLoss) which intervenes the formation of bonds during the training period, instead of directly embedding edges into the latent space. Comprehensive experiments on current benchmarks demonstrate the superiority of GFMDiff.


## Introduction
<p align="center"><img src="./figures/overview_gfmdiff.png" width=100% height=100%></p>
<p align="center"><em>Figure 1.</em> The overview of GFMDiff.</p>

[//]: # ()
[//]: # (<p align="center"><img src="./figures/structure_dtn.png" width=100% height=100%></p>)

[//]: # (<p align="center"><em>Figure 2.</em> Illustration of DTN.</p>)

<p align="center"><img src="./figures/gfloss.png" width=40% height=40%></p>
<p align="center"><em>Figure 2.</em> The illustration of GFLoss.</p>


## Citation
If you find this repository helpful, please cite our [preprint](https://arxiv.org/abs/2401.02683).
```
@article{xu24geometric,
  title={Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation}, 
  author={Can Xu and Haosen Wang and Weigang Wang and Pengfei Zheng and Hongyang Chen},
  journal={arXiv preprint arXiv:2401.02683},
  year={2024}
}
```