<h1 align="center">IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</h1>
<p align="center">
    <a href="https://github.com/HuangJianxjtu">Jian Huang</a><sup>1,2</sup> &emsp;&emsp;
    <a href="https://github.com/forgetable233">Chengrui Dong</a><sup>1,2</sup> &emsp;&emsp;
    <a href="https://github.com/mian-zhi">Xuanhua Chen</a><sup>2,3</sup> &emsp;&emsp;
    <a href="https://ethliup.github.io/">Peidong Liu</a><sup>2*</sup>
</p>


<p align="center">
  <sup>*</sup> denotes corresponding author.
</p>

<p align="center">
    <sup>1</sup>Zhejiang University &emsp;&emsp;
    <sup>2</sup>Westlake University &emsp;&emsp;
    <sup>3</sup>Northeastern University &emsp;&emsp;
</p>

<hr>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.08107-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.08107)
</h5>

> This repository is an official PyTorch implementation of the paper "IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera". We explore the possibility of recovering the 3D Gaussian and camera motion trajectory from a single event camera. 


## 📢 News
<!-- &#9744; The code and data will be made public once the paper is accepted. Stay tuned! -->

`2025.03.03` Our paper is accepted by CVPR 2025!
`2024.10.11` Our paper is available on [arXiv](https://arxiv.org/abs/2410.08107).

## 📋 Overview

<p align="center">
    <img src="./assets/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
IncEventGS processes incoming event stream by dividing it
into chunks and representing the camera trajectory as a continuous model. It randomly samples two
close consecutive timestamps to integrate the corresponding event streams. Two brightness images
are rendered from 3D Gaussian distributions at the corresponding poses, and we minimize the log
difference between the rendered images and the accumulated event images. During initialization, a
pre-trained depth estimation model estimates depth from the rendered images to bootstrap the system.
</div>

## 📋 Qualitative evaluation of novel view image synthesis on synthetic dataset.

<p align="center">
    <img src="./assets/nvs_synthetic.png" alt="nvs_synthetic" style="width:85%; height:auto;">
</p>

## 📋 Qualitative evaluation of novel view image synthesis on real dataset. 

<p align="center">
    <img src="./assets/nvs_real.png" alt="nvs_real" style="width:85%; height:auto;">
</p>

## 📋 Representative trajectory comparison

<p align="center">
    <img src="./assets/traj.png" alt="traj" style="width:85%; height:auto;">
</p>
