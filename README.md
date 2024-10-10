## IncEventGS

<p align="center">
  <small> In this work, we present IncEventGS, a high-quality 3D Gaussian using a single event camera, without requiring ground truth camera poses.  </small>
</p>

> [[Paper](https://arxiv.org/)] &emsp;  <br>

<!-- > [Yang Cao*](https://yangcaoai.github.io/), Yuanliang Ju*, [Dan Xu](https://www.danxurgb.net) <br> -->


**Jian Huang**<sup>1,2</sup> &emsp; **Chengrui Dong**<sup>1,2</sup> &emsp; **Peidong Liu**<sup>2</sup><sup>\*</sup>  
<sup>1</sup> Zhejiang University &emsp; <sup>2</sup> Westlake University  
*Corresponding author

&#9744; The code and data will be made available once the paper is accepted. Stay tuned!

## Pipeline
<img src="./assets/pipeline.png">
IncEventGS processes incoming event stream by dividing it
into chunks and representing the camera trajectory as a continuous model. It randomly samples two
close consecutive timestamps to integrate the corresponding event streams. Two brightness images
are rendered from 3D Gaussian distributions at the corresponding poses, and we minimize the log
difference between the rendered images and the accumulated event images. During initialization, a
pre-trained depth estimation model estimates depth from the rendered images to bootstrap the system.

## Qualitative evaluation of novel view image synthesis on synthetic dataset. 
<img src="./assets/nvs_synthetic.png">

## Qualitative evaluation of novel view image synthesis on real dataset. 
<img src="./assets/nvs_real.png">

## Representative trajectory comparison
<img src="./assets/traj.png">