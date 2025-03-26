import sys
import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

import imageio
import pypose as pp

# Local imports
import config
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation, save_pose_as_kitti_evo, align_and_est_scale
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

# spline
from spline.spline import SE3_to_se3, se3_to_SE3, se3_to_SE3_m44

from datasets.dataset import get_event_chunk
# from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from datasets.utils import get_camera_rays
from datasets.dataset import get_event_chunk

import torchgeometry as tgm
from depth_warper import depth_warp
import cv2
import cv2 as cv
from scipy import stats

from kornia.filters import median_blur
from median_pool import MedianPool2d
from utils import render_ev_accumulation

from spline.spline_functor import linear_interpolation, cubic_bspline_interpolation

from torchvision.transforms import v2
from colormaps import apply_colormap
# from img_evaluation import compute_img_metric

from tikhonov_regularizor import tikhonov_regularization
from loss_utils import compute_white_balance_loss, compute_ssim_loss

# gsplat
from typing import Dict, List, Optional, Tuple
from utils import knn, rgb_to_sh, set_random_seed
import math
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, add_new_gs
from dataclasses import dataclass, field
# import tyro
from torch import Tensor
from plyfile import PlyData, PlyElement
from utils import BasicPointCloud
from utils import SH2RGB
import numba

log_eps = 1e-3
log = lambda x: torch.log(x + log_eps)
img2mse = lambda x, y: torch.mean((x - y) ** 2)

@numba.jit()
def accumulate_events(xs, ys, ts, ps, out, resolution_level, polarity_offset):
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        out[y // resolution_level, x // resolution_level] += p+polarity_offset

def get_row_major_sliced_ray_bundle(rays_o, rays_d, start_idx, end_idx):
            rays_o = torch.flatten(rays_o, start_dim=0, end_dim=1)[start_idx:end_idx]
            rays_d = torch.flatten(rays_d, start_dim=0, end_dim=1)[start_idx:end_idx]
            return rays_o, rays_d

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def lin_log(color, linlog_thres=1):
    """
    Input: 
    :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
           We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
    """
    # Compute the required slope for linear region (below luma_thres)
    # we need natural log (v2e writes ln and "it comes from exponential relation")
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb

def save_np_image(np_image, path_to_save):
    np_image =  to8b(np_image)
    imageio.imwrite(path_to_save, np_image)

def save_event_np_image(event_map, path_to_save):
    H = event_map.shape[0]
    W = event_map.shape[1]
    event_map = render_ev_accumulation(event_map, H, W)
    save_np_image(event_map, path_to_save)


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # How much to scale the camera origins by
    scale_factor: float = 1.0
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

def pcd_2_gs(
            points: Tensor = None,
            init_opacity: float = 0.1,
            init_scale: float = 1.0,
            scene_scale: float = 1.0,
            sh_degree: int = 3,
            feature_dim: Optional[int] = None,
            device: str = "cuda",
    ) -> torch.nn.ParameterDict:
    
    points = points # pcd
    pnum = points.shape[0]
    rgbs = torch.rand((pnum, 3))
    
    N = points.shape[0]
    
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    return splats

def create_splats_with_optimizers(
    init_type: str = "random",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    points: Optional[Tensor] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    
    if init_type == "sfm":
        print("***** Gaussion init_type: sfm *****")
        # points = torch.from_numpy(parser.points).float()
        # rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        points = points
        pnum = points.shape[0]
        rgbs = torch.rand((pnum, 3))
    elif init_type == "random":
        print("***** Gaussion init_type: random *****")
        points = scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    
    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class SLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_pose_data()
        
        # self.gs_dataset_cfg = gs_dataset_cfg
        # self.gs_opt_cfg = gs_opt_cfg
        # self.gs_pipe_cfg = gs_pipe_cfg

        self.control_knot_poses = None
        self.control_knot_ts = None
        self.control_knot_delta_t = 0.15 # create control knot every 5 frames 

        self.events_for_tracking = None
        self.events_for_BA = [] 
        
        self.gs_cfg = Config()
        
        self.scene_scale = self.config["mapping"]["bounding_size"]
        print("Scene scale(or size of bounding box):", self.scene_scale)
        # Model
        feature_dim = 32 if self.gs_cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            "random",
            init_num_pts=self.gs_cfg.init_num_pts,
            init_extent=self.gs_cfg.init_extent,
            init_opacity=self.gs_cfg.init_opa,
            init_scale=self.gs_cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=self.gs_cfg.sh_degree,
            sparse_grad=self.gs_cfg.sparse_grad,
            batch_size=self.gs_cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        
        # TODO: make this as a configurable parameter
        self.median_filter = MedianPool2d(kernel_size=5, same=True)
        self.median_filter_dvs = MedianPool2d(kernel_size=5, same=True)
        
        height = 480
        width = 640
        # mask = np.zeros((height, width), dtype=np.uint8)
        # radius = 325
        # center = (width // 2-15, height // 2)
        # cv2.circle(mask, center, radius, (1), -1)
        mask = np.ones((height, width), dtype=np.uint8)
        self.vector_mask = torch.from_numpy(mask).float().cuda()
        
        # self.scene_extent = None
        self.color_mask = np.zeros((self.dataset.H, self.dataset.W, 3))

        if self.config["mapping"]["color_channels"]==3:
            self.color_mask[0::2, 0::2, 0] = 1  # r

            self.color_mask[0::2, 1::2, 1] = 1  # g
            self.color_mask[1::2, 0::2, 1] = 1  # g

            self.color_mask[1::2, 1::2, 2] = 1  # b
        else:
            self.color_mask[...] = 1

        # self.color_mask = self.color_mask.reshape((-1, 3))
        self.color_mask = torch.from_numpy(self.color_mask).float().cuda()

        print('CoSLAM finished initialization...\n')

    
    # Experimental
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Experimental
    @torch.no_grad()
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        # copy sh0 and shN
        sh0_cp = self.splats["sh0"].detach()
        shN_cp = self.splats["shN"].detach()
        sh0_cp[:,:,1] = sh0_cp[:,:,0]
        sh0_cp[:,:,2] = sh0_cp[:,:,0]
        shN_cp[:,:,1] = shN_cp[:,:,0]
        shN_cp[:,:,2] = shN_cp[:,:,0]
        
        f_dc = sh0_cp.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = shN_cp.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_ts = {}
        self.ctrl_knot_se3_all = {}
        self.ctrl_knot_ts_all = {}
        self.load_gt_pose()
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[self.dataset.frame_ids[i]] = pose
    
    # def render_with_gsplat(self, gaussians : GaussianModel, T_cam2wld_se3, bRenderDepth=False, nOutputChannel = 1, render_tumvie_rgb = False):
    #     if render_tumvie_rgb and self.config["dataset"]=="tum_vie":
    #         focal_y = self.dataset.fy_rgb
    #         focal_x = self.dataset.fx_rgb
    #         cx = self.dataset.cx_rgb
    #         cy = self.dataset.cy_rgb
    #         # focal_y = self.dataset.fy 
    #         # focal_x = self.dataset.fx 
    #         # cx = self.dataset.cx
    #         # cy = self.dataset.cy
    #         image_width = self.dataset.W_rgb
    #         image_height = self.dataset.H_rgb
    #         T_cam2wld_SE3_pp = T_cam2wld_se3.Exp()
    #         T_evCam_rgbCam_raw = self.dataset.T_evCam_rgbCam[0]
            
    #         # estimate the scale
    #         save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"])
    #         scale = align_and_est_scale(self.pose_gt, self.est_c2w_data, 1, save_path, "pose", f"estimate_scale")
    #         T_evCam_rgbCam_raw[:3, 3] /= scale
            
    #         T_evCam_rgbCam = T_evCam_rgbCam_raw.copy()
    #         T_evCam_rgbCam_SE3_pp = pp.mat2SE3(torch.from_numpy(T_evCam_rgbCam)).float()
    #         T_rgbcam2wld_SE3_pp = T_evCam_rgbCam_SE3_pp.Inv() * T_cam2wld_SE3_pp
            
    #         T_cam2wld_se3 = T_rgbcam2wld_SE3_pp.Log()
    #     else:
    #         focal_y = self.dataset.fy 
    #         focal_x = self.dataset.fx 
    #         cx = self.dataset.cx
    #         cy = self.dataset.cy
    #         image_width = self.dataset.W
    #         image_height = self.dataset.H

    #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #     screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #     try:
    #         screenspace_points.retain_grad()
    #     except:
    #         pass
    
    #     T_c2w_Rt = torch.eye(4, device=self.device)
    #     T_c2w_Rt[:3,:4] = se3_to_SE3(T_cam2wld_se3)
    #     T_w2c_Rt = torch.linalg.inv(T_c2w_Rt)

    #     BLOCK_X, BLOCK_Y = 16, 16
        
    #     tile_bounds = (
    #         (image_width + BLOCK_X - 1) // BLOCK_X,
    #         (image_height + BLOCK_Y - 1) // BLOCK_Y,
    #         1,
    #     )

    #     xys, depths, radii, conics, _, num_tiles_hit, cov3d = ProjectGaussians.apply(
    #             gaussians._xyz,
    #             gaussians.get_scaling,
    #             1,
    #             gaussians._rotation,
    #             T_w2c_Rt,
    #             None, 
    #             focal_x,
    #             focal_y,
    #             cx,
    #             cy,
    #             image_height,
    #             image_width,
    #             tile_bounds,
    #         )
    #     torch.cuda.synchronize()

    #     shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
    #     dir_pp = (gaussians.get_xyz - T_c2w_Rt[:3, 3].unsqueeze(0).repeat(gaussians.get_features.shape[0], 1))
    #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #     sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    #     background = torch.ones(3, device=self.device)

    #     out_img, out_alpha, out_depth, out_uncertainty = RasterizeGaussians.apply(
    #         xys,
    #         depths,
    #         radii,
    #         conics,
    #         num_tiles_hit,
    #         colors_precomp,
    #         gaussians.get_opacity,
    #         image_height,
    #         image_width,
    #         background,
    #         False,
    #         bRenderDepth)

    #     torch.cuda.synchronize()

    #     # convert rgb image to single channel image for event loss
    #     if nOutputChannel == 1:
    #         out_img = out_img.mean(dim=2) 

    #     return {'image': out_img,
    #             'depth': out_depth,
    #             "viewspace_points": screenspace_points,
    #             'uncertainty': out_uncertainty,
    #             'xys': xys,
    #             'visibility_filter': radii > 0, 
    #             "radii": radii}
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        # read the camera parameters
        Ks = torch.from_numpy(self.dataset.K).to(self.device).unsqueeze(0) # [1, 3, 3]
        width = self.dataset.W
        height = self.dataset.H
        
        near_plane=self.gs_cfg.near_plane,
        far_plane=self.gs_cfg.far_plane,
        
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        # image_ids = kwargs.pop("image_ids", None)
        # if self.cfg.app_opt:
        #     colors = self.app_module(
        #         features=self.splats["features"],
        #         embed_ids=image_ids,
        #         dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        #         sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
        #     )
        #     colors = colors + self.splats["colors"]
        #     colors = torch.sigmoid(colors)
        # else:
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        # set the background color
        # 0-white, 1-black, 2-grey
        if self.config["mapping"]["background"]==0:
            backgrounds=torch.ones(Ks.shape[0], colors.shape[-1], device=self.device) # white
        elif self.config["mapping"]["background"]==1:
            backgrounds=torch.zeros(Ks.shape[0], colors.shape[-1], device=self.device) # black
        elif self.config["mapping"]["background"]==2:
            backgrounds=(159./255.)*torch.ones(Ks.shape[0], colors.shape[-1], device=self.device) # grey
        else:
            raise ValueError("Not implemented background color!")

        rasterize_mode = "antialiased" if self.gs_cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.gs_cfg.packed,
            absgrad=self.gs_cfg.absgrad,
            sparse_grad=self.gs_cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            backgrounds=backgrounds,
            **kwargs,
        )
        render_pkg = {}
        if self.config["mapping"]["color_channels"]==3:
            img=render_colors[0][..., 0:3]  # [H, W, 3]
        else:
            img=render_colors[0][..., 0]  # [H, W]
        
        if render_colors.shape[-1]==4:
            depth_img = render_colors[0][..., 3:4][..., 0]  # [H, W]
        else:
            depth_img = None
        
        return {"image": img,
                "depth": depth_img,
                "alpha": render_alphas,
                "info": info}
    
    def warp_image(self, T_se3_src2wld, T_se3_dst2wld, depth_src, image_dst):
        fx = self.dataset.fx
        fy = self.dataset.fy
        cx = self.dataset.cx
        cy = self.dataset.cy
        height = self.dataset.H
        width = self.dataset.W

        assert(depth_src.shape[0] == height) # N1HW
        assert(depth_src.shape[1] == width)
        assert(image_dst.shape[0] == height) # N1HW
        assert(image_dst.shape[1] == width)

        depth_src = depth_src.unsqueeze(0).unsqueeze(0)

        if image_dst.dim() == 3:
            image_dst = image_dst.permute(2,0,1)
            image_dst = image_dst.unsqueeze(0)
        else:
            image_dst = image_dst.unsqueeze(0).unsqueeze(0)

        intrinsics = torch.zeros(1, 4, 4).to(self.device)
        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0
        
        T_src2wld_Rt = torch.eye(4).repeat(1, 1, 1).to(self.device)
        T_dst2wld_Rt = torch.eye(4).repeat(1, 1, 1).to(self.device)
        
        T_src2wld_Rt[..., :3, :4] = se3_to_SE3(T_se3_src2wld)
        T_dst2wld_Rt[..., :3, :4] = se3_to_SE3(T_se3_dst2wld)
        
        # create image hegith and width
        height_tmp = torch.zeros(1).to(self.device)
        height_tmp[..., 0] += height
        width_tmp = torch.zeros(1).to(self.device)
        width_tmp[..., 0] += width
        
        # creat pinhole cameras 
        pinhole_src = tgm.PinholeCamera(intrinsics, T_src2wld_Rt, height_tmp, width_tmp)
        pinhole_dst = tgm.PinholeCamera(intrinsics, T_dst2wld_Rt, height_tmp, width_tmp)
        
        # 
        image_src = depth_warp(pinhole_dst, pinhole_src, depth_src, image_dst, height, width)  # NxCxHxW
        image_src = image_src.squeeze()
        depth_src = depth_src.squeeze()

        with torch.no_grad():
            if image_src.dim() == 3:
                image_src = image_src.permute(1,2,0)
                mask_src = torch.bitwise_and(image_src.mean(dim=2) > 0, depth_src > 0).detach().float() # no need to propagate gradients for mask 
            else:
                mask_src = torch.bitwise_and(image_src > 0, depth_src > 0).detach().float() # no need to propagate gradients for mask 

            mask_src = 1 - mask_src

            kernel = np.array([ [1, 1, 1], [1, 1, 1], [1, 1, 1] ], dtype=np.float32)
            kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).to(self.device) # size: (1, 1, 3, 3)
            mask_src = 1. - torch.clamp(torch.nn.functional.conv2d(mask_src.unsqueeze(0).unsqueeze(0), kernel_tensor, padding=(1, 1)), 0, 1).squeeze()

        return image_src, mask_src
    
    def post_process_event_image(self, events_stream, _sigma=0.001):
        #
        if self.config["dataset"] == "tum_vie" and self.config["data"]["downsample_factor"]>1:
            events_map = self.accumulate_event_to_img(self.dataset.H_old, self.dataset.W_old, events_stream) 
        else:
            events_map = self.accumulate_event_to_img(self.dataset.H, self.dataset.W, events_stream) 

        if self.config["dataset"] == "tum_vie":
            if self.config["mapping"]["use_median_filter"]:
                events_map = self.median_filter(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
            if self.config["data"]["downsample_factor"]>1:
                events_map = events_map.cpu().numpy()     
                # events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K_old, self.dataset.dist_coeffs, Knew=self.dataset.K_old, new_size=(self.dataset.W_old, self.dataset.H_old))
                events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K_new, new_size=(self.dataset.W, self.dataset.H))
                events_map = cv2.resize(events_map, (self.dataset.W, self.dataset.H), interpolation=cv2.INTER_LINEAR) #cv2.INTER_AREA     
                events_map = torch.from_numpy(events_map).cuda()
            else:
                events_map = events_map.cpu().numpy()
                # events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K, new_size=(self.dataset.W, self.dataset.H))
                events_map = cv2.fisheye.undistortImage(events_map, self.dataset.K, self.dataset.dist_coeffs, Knew=self.dataset.K_new, new_size=(self.dataset.W, self.dataset.H))
                events_map = torch.from_numpy(events_map).cuda()
        # elif self.config["dataset"] == "vector":
        #     events_map = events_map.cpu().numpy()
            
        #     # undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs, newCameraMatrix=self.dataset.K_new)
        #     undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs)
        #     events_map = torch.from_numpy(undisted_img).cuda()
        elif self.config["dataset"] == "dev_real" or self.config["dataset"] == "rpg_evo_stereo":
            events_map = events_map.cpu().numpy()
            # undisted_img = cv2.undistort(events_map, self.dataset.K_old, self.dataset.dist_coeffs, newCameraMatrix=self.dataset.K_new)
            undisted_img = cv2.undistort(events_map, self.dataset.K, self.dataset.dist_coeffs)
            events_map = torch.from_numpy(undisted_img).cuda()
            if self.config["mapping"]["use_median_filter"]:
                events_map = self.median_filter_dvs(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
        
        if self.config["blur_event"]:
            # blur
            blurrer = v2.GaussianBlur(kernel_size=(5,5), sigma=_sigma)
            events_map = blurrer(events_map.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return events_map

    def initialize_gaussian_scene(self, events_stream, num_pixels_to_sample, T_cam_to_wld, threshold = 0.1, depth_map=None):
        if depth_map is not None:
            assert(depth_map.dim()==2)

        threshold = self.config["event"]["threshold"]
        if depth_map is None:
            # depth = 100
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # xyz = np.random.random((num_pts, 3)) * 7.0 - 3.5
            bounding_size = self.config["bounding_size"]
            xyz = np.random.random((num_pts, 3))*bounding_size - 0.5*bounding_size
            
            shs = np.random.random((num_pts, 3)) / 255.0
            xyz = torch.from_numpy(xyz).cuda().float()
            shs = torch.from_numpy(shs).cuda().float()
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=torch.zeros_like(xyz).cuda().float())
        
        else:
            print('Use provided depth map for scene initialization...')
            events_map = self.post_process_event_image(events_stream)
    
            # sample pixels with events
            fx = self.dataset.fx
            fy = self.dataset.fy
            cx = self.dataset.cx
            cy = self.dataset.cy
            H = self.dataset.H
            W = self.dataset.W
            
            min_depth = self.config["mapping"]["min_depth"]
            max_depth = self.config["mapping"]["max_depth"]
            if self.config["initialization"]["gaussian_init_sfm_mask"]==1: # using event map mask; 
                pixel_uv_with_event = torch.where(events_map.abs() > threshold)
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_pixels_to_sample]
                indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
            elif self.config["initialization"]["gaussian_init_sfm_mask"]==0: # using depth map mask
                pixel_uv_with_event = torch.where((depth_map.abs() > min_depth) & (depth_map.abs() < max_depth))
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_pixels_to_sample]
                indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
            elif self.config["initialization"]["gaussian_init_sfm_mask"]==2: # using both event and depth map mask
                pixel_uv_with_event = torch.where((depth_map.abs() > min_depth) & (depth_map.abs() < max_depth))
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:int(num_pixels_to_sample*0.5)]
                indice_h1 = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w1 = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
                
                pixel_uv_with_event = torch.where(events_map.abs() > threshold)
                sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:int(num_pixels_to_sample*0.5)]
                indice_h2 = pixel_uv_with_event[0][sampled_event_pixel_idx].to(self.device)
                indice_w2 = pixel_uv_with_event[1][sampled_event_pixel_idx].to(self.device)
                
                indice_h = torch.cat([indice_h1, indice_h2])
                indice_w = torch.cat([indice_w1, indice_w2])
            else:
                raise ValueError("wrong option for gaussian_init_sfm_mask")
            
            
            #
            sampled_rays = get_camera_rays(H, W, fx, fy, cx, cy, type='OpenCV').to(self.device)
            sampled_rays = sampled_rays[indice_h, indice_w, :]
        
            depth = depth_map[indice_h, indice_w].unsqueeze(-1).float()
    
            sampled_rays = (sampled_rays * depth).transpose(1,0).to(self.device)
            #
            sampled_rays = torch.matmul(T_cam_to_wld[:3, :3], sampled_rays) + T_cam_to_wld[:3, 3].unsqueeze(-1) 
            points = sampled_rays.transpose(1, 0)
            # colors
            min = events_map.min()
            max = events_map.max()
            events_map = (events_map - min) / (max - min)
            colors = events_map[indice_h, indice_w].unsqueeze(-1).repeat(1, 3)
            # normals
            normals = torch.zeros_like(colors)
            normals[:, 2] = 1.
            # create pcd
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

        return pcd

    # spline pose and get 7 rays_o/rays_d, then get color and average
    def initialization(self, init_batch, niters, traj_mode='cspline', depth_map=None):
        traj_mode = 'linear'
        
        num_batch = len(init_batch)
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        preSum_event_batch = [0]
        for i in range(num_batch):
            frame_id = init_batch[i]["frame_id"].item()
            frame_ts = init_batch[i]["pose_ts"].item()
            frame_id_all.append(frame_id)
            frame_ts_all.append(frame_ts)
            cur_event = init_batch[i]['events'].squeeze().to(self.device)
            preSum_event_batch.append(cur_event.shape[0]+preSum_event_batch[-1])
            events_all.append(cur_event)
        num_events_to_skip = self.config["num_events_to_skip"]
        preSum_event_batch[0] = preSum_event_batch[0]+num_events_to_skip
        preSum_event_batch[-1] = preSum_event_batch[-1]-num_events_to_skip
        
        if depth_map is None:
            print('#########################   random initialization   #########################')
        else:
            # initialize Gaussians with depth_map
            print('#########################   initialization with depth_map   #########################')
            # initialize Gaussians
            if self.config["initialization"]["retain_pose"]:
                assert(len(self.ctrl_knot_se3_all)>0)
                idx_ = frame_id_all[-1]
                T_cam2wld_Rt = se3_to_SE3_m44(self.ctrl_knot_se3_all[idx_]).cuda()
            else:
                T_cam2wld_Rt = torch.eye(4).to(self.device)
            
            gaussian_num_sfm = self.config["initialization"]["gaussian_num_sfm"]
            pcd = self.initialize_gaussian_scene(events_all[0], gaussian_num_sfm, T_cam2wld_Rt, depth_map=depth_map)
            # self.gs_model.create_from_tensor_pcd(pcd, spatial_lr_scale=8.23)
            # self.gs_model.training_setup(self.gs_opt_cfg)
            
            feature_dim = 32 if self.gs_cfg.app_opt else None
            if self.config["retain_old_gs"]:
                print(f"============================= retain old gs ===============================")
                new_gs = pcd_2_gs(
                        points= pcd.points.detach(),
                        init_opacity=self.gs_cfg.init_opa,
                        init_scale=self.gs_cfg.init_scale,
                        scene_scale=self.scene_scale,
                        sh_degree=self.gs_cfg.sh_degree,
                        feature_dim=feature_dim,
                        device=self.device,
                        )
                add_new_gs(self.splats, self.optimizers, new_gs)
            else:
                feature_dim = 32 if self.gs_cfg.app_opt else None
                self.splats, self.optimizers = create_splats_with_optimizers(
                    "sfm",
                    init_num_pts=self.gs_cfg.init_num_pts,
                    init_extent=self.gs_cfg.init_extent,
                    init_opacity=self.gs_cfg.init_opa,
                    init_scale=self.gs_cfg.init_scale,
                    scene_scale=self.scene_scale,
                    sh_degree=self.gs_cfg.sh_degree,
                    sparse_grad=self.gs_cfg.sparse_grad,
                    batch_size=self.gs_cfg.batch_size,
                    feature_dim=feature_dim,
                    device=self.device,
                    points=pcd.points
                )
        
        # densification setting
        self.gs_cfg.prune_opa = self.config["mapping"]["prune_opa"]
        self.gs_cfg.refine_start_iter = self.config["mapping"]["refine_start_iter"]
        self.gs_cfg.refine_stop_iter= self.config["mapping"]["refine_stop_iter"]
        self.gs_cfg.refine_every = self.config["mapping"]["refine_every"]
        self.gs_cfg.grow_grad2d = self.config["mapping"]["grow_grad2d"]
        self.gs_cfg.grow_scale3d = self.config["mapping"]["grow_scale3d"]
        # prune_scale3d: float = 0.1
        self.gs_cfg.prune_scale3d = self.config["mapping"]["prune_scale3d"]
        
        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            scene_scale=self.scene_scale,
            prune_opa=self.gs_cfg.prune_opa,
            grow_grad2d=self.gs_cfg.grow_grad2d,
            grow_scale3d=self.gs_cfg.grow_scale3d,
            prune_scale3d=self.gs_cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=self.gs_cfg.refine_start_iter,
            refine_stop_iter=self.gs_cfg.refine_stop_iter,
            reset_every=self.gs_cfg.reset_every,
            refine_every=self.gs_cfg.refine_every,
            absgrad=self.gs_cfg.absgrad,
            revised_opacity=self.gs_cfg.revised_opacity,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()
        
        
        frame_ts_all_withzero = frame_ts_all.copy()
        frame_ts_all_withzero.insert(0, 0.0)
        
        events_all = torch.cat(events_all, dim=0)
        ts_all = events_all[:,2].detach().cpu().numpy()
        t_data_min =  ts_all.min()
        t_data_max = ts_all.max()
        event_total_num = ts_all.shape[0]
        
        incre_sampling_seg_num_expected = self.config["initialization"]["incre_sampling_seg_num_expected"]
        min_n_winsize = self.config["initialization"]["min_n_winsize"]
        max_n_winsize = self.config["initialization"]["max_n_winsize"]
        
        incre_sampling_segs_end = np.linspace(10, event_total_num-10, incre_sampling_seg_num_expected).astype(int)
        start_tmp_ = np.searchsorted(incre_sampling_segs_end, max_n_winsize)
        if(incre_sampling_segs_end[start_tmp_]<=max_n_winsize):
            start_tmp_ = start_tmp_+1
        incre_sampling_segs_end = incre_sampling_segs_end[start_tmp_:]
        assert incre_sampling_segs_end.shape[0]>1
        assert incre_sampling_segs_end[0]>max_n_winsize
        events_per_seg = incre_sampling_segs_end[1]-incre_sampling_segs_end[0]
        incre_sampling_seg_num = incre_sampling_segs_end.shape[0]
        print(f"*******************************  events_num: {event_total_num}  *********************************")
        print(f"incremental sampling number: {incre_sampling_seg_num}")
        print(f"****events_per_seg={events_per_seg}, min_n_winsize={min_n_winsize},max_n_winsize={max_n_winsize}, ")
        
        ctrl_knot_ts = frame_ts_all.copy()
        ctrl_knot_ts.insert(0, t_data_min)
        if ctrl_knot_ts[-1]<frame_ts_all[-1]:
            ctrl_knot_ts.append(frame_ts_all[-1])
        ctrl_knot_ts[0] = ctrl_knot_ts[0]-0.0001
        ctrl_knot_ts[-1] = ctrl_knot_ts[-1]+0.0001
        ctrl_knot_idx = frame_id_all.copy()
        ctrl_knot_idx.insert(0, frame_id_all[0]-1)
        
        print(f"****t_data_min={t_data_min}, t_data_max={t_data_max}")
        print(f"control knot ts: {ctrl_knot_ts}")
        
        
        if len(self.ctrl_knot_ts_all)==0 or (not self.config["initialization"]["retain_pose"]):
            print("============== initialize control knots with random noise ==============")
            # set poses of control knots
            if self.config["use_gt_pose_to_opt"]:
                print("Setting trajectory")
                ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
                ctrl_knot_se3[0] = SE3_to_se3(self.pose_gt[frame_id_all[0]-1])
                for i in range(1, len(ctrl_knot_ts)):
                    ctrl_knot_se3[i] = SE3_to_se3(self.pose_gt[frame_id_all[i-1]])
            else:
                ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
        else:
            print("============== initialize control knots with pose of last stage ==============")
            assert(len(self.ctrl_knot_se3_all) == len(ctrl_knot_ts))
            ctrl_knot_se3 = pp.randn_se3(len(ctrl_knot_ts), sigma=0.001) #LieTensor
            for i in range(len(ctrl_knot_ts)):
                ctrl_knot_se3[i] = self.ctrl_knot_se3_all[i]
            
        print("finish control knots initialization")
        
        ctrl_knot_se3 = torch.nn.Parameter(ctrl_knot_se3, requires_grad=True)
        pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3, "lr": self.config['pose_lr']}])
        
        color_channels=self.config["mapping"]["color_channels"]
        
        training_batch_size = self.config["initialization"]["training_batch_size"]
        visualize_every_iter = self.config["initialization"]["visualize_every_iter"]
        blur_sigma = self.config["blur_sigma"]
        for iter in range(niters):
            # if i%100 == 0:
            #     blur_sigma = blur_sigma/2
            
            loss_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_no_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_ssim = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_white_balance = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_tr = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # incremental random sampling
            random_start_end_idx = []
            for ii in range(incre_sampling_seg_num):
                end_idx_ = incre_sampling_segs_end[ii]
                winsize_ = np.random.randint(min_n_winsize, max_n_winsize)
                start_idx_ = end_idx_-winsize_
                random_start_end_idx.append([start_idx_, end_idx_])
            indices = np.random.permutation(len(random_start_end_idx)).tolist()[:training_batch_size]
            
            list_img_ev_start = []
            list_img_ev_end = []
            list_gt_events_acc = []
            list_syn_event_acc = []
            
            linlog_thres = self.config["event"]["linlog_thres"]
            # intra-chunk sampling
            for j in indices:
                idx_ev_start = random_start_end_idx[j][0]
                idx_ev_end = random_start_end_idx[j][1]
                selected_event_stream = events_all[idx_ev_start:idx_ev_end]
                t_ev_start = ts_all[idx_ev_start]
                t_ev_end = ts_all[idx_ev_end]

                T_SE3_ev_start = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_start, mode=traj_mode)
                T_SE3_ev_end = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_end, mode=traj_mode)
                
                # forward
                c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_start = self.rasterize_splats(camtoworlds=c2w_start, render_mode="RGB+ED")
                c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_end = self.rasterize_splats(camtoworlds=c2w_end, render_mode="RGB+ED")
                
                img_ev_start = render_pkg_start["image"]
                img_ev_end = render_pkg_end["image"]
                
                if self.config["use_linLog"]:
                    pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    syn_event_acc = pred_linlog_end - pred_linlog_start
                else:
                    # syn_event_acc = (log(img_ev_end) - log(img_ev_start))
                    eps = 0.1
                    syn_event_acc = torch.log(img_ev_end**2.2+eps)-torch.log(img_ev_start**2.2+eps)
                
                # compute event loss
                gt_events_acc = self.post_process_event_image(selected_event_stream, _sigma=blur_sigma)    
                
                # if self.config["use_mask_event_loss"]:
                event_mask = (gt_events_acc.abs() != 0).float() 
                no_event_mask = (gt_events_acc.abs() == 0).float()
                
                # no-event loss
                no_event_gaussian_cov = self.config["mapping"]["no_event_gaussian_cov"]
                gt_no_events = self.config["event"]["threshold"] * no_event_gaussian_cov * torch.randn_like(gt_events_acc).cuda()
                gt_no_events = gt_no_events*no_event_mask
                
                if self.config["mapping"]["color_channels"]==3:
                    gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                    # gt_events_acc = np.tile(gt_events_acc[..., None], (1, 1, 3))
                    gt_events_acc = gt_events_acc*self.color_mask
                    syn_event_acc = syn_event_acc*self.color_mask
                
                if self.config["seprate_event_noevent_loss"]:
                    loss_event = loss_event + (event_mask*(gt_events_acc - syn_event_acc)**2).sum() / event_mask.sum()
                    loss_no_event = loss_no_event + (no_event_mask*(gt_no_events - syn_event_acc)**2).sum() / no_event_mask.sum()
                else:
                    # gt_events_acc = gt_events_acc+gt_no_events # TODO: xiugai !!!
                    gt_events_acc = gt_events_acc
                    loss_event = loss_event + ((gt_events_acc - syn_event_acc)**2).mean()
                
                
                if self.config["mapping"]["color_channels"]==3:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
                else:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=1)
                    
                # # visualize vector mask on event image
                # vector_vis_event = gt_events_acc*self.vector_mask
                # vector_events_acc = render_ev_accumulation(vector_vis_event.cpu().numpy(), self.dataset.H, self.dataset.W)
                # save_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"])
                # img_path = os.path.join(save_dir, f"vector_mask_event.jpg")  
                # imageio.imwrite(img_path, vector_events_acc)
                # os.system.exit(0)
                
                list_img_ev_start.append(img_ev_start.detach())
                list_img_ev_end.append(img_ev_end.detach())
                list_gt_events_acc.append(gt_events_acc)
                list_syn_event_acc.append(syn_event_acc.detach())
            
            # rgb loss
            if self.config["mapping"]["use_rgb_loss"]:
                rgb_idx_ = np.random.randint(num_batch)
                rgb_c2w_ = init_batch[rgb_idx_]["c2w"].cuda()
                rgb_gt_ = init_batch[rgb_idx_]["rgb"][0][:, :,0].cuda()
                render_pkg_rgb = self.rasterize_splats(camtoworlds=rgb_c2w_, render_mode="RGB")
                rgb_est = render_pkg_rgb["image"]
                rgb_w = self.config["mapping"]["loss_rgb_weight"]
                loss_rgb = rgb_w*((rgb_est-rgb_gt_)**2).mean()
            else:
                loss_rgb = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # tikhonov regularization loss
            tr_w = self.config["mapping"]["tr_loss_weight"]
            loss_tr = tr_w*tikhonov_regularization(render_pkg_start["depth"].unsqueeze(-1))
            
            if self.config["mapping"]["use_white_balance_loss"]:
                # white balance loss 
                white_balance_weight = self.config["mapping"]["white_balance_weight"]
                loss_white_balance = white_balance_weight*compute_white_balance_loss(img_ev_end.mean())
            
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=iter,
                info=render_pkg_end["info"],
            )
            
            loss_event = loss_event/len(indices)
            loss_ssim = loss_ssim/len(indices)
            loss_no_event = loss_no_event/len(indices)
            # summary the loss
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim
            
            # # isotropic loss
            # scaling = self.splats["scales"]  # [N, 3]
            # isotropic_loss_all = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # iso_w = self.config["mapping"]["loss_isotropic_weight"]
            # loss_isotropic = iso_w * isotropic_loss_all.mean()
            
            # loss_total = loss_event + loss_ssim + loss_isotropic + loss_rgb 
            loss_total = loss_event + loss_ssim + loss_no_event + loss_white_balance + loss_tr
            
            # optimize 
            pose_optimizer.zero_grad()
            loss_total.backward()
            torch.cuda.synchronize()
            
            # densification
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=iter,
                info=render_pkg_end["info"],
            )
            
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # if self.config["opt_pose"] and i>1000:
            if self.config["opt_pose"]:
                if (depth_map is None) or (depth_map is not None and iter>200):
                    pose_optimizer.step()

            # visualization
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "initialization")
            if iter % visualize_every_iter == 0 and self.config["visualize_inter_img"]:
                print_str = f"iter {iter}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}, white={loss_white_balance.item()}, noevent={loss_no_event.item()}, tr={loss_tr.item()}"
                print(print_str)
                with torch.no_grad():
                    # plot pose estimation
                    accum_est_poses_Rt = []
                    accum_gt_poses_Rt = []
                    tag = f"pose"
                    
                    # # save ply file
                    # ply_path = os.path.join(save_path, f"pointCloud_it{iter}.ply")
                    # self.save_ply(ply_path)
                    
                    for n in range(num_batch):
                        if traj_mode == "cspline":
                            if n==0 or n==(num_batch-1):
                                continue
                        k = frame_id_all[n]
                        ts = frame_ts_all[n]                        
                        accum_est_poses_Rt.append(self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode).matrix()) 
                        accum_gt_poses_Rt.append(init_batch[n]['c2w'].squeeze())                        
                    
                    if self.config["dataset"] == "rpg_evo":
                        accum_gt_poses_Rt = accum_est_poses_Rt
                    # plot 
                    pose_evaluation(accum_gt_poses_Rt, accum_est_poses_Rt, 1, save_path, tag, f"init_f{iter:03}")
                    save_pose_as_kitti_evo(accum_gt_poses_Rt, accum_est_poses_Rt, save_path, f"init_f{iter:03}")
                    
                    if self.config["initialization"]["visualize_intermediate_img"]:
                        # save intermediate images 
                        images_combined = []
                        for m in range(len(list_img_ev_start)):
                            diff = (list_gt_events_acc[m] - list_syn_event_acc[m]) **2
                            
                            if color_channels == 1:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].cpu().numpy(), self.dataset.H, self.dataset.W)
                                gt_events_grey = np.repeat(list_gt_events_acc[m].cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                
                                syn_event_acc = list_syn_event_acc[m].detach().cpu().numpy()
                                syn_event_acc1 = np.where(syn_event_acc > self.config["event"]["threshold"], syn_event_acc, 0)
                                syn_event_acc2 = np.where(syn_event_acc < -self.config["event"]["threshold"], syn_event_acc, 0)
                                syn_event_acc = syn_event_acc1 + syn_event_acc2
                                syn_event_acc = render_ev_accumulation(syn_event_acc, self.dataset.H, self.dataset.W)
                                
                                syn_event_grey = np.repeat(list_syn_event_acc[m].detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                                
                                # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = np.repeat(list_img_ev_start[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                                img_ev_end = np.repeat(list_img_ev_end[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            else:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                                syn_event_acc = list_syn_event_acc[m].detach().abs().cpu().numpy()
                                diff = diff.detach().abs().cpu().numpy()
                                
                                # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = list_img_ev_start[m].detach().cpu().numpy()
                                img_ev_end = list_img_ev_end[m].detach().cpu().numpy()
                                
                                syn_event_grey = list_gt_events_acc[m].cpu().numpy()
                                gt_events_grey = list_gt_events_acc[m].cpu().numpy()

                            image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, gt_events_grey,syn_event_acc, syn_event_grey, diff], axis=1)
                            images_combined.append(image_combined)
                        
                        images_combined = np.concatenate(images_combined, axis=0)
                        images_combined =  to8b(images_combined)
                        tag_img = "init_"
                        img_path = os.path.join(save_path, tag_img + f"f{iter:03}_img.jpg")  
                        imageio.imwrite(img_path, images_combined)
                    
                    # render ALL images with depth
                    if self.config["render_eventview_img"]:
                        images_combined = []
                        for n in range(num_batch):
                            if n%30!=0:
                                continue
                            idx_ = frame_id_all[n]
                            ts = frame_ts_all[n]
                            T_SE3 = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode)
                            
                            c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                            render_pkg_start = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB+ED")
                            
                            depth_img_gs = render_pkg_start["depth"]  # [H, W, 1]
                            depth_tmp = depth_img_gs.detach()
                            depth_tmp = depth_tmp.view(depth_tmp.shape[0], depth_tmp.shape[1], -1)
                            depth_img = apply_colormap(depth_tmp)
                            depth_img = depth_img.cpu().numpy()
                            depth_img = to8b(depth_img)
                            
                            # rendered_img = render_pkg_start[0][..., 0:3][0]  # [H, W, 3]
                            rendered_img = render_pkg_start["image"]  # [H, W] or [H, W, 3]
                            if rendered_img.shape[-1]!=3:
                                rendered_img = torch.tile(rendered_img[:,:,None], (1,1,3))
                            img = rendered_img.detach().cpu().numpy()
                            img = to8b(img)
                            
                            image_combined = np.concatenate([img, depth_img], axis=1)
                            images_combined.append(image_combined)
                        images_combined = np.concatenate(images_combined, axis=0)
                        im_name = f"iter_{iter}_vis.jpg"
                        imageio.imwrite(os.path.join(save_path, im_name), images_combined)

        print("********** Number of GS:", len(self.splats["means"]))
        
        # update est_poses
        for n in range(num_batch):
            k = frame_id_all[n]
            ts = frame_ts_all[n]                      
            self.est_c2w_data[k] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode).matrix().cuda()
            self.est_c2w_ts[k] = ts
        
        # render ALL images 
        if self.config["render_eventview_img"]:
            with torch.no_grad():
                for n in range(num_batch):
                    idx_ = frame_id_all[n]
                    ts = frame_ts_all[n]
                    T_SE3 = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode=traj_mode)
                    
                    c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                    rendered_img = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB+ED")['image']
                    
                    rendered_img = rendered_img.detach().cpu().numpy()
                    rendered_img =  to8b(rendered_img)
                    im_name = f"frame{idx_}_init.jpg"
                    imageio.imwrite(os.path.join(save_path, im_name), rendered_img)
                    if self.config["evaluate_init_image"]:
                        imageio.imwrite(os.path.join(self.dataset.event_save_path, im_name), rendered_img)
                    
                    if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                        # render rbg camera image
                        img_ev_start_rgb = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                        img_ev_start_rgb = np.repeat(img_ev_start_rgb.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                        img_path = os.path.join(save_path, tag_img + f"frame{idx_}_init_rgbCam.jpg")  
                        imageio.imwrite(img_path, to8b(img_ev_start_rgb))
        
        # save control knot pose and ts
        for ii in range(len(ctrl_knot_idx)):
            idx_ = ctrl_knot_idx[ii]
            self.ctrl_knot_ts_all[idx_] = ctrl_knot_ts[ii]
            self.ctrl_knot_se3_all[idx_] = ctrl_knot_se3[ii].detach().clone()  # [], cuda
        
        print("======================= finished initialization  =========================")
    
    def save_model(self, path_to_save, tag, accum_ctrl_se3, accum_ctrl_ts):
        print('saving model in', path_to_save, 'with tag', tag, '...')

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        with torch.no_grad():
            accum_ctrl_se3 = torch.stack(accum_ctrl_se3, dim=0).cpu().numpy()
            accum_ctrl_ts = np.asarray(accum_ctrl_ts)

            np.save(os.path.join(path_to_save, tag + '_ctrl_poses.npy'), accum_ctrl_se3)
            np.save(os.path.join(path_to_save, tag + '_ctrl_ts.npy'), accum_ctrl_ts)
            self.gs_model.save_ply(os.path.join(path_to_save, tag + '_gaussians.npy'))
        return 
    
    def load_model(self, path_to_load, tag):
        print('Loading model from', path_to_load, 'with tag', tag, '...')
        
        self.gs_model.load_ply(os.path.join(path_to_load, tag + '_gaussians.npy'))
        self.gs_model.training_setup(self.gs_opt_cfg)

        accum_ctrl_se3 = torch.from_numpy(np.load(os.path.join(path_to_load, tag + '_ctrl_poses.npy'))).to(self.device)
        accum_ctrl_se3 = [accum_ctrl_se3[i].detach() for i in range(accum_ctrl_se3.shape[0])]
        accum_ctrl_ts = np.load(os.path.join(path_to_load, tag + '_ctrl_ts.npy')).tolist()

        return accum_ctrl_se3, accum_ctrl_ts
    
    def predict_velocity(self, T_Rt_prev, T_Rt_prevprev):
        T_Rt_c2w_prev = torch.eye(4, device=self.device)
        T_Rt_c2w_prevprev = torch.eye(4, device=self.device)

        T_Rt_c2w_prev[:3,:4] = T_Rt_prev
        T_Rt_c2w_prevprev[:3,:4] = T_Rt_prevprev

        T_Rt_delta = torch.matmul(torch.linalg.inv(T_Rt_c2w_prevprev), T_Rt_c2w_prev)
        # T_Rt_cur = torch.matmul(T_Rt_c2w_prev, T_Rt_delta)

        return T_Rt_delta #T_Rt_cur[:3,:4]

    def predict_pose_se3_lie(self, pose_prev_se3_lie, pose_prevprev_se3_lie):
        pose_delta_SE3_lie = pose_prevprev_se3_lie.Inv().Exp() * pose_prev_se3_lie.Exp()
        predicted_SE3_lie = pose_prev_se3_lie.Exp() * pose_delta_SE3_lie
        return predicted_SE3_lie.Log()
    
    def get_poses(self, control_knot_poses, control_knot_ts, query_t, mode = 'cspline'):
        if mode == 'linear':
            if np.abs(query_t - control_knot_ts[0]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-1]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[0]   # numpy
            assert query_t <= control_knot_ts[-1]
            
            idx_end = np.searchsorted(control_knot_ts, query_t)
            idx_start = idx_end - 1
            assert idx_start>=0
            assert control_knot_ts[idx_start]<=query_t
            assert control_knot_ts[idx_end]>=query_t
            
            ctrl_knot_t_start = control_knot_ts[idx_start]
            ctrl_knot_delta_t = control_knot_ts[idx_end] - control_knot_ts[idx_start]
            
            elapsed_t = query_t - ctrl_knot_t_start
            tau = elapsed_t/ctrl_knot_delta_t # normalize to (0,1)
            
            pose_interp = (1. - tau) * control_knot_poses[idx_start] + tau * control_knot_poses[idx_end]        
        elif mode == 'cspline':
            if np.abs(query_t - control_knot_ts[1]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-2]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[1]
            assert query_t <= control_knot_ts[-2]

            ctrl_knot_t_start = control_knot_ts[1]
            ctrl_knot_delta_t = control_knot_ts[1] - control_knot_ts[0]

            elapsed_t = query_t - ctrl_knot_t_start
            idx = int(elapsed_t / ctrl_knot_delta_t)
            tau = (elapsed_t % ctrl_knot_delta_t) / ctrl_knot_delta_t # normalize to (0,1)

            pose_interp = Spline4N_new(control_knot_poses[idx], control_knot_poses[idx+1], control_knot_poses[idx+2], control_knot_poses[idx+3], tau)
            pose_interp = SE3_to_se3(pose_interp.squeeze(0))
        return pose_interp 

    def get_poses_lie(self, control_knot_poses, control_knot_ts, query_t, mode = 'cspline'):
        if mode == 'linear':
            if np.abs(query_t - control_knot_ts[0]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-1]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[0]   # numpy
            assert query_t <= control_knot_ts[-1]
            
            idx_end = np.searchsorted(control_knot_ts, query_t)
            idx_start = idx_end - 1
            assert idx_start>=0
            assert control_knot_ts[idx_start]<=query_t
            assert control_knot_ts[idx_end]>=query_t
            
            ctrl_knot_t_start = control_knot_ts[idx_start]
            ctrl_knot_delta_t = control_knot_ts[idx_end] - control_knot_ts[idx_start]
            
            elapsed_t = query_t - ctrl_knot_t_start
            tau = elapsed_t/ctrl_knot_delta_t # normalize to (0,1)
            
            # tau = torch.tensor(tau).reshape([1,1])
            # segment = torch.stack([control_knot_poses[idx_start], control_knot_poses[idx_end]], dim=0).unsqueeze(0)
            # pose_interp = linear_interpolation(segment, tau)[0][0]  # SE3, [G]
            # pose_interp = pose_interp.Log() # se3, [6]
            
            pose_interp_se3_lie =  control_knot_poses[idx_start]*(1.-tau) +  control_knot_poses[idx_end]*tau
            pose_interp = pose_interp_se3_lie.Exp()
            
        elif mode == 'cspline':
            if np.abs(query_t - control_knot_ts[1]) < 1e-6:
                query_t = query_t + 1e-6
            if np.abs(query_t - control_knot_ts[-2]) < 1e-6:
                query_t = query_t - 1e-6

            assert query_t >= control_knot_ts[1]
            assert query_t <= control_knot_ts[-2]
            
            idx_end = np.searchsorted(control_knot_ts, query_t)
            idx_start = idx_end - 1
            assert idx_start>=0
            assert control_knot_ts[idx_start]<=query_t
            assert control_knot_ts[idx_end]>=query_t
            
            ctrl_knot_t_start = control_knot_ts[idx_start]
            ctrl_knot_delta_t = control_knot_ts[idx_end] - control_knot_ts[idx_start]
            
            elapsed_t = query_t - ctrl_knot_t_start
            tau = elapsed_t/ctrl_knot_delta_t # normalize to (0,1)
            
            segment = torch.stack([control_knot_poses[idx_start-1],control_knot_poses[idx_start], 
                                control_knot_poses[idx_end], control_knot_poses[idx_end]], dim=0).unsqueeze(0)
            tau = torch.tensor(tau).reshape([1,1])
            
            pose_interp = cubic_bspline_interpolation(segment, tau)[0][0]  # SE3, [7]
        return pose_interp 
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def bundle_adjustment(self, BA_batch,  _traj_mode, _num_iters=100001, _seg_num=50, _opt_pose=True, _global_BA=False):
        # Bundle adjustment 
        
        num_batch = len(BA_batch)
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        preSum_event_batch = [0]
        for i in range(num_batch):
            frame_id = BA_batch[i]["frame_id"].item()
            frame_ts = self.est_c2w_ts[frame_id]
            frame_id_all.append(frame_id)
            frame_ts_all.append(frame_ts)
            cur_event = BA_batch[i]['events'].squeeze().to(self.device)
            preSum_event_batch.append(cur_event.shape[0]+preSum_event_batch[-1])
            events_all.append(cur_event)
        num_events_to_skip = self.config["num_events_to_skip"]
        preSum_event_batch[0] = preSum_event_batch[0]+num_events_to_skip
        preSum_event_batch[-1] = preSum_event_batch[-1]-num_events_to_skip
        
        events_all = torch.cat(events_all, dim=0)
        ts_all = events_all[:,2].detach().cpu().numpy()
        t_data_min =  ts_all.min()
        t_data_max = ts_all.max()
        
        # "incremental" random sampling
        event_total_num = ts_all.shape[0]
        incre_sampling_seg_num_expected = _seg_num
        min_n_winsize = self.config["BA"]["min_n_winsize"]
        max_n_winsize = self.config["BA"]["max_n_winsize"]
        
        incre_sampling_segs_end = np.linspace(10, event_total_num-10, incre_sampling_seg_num_expected).astype(int)
        start_tmp_ = np.searchsorted(incre_sampling_segs_end, max_n_winsize)
        if(incre_sampling_segs_end[start_tmp_]<=max_n_winsize):
            start_tmp_ = start_tmp_+1
        incre_sampling_segs_end = incre_sampling_segs_end[start_tmp_:]
        assert incre_sampling_segs_end.shape[0]>1
        assert incre_sampling_segs_end[0]>max_n_winsize
        events_per_seg = incre_sampling_segs_end[1]-incre_sampling_segs_end[0]
        incre_sampling_seg_num = incre_sampling_segs_end.shape[0]
        print(f"*******************************  BA events_num: {event_total_num}  *********************************")
        print(f"incremental sampling number: {incre_sampling_seg_num}")
        print(f"****events_per_seg={events_per_seg}, min_n_winsize={min_n_winsize},max_n_winsize={max_n_winsize}, ")
        
        
        if frame_id_all[-1] >= self.config["start_cspline_idx"] and self.config["traj_mode_BA"]=="cspline":
            traj_mode = "cspline"
        else:
            traj_mode = "linear"
        
        # initialize trajectories
        active_ctrl_knot_ts = []
        active_ctrl_knot_idx = frame_id_all.copy()
        active_ctrl_knot_idx.insert(0, frame_id_all[0]-1)
        
        if traj_mode == "cspline":
            active_ctrl_knot_se3 = pp.identity_se3(len(active_ctrl_knot_idx)+1) #LieTensor
            for ii in range(len(active_ctrl_knot_idx)):
                idx = active_ctrl_knot_idx[ii]
                active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx])
                active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx]
            # add another control knot
            print("============================== use cspline interplation ==================================")
            active_ctrl_knot_idx.append(active_ctrl_knot_idx[-1]+1)
            active_ctrl_knot_ts.append(active_ctrl_knot_ts[-1] + active_ctrl_knot_ts[-1]-active_ctrl_knot_ts[-2])
            # predict pose
            pred_vel = self.predict_velocity(se3_to_SE3(active_ctrl_knot_se3[-2]), se3_to_SE3(active_ctrl_knot_se3[-3]))
            active_ctrl_knot_se3[-1] = self.predict_pose_se3_lie(active_ctrl_knot_se3[-2], active_ctrl_knot_se3[-3])
            # don't sample the first chunk
            active_frame_id_all = frame_id_all.copy()
            active_frame_ts_all = frame_ts_all.copy()
            del active_frame_id_all[0]
            del active_frame_ts_all[0]
            del preSum_event_batch[0]
            # _num_iters += 500
        else:
            active_frame_id_all = frame_id_all.copy()
            active_frame_ts_all = frame_ts_all.copy()
            active_ctrl_knot_se3 = pp.identity_se3(len(active_ctrl_knot_idx)) #LieTensor
            for ii in range(len(active_ctrl_knot_idx)):
                idx = active_ctrl_knot_idx[ii]
                # TODO: ALL the idx is from 1
                active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx])
                active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx]
        
        active_ctrl_knot_ts[0] = t_data_min-0.0001
        active_ctrl_knot_ts[-1] = t_data_max+0.0001
        
        print(f"****t_data_min={t_data_min}, t_data_max={t_data_max}")
        print(f"control knot ts: {active_ctrl_knot_ts}")
        
        if self.config["use_relative_pose_to_opt"]==1:
            active_ctrl_knot_se3_rel_opt = pp.identity_se3(active_ctrl_knot_se3.shape[0]) #LieTensor
            active_ctrl_knot_se3_base = active_ctrl_knot_se3.clone()
            active_ctrl_knot_se3_rel_opt = torch.nn.Parameter(active_ctrl_knot_se3_rel_opt, requires_grad=True)        
            pose_optimizer = torch.optim.Adam([{"params": active_ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==0:
            active_ctrl_knot_se3 = torch.nn.Parameter(active_ctrl_knot_se3, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": active_ctrl_knot_se3, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==2:  # fix the first pose
            base_idx = active_ctrl_knot_idx[0]-1
            if base_idx<0:
                base_idx = 0
                print("====================== Warning: Don't fix the first pose ======================")
            ctrl_knot_len = len(active_ctrl_knot_idx)
            ctrl_knot_se3_base = self.ctrl_knot_se3_all[base_idx] 
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                ctrl_knot_se3_rel_opt[ii] = (self.ctrl_knot_se3_all[active_ctrl_knot_idx[ii]].Exp()*(ctrl_knot_se3_base.Inv().Exp())).Log()
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        else:
            raise ValueError("wrong option for pose optimization")
        
        active_num_batch = len(active_frame_id_all)
        
        blur_sigma = self.config["blur_sigma"]
        batch_cnt = 0
        loss_total = None
        pose_optimizer.zero_grad()
        
        # densification setting
        self.gs_cfg.refine_start_iter = self.config["mapping"]["refine_start_iter"]
        self.gs_cfg.refine_stop_iter= self.config["mapping"]["refine_stop_iter"]
        self.gs_cfg.refine_every = self.config["mapping"]["refine_every"]
        
        # Densification Strategy
        strategy_BA = DefaultStrategy(
            verbose=True,
            scene_scale=self.scene_scale,
            prune_opa=self.gs_cfg.prune_opa,
            grow_grad2d=self.gs_cfg.grow_grad2d,
            grow_scale3d=self.gs_cfg.grow_scale3d,
            prune_scale3d=self.gs_cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=self.gs_cfg.refine_start_iter,
            refine_stop_iter=self.gs_cfg.refine_stop_iter,
            refine_every=self.gs_cfg.refine_every,
            
            reset_every=self.gs_cfg.reset_every,
            absgrad=self.gs_cfg.absgrad,
            revised_opacity=self.gs_cfg.revised_opacity,
        )
        strategy_BA.check_sanity(self.splats, self.optimizers)
        strategy_state_BA = strategy_BA.initialize_state()
        
        training_batch_size = self.config["BA"]["training_batch_size"]
        visualize_every_iter = self.config["BA"]["visualize_every_iter"]
        if _global_BA:
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "global_BA")
        else:
            save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "BA")
        for i in range(_num_iters):
            if i%100 == 0:
                blur_sigma = blur_sigma/2
            
            batch_cnt += 1
            if self.config["use_relative_pose_to_opt"]==1:
                active_ctrl_knot_se3 = (active_ctrl_knot_se3_base.Exp()*active_ctrl_knot_se3_rel_opt.Exp()).Log()
            elif self.config["use_relative_pose_to_opt"]==2:
                active_ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log() 
            
            loss_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_no_event = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_ssim = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_white_balance = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            loss_tr = torch.zeros([1], dtype=torch.float64, device=self.device, requires_grad=True)
            
            # incremental random sampling
            random_start_end_idx = []
            for ii in range(incre_sampling_seg_num):
                end_idx_ = incre_sampling_segs_end[ii]
                winsize_ = np.random.randint(min_n_winsize, max_n_winsize)
                start_idx_ = end_idx_-winsize_
                random_start_end_idx.append([start_idx_, end_idx_])
            indices = np.random.permutation(len(random_start_end_idx)).tolist()[:training_batch_size]
            
            list_img_ev_start = []
            list_img_ev_end = []
            list_gt_events_acc = []
            list_syn_event_acc = []
            
            linlog_thres = self.config["event"]["linlog_thres"]
            # incremental random sampling
            for j in indices:
                idx_ev_start = random_start_end_idx[j][0]
                idx_ev_end = random_start_end_idx[j][1]
                selected_event_stream = events_all[idx_ev_start:idx_ev_end]
                t_ev_start = ts_all[idx_ev_start]
                t_ev_end = ts_all[idx_ev_end]
                
                T_SE3_ev_start = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, t_ev_start, mode=traj_mode)
                T_SE3_ev_end = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, t_ev_end, mode=traj_mode)
                
                c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_start = self.rasterize_splats(camtoworlds=c2w_start, render_mode="RGB+ED")
                c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                render_pkg_end = self.rasterize_splats(camtoworlds=c2w_end, render_mode="RGB+ED")
                
                img_ev_start = render_pkg_start["image"]
                img_ev_end = render_pkg_end["image"]
                
                strategy_BA.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=strategy_state_BA,
                    step=i,
                    info=render_pkg_end["info"],
                )
                
                if self.config["use_linLog"]:
                    pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                    syn_event_acc = pred_linlog_end - pred_linlog_start
                else:
                    # syn_event_acc = (log(img_ev_end) - log(img_ev_start))
                    eps = 0.1
                    syn_event_acc = torch.log(img_ev_end**2.2+eps)-torch.log(img_ev_start**2.2+eps)
                
                # compute event loss
                gt_events_acc = self.post_process_event_image(selected_event_stream, _sigma=blur_sigma)
                # if self.config["use_mask_event_loss"]:
                event_mask = (gt_events_acc.abs() != 0).float() 
                no_event_mask = (gt_events_acc.abs() == 0).float()
                
                # if self.config["dataset"] == "vector":
                #     loss_event = loss_event + (self.vector_mask * (gt_events_acc - syn_event_acc)**2).sum() / self.vector_mask.sum()
                # else:
                
                # no-event loss
                no_event_gaussian_cov = self.config["mapping"]["no_event_gaussian_cov"]
                gt_no_events = self.config["event"]["threshold"] * no_event_gaussian_cov * torch.randn_like(gt_events_acc).cuda()
                gt_no_events = gt_no_events*no_event_mask
                
                if self.config["mapping"]["color_channels"]==3:
                    gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                    gt_no_events = gt_no_events.unsqueeze(-1).repeat(1, 1, 3)
                    # gt_events_acc = np.tile(gt_events_acc[..., None], (1, 1, 3))
                    gt_events_acc = gt_events_acc*self.color_mask
                    syn_event_acc = syn_event_acc*self.color_mask
                
                if self.config["seprate_event_noevent_loss"]:
                    loss_event = loss_event + (event_mask*(gt_events_acc - syn_event_acc)**2).sum() / event_mask.sum()
                    loss_no_event = loss_no_event + (no_event_mask*(gt_no_events - syn_event_acc)**2).sum() / no_event_mask.sum()
                else:
                    gt_events_acc = gt_events_acc+gt_no_events
                    loss_event = loss_event + ((gt_events_acc - syn_event_acc)**2).mean()
                
                if self.config["mapping"]["color_channels"]==3:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
                else:
                    loss_ssim = loss_ssim + compute_ssim_loss(gt_events_acc, syn_event_acc, channel=1)
                
                list_img_ev_start.append(img_ev_start.detach())
                list_img_ev_end.append(img_ev_end.detach())
                list_gt_events_acc.append(gt_events_acc)
                list_syn_event_acc.append(syn_event_acc.detach())
            
            if self.config["mapping"]["use_white_balance_loss"]:
                # white balance loss 
                white_balance_weight = self.config["mapping"]["white_balance_weight"]
                # loss_white_balance = white_balance_weight * (img_ev_end.mean() - 0.5) ** 2
                loss_white_balance = white_balance_weight*torch.mean((img_ev_end - 0.5) ** 2)
            
            # tikhonov regularization loss
            tr_w = self.config["mapping"]["tr_loss_weight"]
            loss_tr = tr_w*tikhonov_regularization(render_pkg_start["depth"].unsqueeze(-1))
            
            loss_event = loss_event/len(indices)
            loss_ssim = loss_ssim/len(indices)
            loss_no_event = loss_no_event/len(indices)
            # summary the loss
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim
            
            # # isotropic loss
            # scaling = self.splats["scales"]  # [N, 3]
            # isotropic_loss_all = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # iso_w = self.config["mapping"]["loss_isotropic_weight"]
            # loss_isotropic = iso_w * isotropic_loss_all.mean()
            
            # loss_total = loss_event + loss_ssim + loss_isotropic+loss_white_balance
            loss_total = loss_event + loss_ssim + loss_white_balance + loss_no_event + loss_tr
            
            loss_total.backward()
            # loss_total = None
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if _opt_pose:
                pose_optimizer.step()
                pose_optimizer.zero_grad()
            
            # densification
            strategy_BA.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=strategy_state_BA,
                step=i,
                info=render_pkg_end["info"],
            )
            
            # visualization
            frame_id_vis = frame_id_all[-1]
            if i % visualize_every_iter == 0 and self.config["visualize_inter_img"]:
                print_str = f"iter {i}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}, white={loss_white_balance.item()}, noevent={loss_no_event.item()}, tr={loss_tr.item()}"
                print(print_str)
                
                with torch.no_grad():
                    # plot pose estimation
                    accum_est_poses_Rt = []
                    accum_gt_poses_Rt = []
                    
                    tag = f"chunk_BA_{num_batch:03}_{i:04}"
                    
                    # save ply file
                    # ply_path = os.path.join(save_path, f"pointCloud_f{frame_id_vis}_it{i}.ply")
                    # self.save_ply(ply_path)
                    
                    for n in range(active_num_batch):
                        k = active_frame_id_all[n]
                        ts = active_frame_ts_all[n]                        
                        accum_est_poses_Rt.append(self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode).matrix())
                        if traj_mode == "linear":
                            accum_gt_poses_Rt.append(BA_batch[n]['c2w'].squeeze())
                        else:
                            accum_gt_poses_Rt.append(BA_batch[n+1]['c2w'].squeeze())
                    
                    # plot 
                    if self.config["dataset"] == "rpg_evo":
                        accum_gt_poses_Rt = accum_est_poses_Rt
                    pose_evaluation(accum_gt_poses_Rt, accum_est_poses_Rt, 1, save_path, "pose", f"BA_f{frame_id_vis:03}_{i:04}")
                    save_pose_as_kitti_evo(accum_gt_poses_Rt, accum_est_poses_Rt, save_path, f"BA_f{frame_id_vis:03}_{i:04}")
                    
                    if self.config["visualize_inter_img"]:
                        # save intermediate images 
                        images_combined = []
                        for m in range(len(list_img_ev_start)):
                            diff = (list_gt_events_acc[m] - list_syn_event_acc[m]) **2
                            
                            if self.config["mapping"]["color_channels"] == 1:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].cpu().numpy(), self.dataset.H, self.dataset.W)
                                diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                                syn_event_acc = np.repeat(list_syn_event_acc[m].detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                                img_ev_start = np.repeat(list_img_ev_start[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                                img_ev_end = np.repeat(list_img_ev_end[m].detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            else:
                                gt_events_acc = render_ev_accumulation(list_gt_events_acc[m].sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                                img_ev_start = list_img_ev_start[m].detach().cpu().numpy()
                                img_ev_end = list_img_ev_end[m].detach().cpu().numpy()
                                syn_event_acc = list_syn_event_acc[m].detach().abs().cpu().numpy()
                                diff = diff.detach().cpu().numpy()
                            
                            image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, syn_event_acc, diff], axis=1)
                            images_combined.append(image_combined)
                        
                        images_combined = np.concatenate(images_combined, axis=0)
                        images_combined =  to8b(images_combined)
                        img_path = os.path.join(save_path, f"BA_f{frame_id_vis:03}_{i:04}" + "_img.jpg")  
                        imageio.imwrite(img_path, images_combined)
        
        if self.config["use_relative_pose_to_opt"]==1:
            active_ctrl_knot_se3 = (active_ctrl_knot_se3_base.Exp()*active_ctrl_knot_se3_rel_opt.Exp()).Log()
        elif self.config["use_relative_pose_to_opt"]==2:
            active_ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
        
        # update est_poses
        for n in range(active_num_batch):
            k = active_frame_id_all[n]
            ts = active_frame_ts_all[n]                      
            self.est_c2w_data[k] = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode).matrix().cuda()
            self.est_c2w_ts[k] = ts
        
        if self.config["visualize_inter_img"] or _global_BA:
            if self.config["render_eventview_img"]:
                # render ALL images 
                with torch.no_grad():
                    for n in range(active_num_batch):
                        k = active_frame_id_all[n]
                        ts = active_frame_ts_all[n]                      
                        T_SE3 = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode=traj_mode)
                        
                        c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                        rendered_img = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB+ED")["image"]
                                
                                
                        event_img_ = rendered_img.detach().cpu().numpy()
                        event_img_ =  to8b(event_img_)
                        im_name = f"BA_f{frame_id_vis:03}_f{k}" + "_img.jpg"
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                        
                        if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                            # render rbg camera image
                            img_rgbCam = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                            img_rgbCam = np.repeat(img_rgbCam.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_path = os.path.join(save_path, f"BA_f{frame_id_vis:03}_f{k}" + "_img_rgbCam.jpg")  
                            imageio.imwrite(img_path, to8b(img_rgbCam))
            # plot the whole trajectory
            if self.config["dataset"] == "rpg_evo":
                self.pose_gt = self.est_c2w_data
            pose_evaluation(self.pose_gt, self.est_c2w_data, 1, save_path, "pose", f"BA_f{frame_id_vis:03}_whole")
            save_pose_as_kitti_evo(self.pose_gt, self.est_c2w_data, save_path, f"BA_f{frame_id_vis:03}_whole")
        
        
        # save control knot pose
        for ii in range(len(active_ctrl_knot_idx)):
            idx = active_ctrl_knot_idx[ii]
            self.ctrl_knot_se3_all[idx] = active_ctrl_knot_se3[ii].detach().clone()  # [], cuda
        
        print("======================= finished BA  =========================")
        

    def validate_depth_and_pose(self, accumulated_data_chunks):
        import cv2

        ## Data from Wang Peng
        # fx = 600 
        # fy = 600 
        # cx = 599.5 
        # cy = 339.5 
        # H = 680
        # W = 1200 
        
        # T_Rt_src = torch.tensor([[9.062491181555123454e-01, -2.954311239679592860e-01, 3.023788796086531172e-01, -3.569159214564542326e-01], 
        #                          [-4.227440547687880690e-01, -6.333245673155291078e-01, 6.482186796076164770e-01, -6.602722315763628336e-01],
        #                          [8.759610522377010340e-17, -7.152764804085384176e-01, -6.988415818870352680e-01, 8.192365926179191460e-01],
        #                          [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]], device=self.device)
        
        # T_Rt_tar = torch.tensor([[9.381357201392316325e-01, -2.130208192217611096e-01, 2.729899287097143912e-01, -2.456696751290185499e-01],
        #                          [-3.462677729717930086e-01, -5.771326564125134340e-01, 7.396056559433482613e-01, -6.169610161336236409e-01],
        #                          [9.654847158446050965e-17, -7.883780993155031780e-01, -6.151910049079671872e-01, 7.409320022634876546e-01],
        #                          [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]], device=self.device)
        
        # img_src = cv2.imread('output/replica_test_init_rot/images_poses/frame000000.jpg', cv2.IMREAD_UNCHANGED)  #accumulated_data_chunks[idx_src]['rgb'][0].cpu().numpy()
        # img_tar = cv2.imread('output/replica_test_init_rot/images_poses/frame000010.jpg', cv2.IMREAD_UNCHANGED)  #accumulated_data_chunks[idx_tar]['rgb'][0].cpu().numpy()
        
        ## Data from Jian Huang 
        # fx = 384 
        # fy = 384 
        # cx = 384 
        # cy = 240 
        
        # poses = np.load('/run/determined/workdir/data/event_slam/replica_event_baseline-1000hz/room0-old/poses_bounds.npy')
        # T_Rt_src = torch.eye(4).to(self.device)
        # T_Rt_tar = torch.eye(4).to(self.device)

        # T_Rt_src[:3,:4] = torch.from_numpy(poses[0][:12]).reshape(3,4).to(self.device)
        # T_Rt_tar[:3,:4] = torch.from_numpy(poses[20][:12]).reshape(3,4).to(self.device)
        
        # img_src = cv2.imread('/run/determined/workdir/data/event_slam/replica_event_baseline-1000hz/room0-old/Gray/000.png')  #accumulated_data_chunks[idx_src]['rgb'][0].cpu().numpy()
        # img_tar = cv2.imread('/run/determined/workdir/data/event_slam/replica_event_baseline-1000hz/room0-old/Gray/020.png')  #accumulated_data_chunks[idx_tar]['rgb'][0].cpu().numpy()

        ##
        # fx = self.dataset.fx
        # fy = self.dataset.fy
        # cx = self.dataset.cx
        # cy = self.dataset.cy 

        idx_src = 1
        idx_tar = 4
        
        T_Rt_src = torch.eye(4).to(self.device)
        T_Rt_tar = torch.eye(4).to(self.device)

        T_Rt_src[:3,:4] = accumulated_data_chunks[idx_src]['GT_poses_Rt'][0]
        T_Rt_tar[:3,:4] = accumulated_data_chunks[idx_tar]['GT_poses_Rt'][0]

        T_se3_src = SE3_to_se3(T_Rt_src[:3,:4])
        T_se3_tar = SE3_to_se3(T_Rt_tar[:3,:4])

        img_src = accumulated_data_chunks[idx_src]['rgb'][0]
        img_tar = accumulated_data_chunks[idx_tar]['rgb'][0]

        depth_src = accumulated_data_chunks[idx_src]['depth_maps'][0]
        depth_tar = accumulated_data_chunks[idx_tar]['depth_maps'][0]
        
        ## 
        # px_tar_x = 200
        # px_tar_y = 280        
        
        # npx_tar_x = (px_tar_x - cx) / fx
        # npx_tar_y = (px_tar_y - cy) / fy

        # cv2.circle(img_tar, (px_tar_x, px_tar_y), 5, (255,0,0), 2)
        
        # for i in range(1000):
        #     depth = i * 0.01
        #     # depth = depth_tar[px_tar_y, px_tar_x]
        #     cpx_tar_x = npx_tar_x * depth
        #     cpx_tar_y = npx_tar_y * depth
        #     cpx_tar_z = depth
            
        #     T_Rt_tar2src = torch.linalg.inv(T_Rt_src) @ T_Rt_tar
            
        #     cpx_src_xyz = T_Rt_tar2src @ torch.tensor([cpx_tar_x, cpx_tar_y, cpx_tar_z, 1.], device=self.device, dtype=T_Rt_tar2src.dtype).unsqueeze(-1)
            
        #     npx_src_x = cpx_src_xyz[0] / cpx_src_xyz[2]
        #     npx_src_y = cpx_src_xyz[1] / cpx_src_xyz[2]
            
        #     px_src_x = npx_src_x * fx + cx
        #     px_src_y = npx_src_y * fy + cy
    
        #     # draw circle 
        #     cv2.circle(img_src, (int(px_src_x), int(px_src_y)), 2, (255,0,0), 2)

        # img_combined = np.concatenate([img_src, img_tar], axis=1)
        # img_combined = to8b(img_combined)

        # img_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "validate_pose_and_depth.jpg")
        # imageio.imwrite(img_path, img_combined)

        # warp image        
        syn_img_src, syn_msk_src = self.warp_image(T_se3_src, T_se3_tar, depth_src, img_tar)

        image_combined0 = torch.cat([img_src.mean(dim=2), img_tar.mean(dim=2)], dim=1)
        image_combined1 = torch.cat([1./depth_src, 1./depth_tar], dim=1)
        image_combined2 = torch.cat([syn_img_src.mean(dim=2), syn_msk_src * (img_src - syn_img_src).abs().mean(dim=2)], dim=1)
        
        image_combined = torch.cat([image_combined0, image_combined1, image_combined2], dim=0)

        image_combined = image_combined.cpu().numpy()
        image_combined =  to8b(image_combined)
        img_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "validate_pose_and_depth.jpg")
        imageio.imwrite(img_path, image_combined)

        sys

        return 
    
    def tracking(self, cur_frame_id, tracking_batch, niter=1000):
        num_batch = len(tracking_batch)
        
        print(f"Tracking frame{cur_frame_id-num_batch+1} to frame{cur_frame_id}")
        
        frame_ts_all = []
        frame_id_all = []
        events_all = []
        for i in range(num_batch):
            fid_ = tracking_batch[i]["frame_id"].item()
            frame_ts = tracking_batch[i]["pose_ts"].item()
            
            frame_id_all.append(fid_)
            frame_ts_all.append(frame_ts)
            
            cur_event = tracking_batch[i]['events'].squeeze()
            events_all.append(cur_event.to(self.device))
        events_all = torch.cat(events_all, dim=0)
        
        t_data_start = events_all[0][2].item()
        t_data_end = events_all[-1][2].item()
        
        # control knot management        
        active_ctrl_knot_idx = []
        for ii in range(len(frame_id_all)):
            print("add a new control knot")
            idx_ = frame_id_all[ii]
            self.ctrl_knot_ts_all[idx_] =  frame_ts_all[ii]
            active_ctrl_knot_idx.append(idx_)
            if not idx_ in self.ctrl_knot_se3_all:
                if self.config["use_gt_pose_to_opt"]:
                    self.ctrl_knot_se3_all[idx_] = SE3_to_se3(self.pose_gt[idx_])
                else:
                    # predict next control pose
                    self.ctrl_knot_se3_all[idx_] = self.predict_pose_se3_lie(self.ctrl_knot_se3_all[idx_-1], self.ctrl_knot_se3_all[idx_-2])
        
        if frame_ts_all[-1]>t_data_end:
            print(f"&&&&&&&&&&&        WARNNING: t_data_end-frame_ts_all[-1]={t_data_end-frame_ts_all[-1]}    &&&&&&&&&&&")
            print("&&&&&&&&&&&        Hard assignment:  frame_ts_all[-1] = t_data_end-1e-4    &&&&&&&&&&&")
            frame_ts_all[-1] = t_data_end-1e-4
        
        active_ctrl_knot_idx.insert(0, active_ctrl_knot_idx[0]-1)
        ctrl_knot_ts = []
        for idx_ in active_ctrl_knot_idx:
            ctrl_knot_ts.append(self.ctrl_knot_ts_all[idx_])
        # ctrl_knot_len = len(active_ctrl_knot_idx)
        
        # change to 2-knot batch tracking mode
        ctrl_knot_len = 2
        active_ctrl_knot_idx_src = active_ctrl_knot_idx.copy()
        ctrl_knot_ts_src = ctrl_knot_ts.copy()
        active_ctrl_knot_idx = [active_ctrl_knot_idx[0], active_ctrl_knot_idx[-1]]
        ctrl_knot_ts = [ctrl_knot_ts[0], ctrl_knot_ts[-1]]
        
        t_event_min = events_all[:, 2].min().cpu().numpy()
        t_event_max = events_all[:, 2].max().cpu().numpy()
        ctrl_knot_ts[0] = t_event_min-1e-6
        ctrl_knot_ts[-1] = t_event_max+1e-6
        
        if self.config["use_relative_pose_to_opt"]==1:
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len) #LieTensor
            ctrl_knot_se3_base = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                idx_ = active_ctrl_knot_idx[ii]
                ctrl_knot_se3_base[ii] = self.ctrl_knot_se3_all[idx_]
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==0:
            ctrl_knot_se3 = pp.identity_se3(ctrl_knot_len) #LieTensor
            for ii in range(ctrl_knot_len):
                idx_ = active_ctrl_knot_idx[ii]
                ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[idx_]
            ctrl_knot_se3 = torch.nn.Parameter(ctrl_knot_se3, requires_grad=True)        
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3, "lr": self.config['pose_lr']}])
        elif self.config["use_relative_pose_to_opt"]==2:  # fix the first pose
            ctrl_knot_se3_base = self.ctrl_knot_se3_all[active_ctrl_knot_idx[0]-1] 
            ctrl_knot_se3_rel_opt = pp.identity_se3(ctrl_knot_len)
            for ii in range(ctrl_knot_len):
                ctrl_knot_se3_rel_opt[ii] = (self.ctrl_knot_se3_all[active_ctrl_knot_idx[ii]].Exp()*(ctrl_knot_se3_base.Inv().Exp())).Log()
            ctrl_knot_se3_rel_opt = torch.nn.Parameter(ctrl_knot_se3_rel_opt, requires_grad=True)
            pose_optimizer = torch.optim.Adam([{"params": ctrl_knot_se3_rel_opt, "lr": self.config['pose_lr']}])
        else:
            raise ValueError("wrong option for pose optimization")

        # optimize the pose of new added control knot
        num_totel_events = events_all.shape[0]
        _num_events_for_optim = self.config["num_events_window_for_tracking"]
        num_events_to_skip = self.config["num_events_to_skip"]
        
        # num_events_for_optim = np.min([num_totel_events-1, _num_events_for_optim])
        num_events_for_optim = np.min([_num_events_for_optim, num_totel_events-2*num_events_to_skip])-1
        print(f"*******************************  events_num: {num_totel_events}  *********************************")
        print(f"************************  N_window for init: {num_events_for_optim}  ***************")
        prev_loss = 1e6 
        
        mask_boundary_size = self.config["mask_boundary_size"]
        blur_sigma = self.config["blur_sigma"]
        
        linlog_thres = self.config["event"]["linlog_thres"]
        for i in range(niter):
            if i%100 == 0:
                blur_sigma = blur_sigma/2
            if self.config["use_relative_pose_to_opt"]==1:
                ctrl_knot_se3 = (ctrl_knot_se3_base.Exp()*ctrl_knot_se3_rel_opt.Exp()).Log()
            elif self.config["use_relative_pose_to_opt"]==2:
                ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
            
            # sample random chunks of events with num_events_for_optim events in total
            
            idx_ev_start = np.random.randint(num_events_to_skip, num_totel_events-num_events_for_optim-num_events_to_skip)
            t_ev_start = events_all[idx_ev_start][2].item()
            t_ev_end = events_all[idx_ev_start + num_events_for_optim][2].item()

            # interpolate event start pose + end pose
            T_SE3_ev_start = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_start, mode='linear')
            T_SE3_ev_end = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, t_ev_end, mode='linear')
            
            # forward
            c2w_start = T_SE3_ev_start.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
            render_pkg_start = self.rasterize_splats(camtoworlds=c2w_start, render_mode="RGB")
            c2w_end = T_SE3_ev_end.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
            render_pkg_end = self.rasterize_splats(camtoworlds=c2w_end, render_mode="RGB")
            
            img_ev_start = render_pkg_start["image"]
            img_ev_end = render_pkg_end["image"]
            
            if self.config["use_linLog"]:
                pred_linlog_start = lin_log(img_ev_start*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                pred_linlog_end = lin_log(img_ev_end*255, linlog_thres=linlog_thres) # (B, Nevs, 1)
                syn_event_acc = pred_linlog_end - pred_linlog_start
            else:
                syn_event_acc = (log(img_ev_end) - log(img_ev_start))
            
            # compute event loss
            with torch.no_grad():
                gt_events_acc = self.post_process_event_image(events_all[idx_ev_start:idx_ev_start+num_events_for_optim, :], _sigma=blur_sigma)
                if self.config["only_track_event_area"]:
                    mask = (gt_events_acc.abs() > 0.1).double().detach()
                else:
                    mask = torch.ones_like(gt_events_acc)
                
                mask_boundary = torch.ones_like(mask)
                mask_boundary[:mask_boundary_size,:] = 0
                mask_boundary[-mask_boundary_size:,:] = 0
                mask_boundary[:,:mask_boundary_size] = 0
                mask_boundary[:,-mask_boundary_size:] = 0
                mask = mask * mask_boundary
                
                # visibility mask
                alpha_start = render_pkg_start["alpha"]
                alpha_start = alpha_start[0][:,:,0]  # [H, W]
                alpha_end = render_pkg_end["alpha"]
                alpha_end = alpha_end[0][:,:,0]  # [H, W]
                tracking_mask_alpha_threshold = self.config["mapping"]["tracking_mask_alpha_threshold"]
                visibility_mask = ((alpha_start > tracking_mask_alpha_threshold)*(alpha_end > tracking_mask_alpha_threshold)).double().detach()
                
                mask = mask * visibility_mask
                
                # if self.config["dataset"] == "vector":
                #     mask = mask * self.vector_mask

                if self.config["use_uncertainty_in_Tracking"]:
                    # check uncertainty 
                    img_uncertainty = 1./ (ret_ev_start['uncertainty'] * ret_ev_end['uncertainty'] + 1.0)
                    mask = mask * img_uncertainty
            
            if self.config["mapping"]["color_channels"]==3:
                gt_events_acc = gt_events_acc.unsqueeze(-1).repeat(1, 1, 3)
                gt_events_acc = gt_events_acc*self.color_mask
                syn_event_acc = syn_event_acc*self.color_mask
                mask = mask.unsqueeze(-1).repeat(1, 1, 3)

            loss_event = (mask * (gt_events_acc - syn_event_acc)**2).sum() / mask.sum()
            if self.config["mapping"]["color_channels"]==3:
                loss_ssim = compute_ssim_loss(gt_events_acc, syn_event_acc, channel=3)
            else:
                loss_ssim = compute_ssim_loss(mask*gt_events_acc, mask*syn_event_acc, channel=1)
            
            factor_ = self.config["mapping"]["ssim_loss_factor_"]
            loss_event = (1-factor_)*loss_event
            loss_ssim = factor_*loss_ssim

            loss_total = loss_event + loss_ssim

            # # TODO:check early break criteria
            # if loss_event > prev_loss * 1.5:
            #     print('[tracker]: early break at iteration:', i, 'prev_loss:', prev_loss, 'cur_loss:', loss_event.item())
            #     break
            # prev_loss = loss_event.item()
            
            # backward
            pose_optimizer.zero_grad()
            loss_total.backward()
            torch.cuda.synchronize()
            if self.config["opt_pose"]:
                pose_optimizer.step()
            
            # visualization
            if i % 100 == 0 and self.config["visualize_inter_img"]:
                tracking_info_ = f"iter {i}, loss={loss_total.item()}, event={loss_event.item()}, ssim={loss_ssim.item()}"
                print(tracking_info_)
                path_to_save = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], 'tracking')
                with torch.no_grad():
                    if self.config["visualize_inter_img_in_Tracking_BA"]:
                        # mask
                        gt_events_acc =  gt_events_acc*mask
                        syn_event_acc = syn_event_acc*mask
                        img_ev_start = img_ev_start*mask
                        img_ev_end = img_ev_end*mask
                        
                        # visualize event map using red-blue image
                        diff = (gt_events_acc - syn_event_acc) * 2
                        
                        if self.config["mapping"]["color_channels"] == 1:
                            diff = np.repeat(diff.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)  
                            # render_ev_accumulation(diff.cpu().numpy(), self.dataset.H, self.dataset.W)
                            gt_events_acc = render_ev_accumulation(gt_events_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                            syn_event_acc = np.repeat(syn_event_acc.detach().abs().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2) 
                            # render_ev_accumulation(syn_event_acc.cpu().numpy(), self.dataset.H, self.dataset.W)
                            img_ev_start = np.repeat(img_ev_start.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_ev_end = np.repeat(img_ev_end.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            mask = torch.tile(mask[:,:,None], (1,1,3))
                        else:
                            gt_events_acc = render_ev_accumulation(gt_events_acc.sum(-1).cpu().numpy(), self.dataset.H, self.dataset.W)
                            img_ev_start = img_ev_start.detach().cpu().numpy()
                            img_ev_end = img_ev_end.detach().cpu().numpy()
                            syn_event_acc = syn_event_acc.detach().cpu().numpy()
                            diff = diff.detach().cpu().numpy()
                        
                        image_combined = np.concatenate([img_ev_start,img_ev_end, gt_events_acc, syn_event_acc, diff], axis=1)
                        image_combined =  to8b(image_combined)
                        frame_id = cur_frame_id
                        im_name = f"tracking_f{frame_id:03}_{i:04}_img.jpg"
                        imageio.imwrite(os.path.join(path_to_save, im_name), image_combined)
                        
                        # visualize mask
                        alpha_end = torch.tile(alpha_end[:,:,None], (1,1,3))
                        alpha_img_end = alpha_end.detach().cpu().numpy()
                        alpha_start = torch.tile(alpha_start[:,:,None], (1,1,3))
                        alpha_img_start = alpha_start.detach().cpu().numpy()
                        
                        alpha_start_mask = (alpha_start > 0.8).double().detach()
                        alpha_start_mask = alpha_start_mask.detach().cpu().numpy()
                        
                        alpha_end_mask = (alpha_end > 0.8).double().detach()
                        alpha_end_mask = alpha_end_mask.detach().cpu().numpy()
                        
                        visibility_mask = torch.tile(visibility_mask[:,:,None], (1,1,3))
                        vis_mask_img = visibility_mask.detach().cpu().numpy()
                        
                        mask_img = mask.detach().cpu().numpy()
                        mask_img1 = np.concatenate([alpha_img_start, alpha_img_end, mask_img], axis=1)
                        mask_img2 = np.concatenate([alpha_start_mask, alpha_end_mask, vis_mask_img], axis=1)
                        tracking_mask_img = np.concatenate([mask_img1, mask_img2], axis=0)
                        tracking_mask_img =  to8b(tracking_mask_img)
                        im_name = f"tracking_f{frame_id:03}_{i:04}_mask_img.jpg"
                        imageio.imwrite(os.path.join(path_to_save, im_name), tracking_mask_img)
                    
                    # self.est_c2w_data[cur_frame_id] = se3_to_SE3_m44(self.get_poses(mutable_control_knot_poses, ctrl_knot_ts, frame_ts, mode='linear'))
                    for ii in range(len(frame_ts_all)):
                        ts = frame_ts_all[ii]
                        idx = frame_id_all[ii]
                        self.est_c2w_data[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').matrix().cuda()
                    if self.config["dataset"] == "rpg_evo":
                        self.pose_gt = self.est_c2w_data
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, path_to_save,"pose", f"tracking_f{frame_id:03}_{i:04}")
                    save_pose_as_kitti_evo(self.pose_gt, self.est_c2w_data, path_to_save, f"tracking_f{frame_id:03}_{i:04}")
        
        if self.config["use_relative_pose_to_opt"]==1:
            ctrl_knot_se3 = (ctrl_knot_se3_base.Exp()*ctrl_knot_se3_rel_opt.Exp()).Log()
        elif self.config["use_relative_pose_to_opt"]==2:
            ctrl_knot_se3 = (ctrl_knot_se3_rel_opt.Exp()*ctrl_knot_se3_base.Exp()).Log()
        
        # update est_poses
        for ii in range(len(frame_ts_all)):
            ts = frame_ts_all[ii]
            idx = frame_id_all[ii]
            self.est_c2w_data[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').matrix()
            self.est_c2w_ts[idx] = ts
        
        # save control knot pose
        # self.ctrl_knot_se3_all[cur_frame_id-1] = mutable_control_knot_poses[0].detach().clone()  # [], cuda # TODO: 保存这个control knot pose后，轨迹反而不好，待研究
        for ii in range(len(active_ctrl_knot_idx)):
            # if self.config["use_relative_pose_to_opt"]!=2:
            if ii==0:
                continue
            idx = active_ctrl_knot_idx[ii]
            self.ctrl_knot_se3_all[idx] = ctrl_knot_se3[ii].detach().clone()  # [],lieTensor, cuda
        # interpolate the control knot pose
        for ii in range(len(active_ctrl_knot_idx_src)-2):
            ii = ii+1
            ts = ctrl_knot_ts_src[ii]
            idx = active_ctrl_knot_idx_src[ii]
            self.ctrl_knot_se3_all[idx] = self.get_poses_lie(ctrl_knot_se3, ctrl_knot_ts, ts, mode='linear').Log().detach().clone()  # se3,lieTensor, cuda
        
        print("======================= finished tracking  =========================")


    def get_cspline(self, T_se3_poses, pose_ts, niters, t_offset=0.005):
        # construct spline control_knots 
        control_knot_poses = torch.zeros(4, 6, device=self.device).float()
        control_knot_poses[0] = T_se3_poses[0].detach() - 1e-4
        control_knot_poses[1] = T_se3_poses[0].detach()
        control_knot_poses[2] = T_se3_poses[-1].detach()
        control_knot_poses[3] = T_se3_poses[-1].detach() + 1e-4

        delta_t = pose_ts[-1] - pose_ts[0]
        t_start = pose_ts[0]
        offset = [-3*t_offset, -t_offset, t_offset, 3*t_offset]
        control_knot_ts = [t_start-delta_t, t_start, t_start+delta_t, t_start+2 * delta_t]
        control_knot_ts = [sum(x) for x in zip(control_knot_ts, offset)]

        control_knot_poses = torch.nn.Parameter(control_knot_poses, requires_grad=True)      
        spline_optimizer = torch.optim.Adam([{"params": control_knot_poses, "lr": 0.001}])

        for i in range(niters):
            loss = torch.tensor([0.0], requires_grad=True, device=self.device)
            for j in range(len(pose_ts)):
                inter_pose = self.get_poses(control_knot_poses, control_knot_ts, pose_ts[j], 'cspline')
                loss = loss + ((inter_pose - T_se3_poses[j].detach()) ** 2).mean()
            
            spline_optimizer.zero_grad()
            loss.backward()
            spline_optimizer.step()

            print('[get_cspline] iter:', i, loss.item())

        return control_knot_poses.detach(), control_knot_ts
    
    def convert_linear_to_cspline(self, ctrl_poses_se3, ctrl_ts, niters):
        t_start = ctrl_ts[0]
        t_end = ctrl_ts[-1]
        t_delta = (t_end - t_start) * 0.1

        pose_ts = [t_start + i * t_delta for i in range(11)]
        T_se3_poses = [self.get_poses(ctrl_poses_se3, ctrl_ts, pose_ts[i], 'linear') for i in range(len(pose_ts))]

        return self.get_cspline(T_se3_poses, pose_ts, niters, 0.005)

    
    def init_with_GT_data(self, _chunks_event, _chunks_GT_Rts, _chunks_depth_maps):
        # initialize gaussian scene
        points = []
        colors = []
        normals = []
        chunks_control_knot_se3 = []
        chunks_control_knot_ts = []

        for i in range(len(_chunks_event)):
            seg_events = _chunks_event[i]
            t_start = seg_events[0][2].item()
            t_end = seg_events[-1][2].item()
            t_delta = t_end - t_start

            ctrl_knot_se3 = torch.rand(4, 6).to(self.device) * 0.001
            ctrl_knot_se3[0] = ctrl_knot_se3[0] + SE3_to_se3(_chunks_GT_Rts[i][0])
            ctrl_knot_se3[1] = ctrl_knot_se3[1] + SE3_to_se3(_chunks_GT_Rts[i][0])
            ctrl_knot_se3[2] = ctrl_knot_se3[2] + SE3_to_se3(_chunks_GT_Rts[i][-1])
            ctrl_knot_se3[3] = ctrl_knot_se3[3] + SE3_to_se3(_chunks_GT_Rts[i][-1])
            
            ctrl_knot_ts = [t_start-t_delta-0.015, t_start-0.005, t_end+0.005, t_end+t_delta+0.015]

            chunks_control_knot_se3.append(ctrl_knot_se3)
            chunks_control_knot_ts.append(ctrl_knot_ts)

            pcd = self.initialize_gaussian_scene(seg_events, 3000, _chunks_GT_Rts[i][0].float(), depth_map=_chunks_depth_maps[i][0])
            points.append(pcd.points)
            colors.append(pcd.colors)
            normals.append(pcd.normals)

        points = torch.cat(points, dim=0)
        colors = torch.cat(colors, dim=0)
        normals = torch.cat(normals, dim=0)
        point_cloud = BasicPointCloud(points=points, colors=colors, normals=normals)

        # TODO: check the effect of spatial_lr_scale
        self.gs_model.create_from_tensor_pcd(point_cloud, spatial_lr_scale=1.0)
        self.gs_model.training_setup(self.gs_opt_cfg)
        return chunks_control_knot_se3, chunks_control_knot_ts
    
    def run_with_initialization(self):
        self.setup_seed(20)

        num_frames_to_skip = 151
        
        init_num_chunks = 5
        init_chunk_len = 5
        init_chunk_skip = 0
        mutable_init_chunk_skip = init_chunk_skip

        chunks_event = []
        chunks_GT_Rts = []
        chunks_pose_ts = []
        chunks_depth_maps = []

        chunks_ctrl_se3 = None
        chunks_ctrl_ts = None

        bConverted = False

        accum_events = []
        accum_est_poses_se3 = []
        accum_est_poses_ts = []
        
        # start optimization 
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        for i, batch in tqdm(enumerate(data_loader)):
            if i < num_frames_to_skip:
                continue 

            # accumulate data for initialization 
            if i < num_frames_to_skip + init_chunk_len * init_num_chunks + init_chunk_skip * (init_num_chunks-1):
                if len(chunks_event) > 0 and len(chunks_event) % init_chunk_len == 0 and mutable_init_chunk_skip > 0:
                    mutable_init_chunk_skip = mutable_init_chunk_skip - 1
                    continue 

                mutable_init_chunk_skip = init_chunk_skip

                chunks_event.append(batch['events'].squeeze().to(self.device))
                chunks_GT_Rts.append(batch['c2w'].squeeze().to(self.device))
                chunks_pose_ts.append(batch['pose_ts'].item())
                chunks_depth_maps.append(batch['depth'].squeeze().to(self.device))
            else:
                if bConverted==False:
                    chunks_event = [torch.cat(chunks_event[j:j+init_chunk_len], dim=0) for j in range(0, len(chunks_event), init_chunk_len)]
                    chunks_GT_Rts = [chunks_GT_Rts[j:j+init_chunk_len] for j in range(0, len(chunks_GT_Rts), init_chunk_len)]
                    chunks_pose_ts = [chunks_pose_ts[j:j+init_chunk_len] for j in range(0, len(chunks_pose_ts), init_chunk_len)]
                    chunks_depth_maps = [chunks_depth_maps[j:j+init_chunk_len] for j in range(0, len(chunks_depth_maps), init_chunk_len)]

                    if True:
                        chunks_ctrl_se3, chunks_ctrl_ts = self.init_with_GT_data(chunks_event, chunks_GT_Rts, chunks_depth_maps) 
                    else:
                        self.gs_model.load_ply('./output/replica_test_with_GT_pose_depth_init/model/point_cloud.npy')
                        self.gs_model.training_setup(self.gs_opt_cfg)
                
                        ctrl_knot_poses = torch.from_numpy(np.load('./output/replica_test_with_GT_pose_depth_init/model/control_knots.npy')).to(self.device)
                        ctrl_knot_ts = torch.from_numpy(np.load('./output/replica_test_with_GT_pose_depth_init/model/control_knots_ts.npy')).to(self.device)

                        chunks_ctrl_se3 = [ctrl_knot_poses[j] for j in range(init_chunk_len)]
                        chunks_ctrl_ts = [ctrl_knot_ts[j].tolist() for j in range(init_chunk_len)]

                    bConverted=True
                    chunks_traj_mode = ['cspline' for k in range(len(chunks_event))]
                    GT_poses_Rt, est_poses_Rt, _ = self.bundle_adjustment(chunks_event, chunks_ctrl_se3, chunks_ctrl_ts, chunks_traj_mode, chunks_GT_Rts, chunks_pose_ts, 2501)
                
                pred_pose = self.predict_current_pose(est_poses_Rt[-1], est_poses_Rt[-2])
                est_pose, pose_ts = self.tracking(batch, GT_poses_Rt, est_poses_Rt, est_poses_Rt[-1], pred_pose, 101)

                GT_poses_Rt.append(batch['c2w'].squeeze())
                est_poses_Rt.append(se3_to_SE3(est_pose))

                accum_events.append(batch['events'].squeeze().to(self.device))
                accum_est_poses_se3.append(est_pose)
                accum_est_poses_ts.append(pose_ts)

                # check if we need a keyframe
                if len(accum_events) == 10:
                    new_pcd = self.create_densify_points(accum_events[-1], accum_est_poses_se3[-2], accum_est_poses_se3[-1])
                    self.gs_model.add_new_points(new_pcd, 1.0)
                    
                    # get input data for chunk_BA
                    seg_events = torch.cat(accum_events[-5:], dim=0)
                    seg_ctrl_se3, seg_ctrl_ts = self.get_cspline(accum_est_poses_se3[-6:], accum_est_poses_ts[-6:], niters=20)
                    seg_GT_Rts = GT_poses_Rt[-5:]
                    seg_pose_ts = accum_est_poses_ts[-5:]

                    accum_events.clear()
                    accum_est_poses_se3.clear()
                    accum_est_poses_ts.clear()

                    chunks_event.append(seg_events)
                    chunks_ctrl_se3.append(seg_ctrl_se3)
                    chunks_ctrl_ts.append(seg_ctrl_ts)
                    chunks_GT_Rts.append(seg_GT_Rts)
                    chunks_pose_ts.append(seg_pose_ts)

                    chunks_traj_mode = ['cspline' for k in range(5)]
                    _, _, refined_ctrl_se3 = self.bundle_adjustment(chunks_event[-5:], chunks_ctrl_se3[-5:], chunks_ctrl_ts[-5:], 'cspline', chunks_GT_Rts[-5:], chunks_pose_ts[-5:], 501)

                    # update optimized pose for tracking
                    chunks_ctrl_se3[-5:] = refined_ctrl_se3
                    self.gs_model.prune_transparent_points(min_opacity=0.02)

                # self.get_cspline(est_poses_se3_tracking[-5:], est_poses_ts_tracking[-5:], niters=5)
            
    def run(self): 
        # self.setup_seed(20)
        
        init_batch = []
        BA_batch = []
        tracking_batch = []
        BA_count_cnt = 0
        creat_new_gs_cnt = 0
        
        num_optim_steps_tracking = self.config["num_opti_steps_for_tracking"]
        num_optim_steps_BA = self.config["num_opti_steps_for_BA"]
        num_opti_steps_for_global_BA = self.config["num_opti_steps_for_global_BA"]
        num_opti_steps_for_final_global_BA = self.config["num_opti_steps_for_final_global_BA"]
        refine_img_rendering_ = self.config["refine_img_rendering_"]
        
        # variable for initialization
        bInitIsDone = False

        # start optimization 
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        for i, batch in tqdm(enumerate(data_loader)):
            cur_frame_id = batch["frame_id"].item()
            
            BA_batch.append(batch)
            if(len(BA_batch) > 100):
                BA_batch.pop(0)
            
            if bInitIsDone==False:
                init_batch.append(batch)
                
                if len(init_batch) < self.config["initialization"]['num_frames_for_init']:
                    continue
                
                # estimate the depth_map
                depth_estimation_init_flag = self.config["depth_estimation_init"]
                if depth_estimation_init_flag==0:
                    print("************************ Don't use depth initialization ************************")
                    depth_map = None
                elif depth_estimation_init_flag==1:
                    print("************************ Estimate depth during initialization ************************")
                    
                    # depth estimation
                    from PIL import Image
                    from diffusers import DiffusionPipeline
                    from diffusers.utils import load_image
                    
                    self.initialization(init_batch, niters=self.config['num_opti_steps_for_depthEst_init'], traj_mode='linear', depth_map=None)
                    
                    depthEst_pretrained_model_path = self.config["depthEst_pretrained_model_path"]
                    pipe = DiffusionPipeline.from_pretrained(
                                depthEst_pretrained_model_path,
                                custom_pipeline="marigold_depth_estimation",
                                # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
                                local_files_only=True
                            )
                    pipe.to("cuda")
                    
                    bDepthIsEst = True
                    
                    init_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], 'initialization')
                    save_dir_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"])
                    img_path = os.path.join(init_path, f"frame{cur_frame_id}_init.jpg")
                    image: Image.Image = load_image(img_path)
                    pipeline_output = pipe(
                                            image,                  # Input image.
                                            # denoising_steps=30,     # (optional) Number of denoising steps of each inference pass. Default: 10.
                                            # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
                                            # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
                                            # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
                                            # batch_size=10,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
                                            # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
                                            # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
                                        )
                    depth: np.ndarray = pipeline_output.depth_np                    # Predicted depth map
                    depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction
                    shutil.copy(img_path, os.path.join(save_dir_path, f"init_{cur_frame_id}_rendered_img.jpg"))
                    # Save as uint16 PNG
                    depth_uint16 = (depth * 65535.0).astype(np.uint16)
                    Image.fromarray(depth_uint16).save(os.path.join(save_dir_path, "./depth_map.png"), mode="I;16")
                    # save numpy data
                    np.save(os.path.join(save_dir_path, "./depth_data.npy"), depth)
                    # Save colorized depth map
                    depth_colored.save(os.path.join(save_dir_path, "./depth_colored.png"))
                    
                    # release the GPU memory
                    del pipe
                    torch.cuda.empty_cache()
                    
                    depth_path = os.path.join(save_dir_path, "depth_data.npy")
                    print('[initialization]: load depth map from', depth_path)
                    depth_map = torch.from_numpy(np.load(depth_path)).to(self.device)
                    
                    
                elif depth_estimation_init_flag==2:
                    print("************************ Estimate depth during initialization ************************")
                    depth_path = self.config["initialization"]["depth_path"]
                    print('[initialization]: load depth map from', depth_path)
                    depth_map = torch.from_numpy(np.load(depth_path)).to(self.device)
                
                # initialize...
                self.initialization(init_batch, niters=self.config['num_opti_steps_for_init'], traj_mode='linear', depth_map=depth_map)
                
                bInitIsDone = True
            else:
                if self.config["initOnly"]:
                    # exit(0)
                    break
                print("Process frame", i+1)
                
                tracking_batch.append(batch)
                
                if len(tracking_batch) >= self.config["tracking_batch_size"]:
                    BA_count_cnt = BA_count_cnt+1
                    # tracking 
                    self.tracking(cur_frame_id, tracking_batch, num_optim_steps_tracking)
                    tracking_batch = []
                
                if BA_count_cnt >= self.config["BA_every_track"]:
                    BA_count_cnt = 0
                    creat_new_gs_cnt = creat_new_gs_cnt+1
                    if creat_new_gs_cnt >= self.config["add_new_gs_step"] and self.config["gassian_incre_growing"]:
                        events_stream = BA_batch[-1]['events'].squeeze().to(self.device)
                        new_pcd = self.create_densify_points(cur_frame_id,
                                                            events_stream,
                                                            SE3_to_se3(self.est_c2w_data[cur_frame_id-1][:3,:4]).cuda(), 
                                                            SE3_to_se3(self.est_c2w_data[cur_frame_id][:3,:4]).cuda())
                        # self.gs_model.add_new_points(new_pcd, 8.25)
                        feature_dim = 32 if self.gs_cfg.app_opt else None
                        print(f"============================= point num:{new_pcd.points.shape[0]} ===============================")
                        if(new_pcd.points.shape[0]>50):
                            new_gs = pcd_2_gs(
                                points= new_pcd.points.detach(),
                                init_opacity=self.gs_cfg.init_opa,
                                init_scale=self.gs_cfg.init_scale,
                                scene_scale=self.scene_scale,
                                sh_degree=self.gs_cfg.sh_degree,
                                feature_dim=feature_dim,
                                device=self.device,
                                )
                            add_new_gs(self.splats, self.optimizers, new_gs)
                            print("Number of New GS:", len(new_gs["means"]))
                            print("Number of GS:", len(self.splats["means"]))
                        else:
                            print("Not enough points to create a new GS, skipped.")
                    
                    sliding_window_sz = self.config["sliding_window_sz"]
                    traj_mode = 'linear'
                    seg_num_ = self.config["BA"]["incre_sampling_seg_num_expected"]
                    self.bundle_adjustment(BA_batch[-sliding_window_sz:], traj_mode, num_optim_steps_BA, seg_num_)
                
                if (i+1)>=20 and (i+1)%self.config["BA"]["global_BA_step"]==0:
                    print("******************************** global BA ********************************")
                    traj_mode = 'linear'
                    global_BA_seg = self.config["BA"]["global_BA_seg"]
                    seg_num_ = len(BA_batch)*global_BA_seg
                    if seg_num_<50:
                        seg_num_=50
                    elif seg_num_>1000:
                        seg_num_=1000
                    print(f"\n\n ------------------ num_opti_steps_for_global_BA = {num_opti_steps_for_global_BA}---------------------")
                    self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_global_BA, seg_num_, _global_BA=True)
                    if refine_img_rendering_:
                        self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_global_BA, seg_num_, _opt_pose=False,_global_BA=True)
                
        # global BA
        if self.config["global_BA"] and (not self.config["initOnly"]):
            print("******************************** global BA ********************************")
            traj_mode = 'linear'
            seg_num_ = len(BA_batch)*5
            if seg_num_<50:
                seg_num_=50
            elif seg_num_>1000:
                seg_num_=1000
            self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_final_global_BA, seg_num_, _global_BA=True)
            if refine_img_rendering_:
                self.bundle_adjustment(BA_batch, traj_mode, num_opti_steps_for_final_global_BA, seg_num_, _opt_pose=False, _global_BA=True)
        
        if self.config["evaluate_img"]:
            # render ALL images 
            with torch.no_grad():
                if self.config["use_gt_pose_to_opt"]:
                    print("******* use gt pose to render images *******")
                    for tmp_idx_ in self.dataset.val_img_idx:
                        c2w_ = self.dataset.original_gt_poses[tmp_idx_].reshape(-1, 4, 4).cuda()
                        render_pkg_ = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB")
                        event_img_ = render_pkg_["image"].detach().cpu().numpy()
                        
                        event_img_ =  to8b(event_img_)
                        im_name = f"f{tmp_idx_}_{float(self.dataset.original_gt_poses_ts[tmp_idx_]):07.3f}s.jpg"

                        save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/est")
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                else:
                    print("******* use interpolated pose to render images *******")
                    total_frame_num = len(self.dataset.val_img_idx)
                    
                    cntl_knot_sorted_keys = sorted(list(self.ctrl_knot_ts_all.keys()))
                    cntl_knot_num = len(cntl_knot_sorted_keys)
                    active_ctrl_knot_ts = []
                    active_ctrl_knot_se3 = pp.identity_se3(cntl_knot_num) #LieTensor
                    for ii in range(cntl_knot_num):
                        active_ctrl_knot_ts.append(self.ctrl_knot_ts_all[cntl_knot_sorted_keys[ii]])
                        active_ctrl_knot_se3[ii] = self.ctrl_knot_se3_all[cntl_knot_sorted_keys[ii]]
                    
                    for ii in range(total_frame_num):
                        idx_ = self.dataset.val_img_idx[ii]
                        ts = self.dataset.val_img_ts[ii]
                        T_SE3 = self.get_poses_lie(active_ctrl_knot_se3, active_ctrl_knot_ts, ts, mode="linear")
                        
                        c2w_ = T_SE3.matrix().unsqueeze(0).cuda()  #[1, 4, 4]
                        render_pkg_ = self.rasterize_splats(camtoworlds=c2w_, render_mode="RGB")
                        event_img_ = render_pkg_["image"].detach().cpu().numpy()
                        
                        event_img_ =  to8b(event_img_)
                        im_name = f"f{idx_}_{ts:07.3f}s.jpg"
                        save_path = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/est")
                        imageio.imwrite(os.path.join(save_path, im_name), event_img_)
                        
                        if self.config["render_tumvie_rgbCam_img"] and self.config["dataset"]=="tum_vie":
                            # render rbg camera image
                            img_rgbCam = self.render_with_gsplat(self.gs_model, T_se3, render_tumvie_rgb=True)['image']
                            img_rgbCam = np.repeat(img_rgbCam.detach().cpu().numpy()[:, :, np.newaxis], repeats=3, axis=2)
                            img_path = os.path.join(save_path, f"frame{idx_}_img_rgbCam_final.jpg")  
                            imageio.imwrite(img_path, to8b(img_rgbCam))
        
        print("================ finished ================")

    def depth_alignment(self, _depth_est, _depth_gs):
        # _depth_est: [N]
        # _depth_gs: [N]
        # _depth_aligned = b*_depth_est+a, we need to estimate a and b
        assert _depth_est.shape == _depth_gs.shape
        N = _depth_est.shape[0]
        s = torch.tensor([0.0]).to(_depth_est.device)
        for i in range(N):
            s = s + (_depth_est[i]-_depth_est.mean())*(_depth_gs[i]-_depth_gs.mean())
        b = s/torch.sum((_depth_est-_depth_est.mean())**2)
        a = _depth_gs.mean() - b*_depth_est.mean()
        
        _depth_aligned = b*_depth_est+a
        return _depth_aligned
    
    def create_densify_points(self, _cur_frame_id, events_stream, T_se3_start, T_se3_end, num_new_pts=1000):
        c2w_start = T_se3_start.Exp().matrix().unsqueeze(0).cuda()  #[1, 4, 4]
        render_ev_start = self.rasterize_splats(camtoworlds=c2w_start, render_mode="RGB+ED")
        c2w_end =  T_se3_end.Exp().matrix().unsqueeze(0).cuda()  #[1, 4, 4]
        render_ev_end = self.rasterize_splats(camtoworlds=c2w_end, render_mode="RGB+ED")
        
        eps = 1e-5
        syn_event_acc = torch.log(render_ev_end['image']**2.2+eps)-torch.log(render_ev_start['image']**2.2+eps)
        
        gt_event_map = self.post_process_event_image(events_stream)

        # error_map = (gt_event_map - syn_event_acc).abs()
        # pixel_uv_with_event = torch.where(error_map > 0.2)
        # sampled_event_pixel_idx = torch.randperm(pixel_uv_with_event[0].shape[0])[:num_new_pts]
        # indice_h = pixel_uv_with_event[0][sampled_event_pixel_idx]
        # indice_w = pixel_uv_with_event[1][sampled_event_pixel_idx]
        
        create_densify_points_num = self.config["mapping"]["create_densify_points_num"]
        densify_alpha_threshold = self.config["mapping"]["densify_alpha_threshold"]
        # non-visibility mask
        alpha_start = render_ev_end["alpha"].detach()
        alpha_start = alpha_start[0][:,:,0]  # [H, W]
        non_vis_mask = torch.where(alpha_start < densify_alpha_threshold)
        sampled_event_pixel_idx = torch.randperm(non_vis_mask[0].shape[0])[:create_densify_points_num]
        indice_h = non_vis_mask[0][sampled_event_pixel_idx].to(self.device)
        indice_w = non_vis_mask[1][sampled_event_pixel_idx].to(self.device)

        #
        fx = self.dataset.fx
        fy = self.dataset.fy
        cx = self.dataset.cx
        cy = self.dataset.cy
        H = self.dataset.H
        W = self.dataset.W

        sampled_rays = get_camera_rays(H, W, fx, fy, cx, cy, type='OpenCV').to(self.device) 
        sampled_rays = sampled_rays[indice_h, indice_w, :].to(T_se3_start.device)
        depth_gs = render_ev_end['depth'][indice_h, indice_w].unsqueeze(-1)
        depth_gs = depth_gs.detach()
        depth_img_gs = render_ev_end['depth']
        img_gs = render_ev_end['image'].detach().cpu().numpy()
        
        save_dir_path  = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "depth_for_new_gs")
        depth_tmp = depth_img_gs.detach()
        depth_tmp = depth_tmp.view(depth_tmp.shape[0], depth_tmp.shape[1], -1)
        depth_img = apply_colormap(depth_tmp)
        depth_img = depth_img.cpu().numpy()
        if self.config["visualize_inter_img"]:
            imageio.imwrite(os.path.join(save_dir_path, f"frame{_cur_frame_id}_gs_depth.png"), to8b(depth_img)) 
        
        if self.config["add_new_gs_from_estDepth"]:
            print("&&&&&&&&&&&&&& creat new Gaussian Points from estimated depth map &&&&&&&&&&&&&&&&&&")
            # estimate depth from the rendered image, using diffuser
            from PIL import Image
            from diffusers import DiffusionPipeline
            from diffusers.utils import load_image
            depthEst_pretrained_model_path = self.config["depthEst_pretrained_model_path"]
            pipe = DiffusionPipeline.from_pretrained(
                        depthEst_pretrained_model_path,
                        custom_pipeline="marigold_depth_estimation",
                        # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
                        local_files_only=True
                    )
            pipe.to("cuda")
            # image: Image.Image = load_image(img_path)
            image = Image.fromarray(to8b(img_gs))
            pipeline_output = pipe(
                                    image,                  # Input image.
                                    # denoising_steps=30,     # (optional) Number of denoising steps of each inference pass. Default: 10.
                                    # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
                                    # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
                                    # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
                                    # batch_size=10,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
                                    # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
                                    # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
                                )
            depth_img_est: np.ndarray = pipeline_output.depth_np  
            # save the estimated images
            depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction
            imageio.imwrite(os.path.join(save_dir_path, f"frame{_cur_frame_id}_img.png"), to8b(img_gs)) 
            depth_uint16 = (depth_img_est * 65535.0).astype(np.uint16)
            Image.fromarray(depth_uint16).save(os.path.join(save_dir_path, f"frame{_cur_frame_id}_depth_map.png"), mode="I;16")
            depth_colored.save(os.path.join(save_dir_path, f"frame{_cur_frame_id}_depth_colored.png"))
            # 
            
            depth_est = torch.from_numpy(depth_img_est).cuda()[indice_h, indice_w].unsqueeze(-1)
            depth_aligned = self.depth_alignment(depth_est, depth_gs)
        else:
            depth_aligned = depth_gs

        # initialize points on a frontal parallel plane at infinity
        # depth = 100
        sampled_rays = (sampled_rays * depth_aligned).transpose(1,0).to(T_se3_start.device)

        #
        T_cam_to_wld = se3_to_SE3(T_se3_start)
        sampled_rays = torch.matmul(T_cam_to_wld[:3, :3], sampled_rays) + T_cam_to_wld[:3, 3].unsqueeze(-1) 
        points = sampled_rays.transpose(1, 0)

        # colors
        # colors = render_ev_start['image'][indice_h, indice_w].unsqueeze(-1).repeat(1, 3)
        shs = torch.rand((points.shape[0], 3)) / 255.0

        # normals
        normals = torch.zeros_like(points)
        normals[:, 2] = 1.

        # create pcd
        # pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=normals)

        return pcd
    
    def accumulate_event_to_img(self, height, width, events_stream, polarity_offset=0.0):
        if self.config["dataset"] == 'eventnerf_real':
            events_map = np.zeros((self.dataset.H, self.dataset.W))
            xs = events_stream[:, 0].detach().cpu().numpy()
            ys = events_stream[:, 1].detach().cpu().numpy()
            ps = events_stream[:, 3].detach().cpu().numpy()
            ts = events_stream[:, 2].detach().cpu().numpy()
            polarity_offset = self.config["polarity_offset"]
            resolution_level = 1
            xs = xs.astype(np.int16)
            ys = ys.astype(np.int16)
            accumulate_events(xs, ys, ts, ps, events_map, resolution_level, polarity_offset)
            events_map = torch.from_numpy(events_map).float().cuda()
        else:
            x_window = events_stream[:, 0]
            y_window = events_stream[:, 1]
            pol_window = events_stream[:, 3]  
            polarity_offset = self.config["polarity_offset"]
            pol_window = pol_window + polarity_offset  # offset for polarity
            assert pol_window.shape[0] > 0
        
            # create indices for sparse tensor
            events_num = x_window.shape[0]
            indices= torch.cat([y_window.unsqueeze(0), x_window.unsqueeze(0), torch.arange(events_num).unsqueeze(0).to(events_stream.device)], dim=0)
            events_map_sparse = torch.sparse_coo_tensor(indices=indices, values=pol_window, size=(height, width, events_num))
            events_map_sum = torch.sparse.sum(events_map_sparse, dim=-1)

            events_map = events_map_sum.to_dense()
            
        threshold = self.config["event"]["threshold"]
        events_map =  events_map * threshold
        if self.config["event"]["clip"]:
            events_map = torch.clip(events_map, self.config["event"]["clip_min"], self.config["event"]["clip_max"])
        
        # #debug
        # pos_event_mask = (events_map>0).float()
        # neg_event_mask = (events_map<0).float()
        # pos_event_avg = (events_map*pos_event_mask).sum()/pos_event_mask.sum()
        # neg_event_avg = (events_map*neg_event_mask).sum()/neg_event_mask.sum()
        # print(f"Events_map: min={events_map.min():.3f}, max={events_map.max():.3f}, mean={events_map.mean():.3f}, pos_event_avg={pos_event_avg:.3f}, neg_event_avg={neg_event_avg:.3f}")

        return events_map    

if __name__ == '__main__':            
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments for running the NICE-SLAM/iMAP*.')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str, help='output folder, this have higher priority, can overwrite the one in config file')
    
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pparams_ = PipelineParams(parser)

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output
    
    print(f"\n ***************************************************")
    print(f" experiment name: {cfg['data']['exp_name']}")
    print(f" *************************************************** \n")

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, 'initialization')):
        os.makedirs(os.path.join(save_path, 'initialization'))
    if not os.path.exists(os.path.join(save_path, "tracking")):
        os.makedirs(os.path.join(save_path, "tracking"))
    if not os.path.exists(os.path.join(save_path, "BA")):
        os.makedirs(os.path.join(save_path, "BA"))
    if not os.path.exists(os.path.join(save_path, "global_BA")):
        os.makedirs(os.path.join(save_path, "global_BA"))
    if not os.path.exists(os.path.join(save_path, "depth_for_new_gs")):
        os.makedirs(os.path.join(save_path, "depth_for_new_gs"))
    if not os.path.exists(os.path.join(save_path, "img_eval")):
        os.makedirs(os.path.join(save_path, "img_eval"))
    if not os.path.exists(os.path.join(save_path, "img_eval/gt")):
        os.makedirs(os.path.join(save_path, "img_eval/gt"))
    if not os.path.exists(os.path.join(save_path, "img_eval/est")):
        os.makedirs(os.path.join(save_path, "img_eval/est"))
    # shutil.copy("coslam_blur.py", os.path.join(save_path, 'coslam_blur.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))
        
    # copy the config file to output directory!!!
    config_file_name = args.config.split("/")[-1]
    shutil.copy(args.config, os.path.join(save_path, config_file_name))

    ###############
    slam = SLAM(cfg)

    slam.run()
