import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from .utils import get_camera_rays, alphanum_key, as_intrinsics_matrix
import cv2

import json
import h5py
import hdf5plugin
# import pypose as pp
import tqdm
from tools.pose_utils import *
from tools.event_utils import EventSlicer
from utils import find_nearest_index, pose2homo

from spline.cubicSpline import SE3_to_se3_N, se3_to_SE3
import yaml
import pypose as pp
import shutil

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    if config['dataset'] == 'simu_event':
        dataset = SimuEventDataset
    
    elif config['dataset'] == 'replica_simu_event':
        dataset = ReplicaEventDataset
    
    elif config['dataset'] == 'tum_vie':
        dataset = TUM_VIEDataset
    
    else:
        raise NotImplementedError()
    
    return dataset(config, 
                   config['data']['datadir'], 
                   trainskip=config['data']['trainskip'], 
                   downsample_factor=config['data']['downsample'], 
                   sc_factor=config['data']['sc_factor'])

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def parse_txt(filename, shape):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape(shape).astype(np.float32)
    
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def interpolate_pose_linear(control_knot_poses, control_knot_ts, query_t):
    if np.abs(query_t - control_knot_ts[0]) < 1e-6:
        query_t = query_t + 1e-6
    if np.abs(query_t - control_knot_ts[-1]) < 1e-6:
        query_t = query_t - 1e-6

    assert query_t >= control_knot_ts[0]
    assert query_t <= control_knot_ts[-1]
    
    query_t = torch.tensor(query_t)
    control_knot_ts = torch.tensor(control_knot_ts)
    idx_end = torch.searchsorted(control_knot_ts, query_t)
    idx_start = idx_end - 1
    
    ctrl_knot_t_start = control_knot_ts[0]
    ctrl_knot_delta_t = control_knot_ts[1] - control_knot_ts[0]
    
    elapsed_t = query_t - ctrl_knot_t_start
    # idx = int(elapsed_t / ctrl_knot_delta_t)

    tau = (elapsed_t % ctrl_knot_delta_t) / ctrl_knot_delta_t # normalize to (0,1)
    pose_interp = (1. - tau) * control_knot_poses[idx_start] + tau * control_knot_poses[idx_end] 
    
    return pose_interp


class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H']//cfg['data']['downsample'],\
            cfg['cam']['W']//cfg['data']['downsample']

        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        
    
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()


def get_event_chunk(events, t_lower, t_upper):
    # slice [t_lower, t_upper)
    index_lower_bound = events[:, 2] >= t_lower
    index_upper_bound = events[:, 2] < t_upper
    selected_idx = index_lower_bound*index_upper_bound
    selcted_events = events[selected_idx, :]
    return selcted_events


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    # poses_[:, :3, :4] = poses[:, :3, :4]
    # poses = poses_
    return poses

class SimuEventDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(SimuEventDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/RGB/*.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/depth/*.exr'))
        poses_ts_path = os.path.join(basedir, 'poses_ts.txt')
        gt_pose_path = os.path.join(self.basedir, 'groundtruth.txt')
        # load pose
        self.poses_ts = np.loadtxt(poses_ts_path)
        if self.config["mapping"]["use_pose_format_of_vanilla_nerf"]:
            poses_file_path = os.path.join(basedir, "poses_bounds.npy")
            poses_arr = np.load(poses_file_path)
            poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3x5xN
            poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)    # 列的转换    -y x z : x y z
            poses = np.moveaxis(poses, -1, 0).astype(np.float32)
            bds = poses_arr[:, -2:].transpose([1, 0])  # 2xN
            print('bd: ', bds.min(), bds.max())
            # Rescale if bd_factor is provided
            bd_factor = 0.75
            sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
            poses[:, :3, 3] *= sc  # T
            bds *= sc
            # recenter
            poses = recenter_poses(poses)  # 对位置进行了中心变化
            poses = torch.from_numpy(poses).float()
            self.poses = poses
        else:
            self.poses = self.load_poses_from_gt(gt_pose_path, poses_ts_path) # list of c2w [4,4]. TODO: stack as np

        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.events = np.load(os.path.join(basedir, 'event_threshold_0.1', 'gray_events_data.npy'))
        print("finished event loading!")

    def __len__(self):
        return self.num_frames-1

    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        # depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor
        
        H, W = depth_data.shape[0],depth_data.shape[1]
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float64))
        depth_data = torch.from_numpy(depth_data.astype(np.float64))
        
        selected_events = get_event_chunk(self.events, self.poses_ts[index], self.poses_ts[index+1])
        
        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
            "pose_ts": self.poses_ts[index],
            "events": selected_events
        }
        return ret

    def load_poses_from_gt(self, gt_pose_path, poses_ts_path):
        # load groundtruth poses
        gt_poses = {}
        bottom = np.array([0, 0, 0, 1]).reshape([1, 4])
        with open(gt_pose_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    # image_id = int((float(elems[0])-4153.89)*100)
                    pose_ts_str = elems[0][:8]  # Only use first 8 characters
                    wxyz = [elems[7], elems[4], elems[5], elems[6]]
                    qvec = np.array(tuple(map(float, wxyz)))  # wxyz
                    rot_mat = qvec2rotmat(qvec)
                    tvec = np.array(tuple(map(float, elems[1:4])))
                    tvec = tvec.reshape(3, 1)
                    m_c2w = np.concatenate([np.concatenate([rot_mat, tvec], 1), bottom], 0)
                    gt_poses[pose_ts_str] = m_c2w
        
        # load pose timestamp as string, which is useful for following gt poses selection
        poses_ts_str = []
        with open(poses_ts_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if len(line)>0 and line[0]!="#":
                    poses_ts_str.append(line[:8]) # Only use first 8 characters
        
        # select poses based on pose_ts
        poses = []
        for i in range(len(poses_ts_str)):
            ts_tmp = poses_ts_str[i]
            if ts_tmp in gt_poses.keys():
                c2w = gt_poses[ts_tmp]
                # c2w = torch.from_numpy(c2w).float().cuda()
                c2w = torch.from_numpy(c2w).float()
                poses.append(c2w)
            else:
                raise KeyError(f"Key not found: {ts_tmp}")
        poses = torch.stack(poses, 0)
        return poses


class ReplicaEventDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(ReplicaEventDataset, self).__init__(cfg)
        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/images/*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.basedir}/Depth_exr/*.exr'))
        poses_ts_path = os.path.join(basedir, 'poses_ts.txt')
        gt_pose_path = os.path.join(self.basedir, 'groundtruth.txt')
        # load pose
        gt_poses = np.loadtxt(os.path.join(basedir, "traj.txt"))
        gt_poses_ts = np.loadtxt(poses_ts_path)
        
        self.original_img_file_paths = self.img_files.copy()
        self.original_gt_poses_ts = gt_poses_ts.copy()
        self.original_gt_poses = torch.from_numpy(gt_poses.copy().reshape(-1, 4, 4)).float()
        assert self.original_gt_poses_ts.shape[0] == len(self.original_img_file_paths)
        assert self.original_gt_poses_ts.shape[0] == self.original_gt_poses.shape[0]
        
        # select images by interval
        interval_selection = self.config["data"]["interval_selection"]
        gt_poses = gt_poses[::interval_selection]
        gt_poses_ts = gt_poses_ts[::interval_selection]
        self.img_files = self.img_files[::interval_selection]
        
        training_start_index = self.config["data"]["training_start_index"]
        training_end_index = self.config["data"]["training_end_index"]
        gt_poses_ts = gt_poses_ts[training_start_index:training_end_index]
        gt_poses = gt_poses[training_start_index:training_end_index].reshape(-1, 4, 4)
        self.img_files = self.img_files[training_start_index:training_end_index]
        
        # recenter
        # gt_poses = recenter_poses(gt_poses)
        
        self.poses = torch.from_numpy(gt_poses).float()
        self.poses_ts = gt_poses_ts
        
        self.K = np.zeros((3,3))
        self.K[0,0] = self.config["cam"]["fx"]
        self.K[0,2] = self.config["cam"]["cx"]
        self.K[1,1] = self.config["cam"]["fy"]
        self.K[1,2] = self.config["cam"]["cy"]
        self.K[2, 2] = 1.0
        self.K = self.K.astype("float32")
        
        self.rays_d = None
        self.tracking_mask = None
        self.num_frames = len(self.poses_ts)
        self.frame_ids = range(0, self.num_frames)
        
        print("begin to load event file")
        event_raw = np.load(os.path.join(basedir, 'event_threshold_0.1', 'gray_events_data.npy'))
        delta_t = self.poses_ts[1]-self.poses_ts[0]
        start_time = self.poses_ts[0]-delta_t
        end_time = self.poses_ts[-1]+delta_t
        self.events = get_event_chunk(event_raw, start_time, end_time)
        self.events = self.events.astype(np.float32)
        
        if self.config["evaluate_img"]:
            print("copying gt images")
            eval_start_idx = self.config["eval_start_idx"]
            eval_end_idx = self.config["eval_end_idx"]
            eval_step_size = self.config["eval_step_size"]
            
            self.val_img_idx = list(range(eval_start_idx, eval_end_idx, eval_step_size))
            eval_num = len(self.val_img_idx)
            self.val_img_ts = []
            
            save_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/gt")
            for i in range(eval_num):
                eval_idx_tmp = self.val_img_idx[i]
                # copy the gt image
                src_img_path = self.original_img_file_paths[eval_idx_tmp]
                src_img_ts = self.original_gt_poses_ts[eval_idx_tmp]
                src_img_path_name = f"f{eval_idx_tmp}_{float(src_img_ts):07.3f}s.jpg"
                dst_img_path = os.path.join(save_dir, src_img_path_name)
                shutil.copy(src_img_path, dst_img_path)
                self.val_img_ts.append(src_img_ts)
            
            print(f"valuating image number: {eval_num}")
            print(f"timestamp: {self.val_img_ts}\n")
            assert self.val_img_ts[0]>=self.poses_ts[0]
            assert self.val_img_ts[-1]<=self.poses_ts[-1]
        
        print("finished event loading!")

        # # process events into chunks according frame index
        # for i in range(1, self.num_frames):
        #     selected_events = get_event_chunk(self.events, self.poses_ts[i-1], self.poses_ts[i])
        #     filename = os.path.join(basedir, 'event_threshold_0.1', str(i).zfill(4)+'.npy')
        #     np.save(filename, selected_events)
        #     print('saving', filename)

        # # import sys
        # sys.exit(0)
        

    def __len__(self):
        return self.num_frames-1

    def __getitem__(self, index):
        index = index+1
        color_path = self.img_files[index]
        # depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        # if '.png' in depth_path:
        #     depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # elif '.exr' in depth_path:
        #     depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]
        # if self.distortion is not None:
        #     raise NotImplementedError()
        
        # color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        # # depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor
        
        # H, W = color_data.shape[0],color_data.shape[1]
        # color_data = cv2.resize(color_data, (W, H))

        # if self.downsample_factor > 1:
        #     H = H // self.downsample_factor
        #     W = W // self.downsample_factor
        #     color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
        #     depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        # if self.rays_d is None:
        #     self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float64))
        # depth_data = torch.from_numpy(depth_data.astype(np.float64))
        
        selected_events = get_event_chunk(self.events, self.poses_ts[index-1], self.poses_ts[index])
        # selected_events = np.load(os.path.join(self.basedir, 'event_threshold_0.1', str(index).zfill(4)+'.npy'))
        
        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            # "depth": depth_data,
            # "direction": self.rays_d,
            "pose_ts": self.poses_ts[index],
            "events": selected_events
        }
        return ret

    def load_poses_from_gt(self, gt_pose_path, poses_ts_path):
        # load groundtruth poses
        gt_poses = {}
        bottom = np.array([0, 0, 0, 1]).reshape([1, 4])
        with open(gt_pose_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    # image_id = int((float(elems[0])-4153.89)*100)
                    pose_ts_str = elems[0][:8]  # Only use first 8 characters
                    wxyz = [elems[7], elems[4], elems[5], elems[6]]
                    qvec = np.array(tuple(map(float, wxyz)))  # wxyz
                    rot_mat = qvec2rotmat(qvec)
                    tvec = np.array(tuple(map(float, elems[1:4])))
                    tvec = tvec.reshape(3, 1)
                    m_c2w = np.concatenate([np.concatenate([rot_mat, tvec], 1), bottom], 0)
                    gt_poses[pose_ts_str] = m_c2w
        
        # load pose timestamp as string, which is useful for following gt poses selection
        poses_ts_str = []
        with open(poses_ts_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if len(line)>0 and line[0]!="#":
                    poses_ts_str.append(line[:8]) # Only use first 8 characters
        
        # select poses based on pose_ts
        poses = []
        for i in range(len(poses_ts_str)):
            ts_tmp = poses_ts_str[i]
            if ts_tmp in gt_poses.keys():
                c2w = gt_poses[ts_tmp]
                # c2w = torch.from_numpy(c2w).float().cuda()
                c2w = torch.from_numpy(c2w).float()
                poses.append(c2w)
            else:
                raise KeyError(f"Key not found: {ts_tmp}")
        poses = torch.stack(poses, 0)
        return poses


class TUM_VIEDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                downsample_factor=1, translation=0.0,
                sc_factor=1., crop=0):
        super(TUM_VIEDataset, self).__init__(cfg)
        self.config = cfg
        self.basedir = self.config["data"]["datadir"]
        self.scenedir = os.path.join(self.config["data"]["datadir"], self.config["data"]["scene"])
        self.trainskip = trainskip
        self.downsample_factor = self.config["data"]["downsample_factor"]
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.scenedir}/left_images_undistorted/*.jpg'))
        poses_ts_path = os.path.join(self.scenedir, 'left_images_undistorted/image_timestamps_left.txt')
        
        # load pose
        # self.poses_ts = np.loadtxt(poses_ts_path)
        self.poses_ts_us = np.loadtxt(poses_ts_path, skiprows=1, dtype=float, comments='#',usecols=(0)) # us
        
        with open(os.path.join(self.scenedir, "calib_undist.json"), 'r') as f:
            self.calibdata = json.load(f)["value0"]
        with open(os.path.join(self.basedir, "mocap-imu-calib.json"), 'r') as f:
            self.calibdata.update(json.load(f)["value0"])
        
        mocap_pose_path = os.path.join(self.scenedir, 'mocap_data.txt')
        poses_gt_us = np.loadtxt(mocap_pose_path, skiprows=1)
        
        tss_gt_us = poses_gt_us[:, 0]
        assert np.all(tss_gt_us == sorted(tss_gt_us))
        if not np.median(np.abs(1./np.diff(tss_gt_us)*1e6 - 120.0)) < 1:
            assert np.median(np.abs(1./np.diff(tss_gt_us)*1e6 - 120.0)) < 1
        assert poses_gt_us.shape[0] > 100
        assert poses_gt_us.shape[1] == 8
        
        tss_imgs_us = self.poses_ts_us
        
        bds = np.zeros((len(tss_imgs_us), 2))
        
        eve_cam_id = 2  # left event camera
        camId = 0 # left rgb camera
        T_imu_rgbCam = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][camId])
        T_imu_evCam = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][eve_cam_id])
        T_imu_marker = quat_dict_to_pose_hom(self.calibdata["T_imu_marker"])
        
        # for rendering gt RGB images
        self.T_evCam_rgbCam = np.linalg.inv(T_imu_evCam) @ T_imu_rgbCam
        
        tss_all_poses_ns, all_trafos = quatList_to_poses_hom_and_tss(poses_gt_us)
        tss_all_poses_ns = [t * 1000 for t in tss_all_poses_ns]
        T_imu_evCam = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][eve_cam_id])
        all_trafos_c2w = np.asarray([T_mocap_marker @ np.linalg.inv(T_imu_marker) @ T_imu_evCam for T_mocap_marker in all_trafos]).squeeze()[:, :3, :]
        # all_trafos_c2w = rub_from_rdf(all_trafos_c2w[:, :3, :])
        check_rot_batch(all_trafos_c2w)
        
        poses_dict = []
        for i in range(all_trafos_c2w.shape[0]):
            poses_dict.append({"pose_c2w": all_trafos_c2w[i, :, :], "ts_ns": tss_all_poses_ns[i]})
        
        # upsampling poses_ts_us
        assert type(self.config["data"]["upsample_ts_factor"]) is int
        if self.config["data"]["upsample_ts_factor"]>1:
            poses_ts_us_tmp = []
            factor = self.config["data"]["upsample_ts_factor"]
            for ii in range(self.poses_ts_us.shape[0]-1):
                t_start = self.poses_ts_us[ii]
                t_end = self.poses_ts_us[ii+1]
                delta_t =  (t_end - t_start)/factor
                poses_ts_us_tmp.append(t_start)
                for j in range(factor-1):
                    poses_ts_us_tmp.append(t_start + delta_t*(j+1))
            self.poses_ts_us = np.array(poses_ts_us_tmp)
        
        # interpolation
        self.tss_poses_hf_ns = np.stack([p["ts_ns"] for p in poses_dict])  
        self.rots_hf = np.stack([p["pose_c2w"][:3, :3] for p in poses_dict])  
        self.trans_hf = np.stack([p["pose_c2w"][:3, 3] for p in poses_dict]) 
        self.rot_interpolator = Slerp(self.tss_poses_hf_ns, R.from_matrix(self.rots_hf)) 
        self.trans_interpolator = interp1d(x=self.tss_poses_hf_ns, y=self.trans_hf, axis=0, kind="cubic", bounds_error=True)
        poses = []
        for i in range(self.poses_ts_us.shape[0]):
            eval_tss_evs_ns = self.poses_ts_us[i]*1000
            if eval_tss_evs_ns < self.tss_poses_hf_ns[0]:
                print("&&&&&&& warning time out of range &&&&&&")
                eval_tss_evs_ns = self.tss_poses_hf_ns[1]
            elif eval_tss_evs_ns > self.tss_poses_hf_ns[-1]:
                print("&&&&&&& warning time out of range &&&&&&")
                eval_tss_evs_ns = self.tss_poses_hf_ns[-2]
            rots = self.rot_interpolator(eval_tss_evs_ns).as_matrix().astype(np.float32) 
            trans = self.trans_interpolator(eval_tss_evs_ns).astype(np.float32).reshape(3,1)
            p_m44 = np.concatenate([rots, trans], axis=1)
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)
            p_m44 = np.concatenate([p_m44, bottom], axis=0)
            poses.append(p_m44)
        poses = np.stack(poses)
        
        self.original_img_file_paths = self.img_files.copy()
        self.original_gt_poses_ts = self.poses_ts_us.copy()
        self.original_gt_poses_ts = self.original_gt_poses_ts/1e6 
        self.original_gt_poses = torch.from_numpy(poses.copy().reshape(-1, 4, 4)).float()
        assert self.original_gt_poses_ts.shape[0] == len(self.original_img_file_paths)
        assert self.original_gt_poses_ts.shape[0] == self.original_gt_poses.shape[0]
        
        # select images by interval
        interval_selection = self.config["data"]["interval_selection"]
        self.poses_ts_us = self.poses_ts_us[::interval_selection]
        poses = poses[::interval_selection]
        self.selected_frame_ids = range(0, self.poses_ts_us.shape[0])
        training_start_index = self.config["data"]["training_start_index"]
        training_end_index = self.config["data"]["training_end_index"]
        self.poses_ts_us = self.poses_ts_us[training_start_index:training_end_index]
        self.selected_frame_ids = self.selected_frame_ids[training_start_index:training_end_index]
        poses = poses[training_start_index:training_end_index]
        
        assert(poses.shape[0] == self.poses_ts_us.shape[0])
        
        poses = torch.from_numpy(poses).float()
        self.poses = poses
        
        print(f"**************** total {poses.shape[0]} frames ****************")
            
        event_path = os.path.join(self.scenedir, self.config["data"]["scene"] + "-events_left.h5")
        ef_in = h5py.File(event_path, "r") # keys(): <KeysViewHDF5 ['events', 'ms_to_idx']>
        event_slicer = EventSlicer(ef_in)
        
        delta_t = self.poses_ts_us[1]-self.poses_ts_us[0]
        start_time_us = self.poses_ts_us[0]-delta_t
        end_time_us = self.poses_ts_us[-1]+delta_t
        # FIXME: if end_time_us > ef_in["events"]["t"][-1], return None
        ev_batch = event_slicer.get_events(start_time_us, end_time_us)

        ts = ev_batch['t']  # us
        pol = ev_batch['p']
        x = ev_batch['x']
        y = ev_batch['y']
                
        ts = ts.astype("float32")/1e6   # us to s
        pol = pol.astype("float32")
        pol = pol * 2 - 1
        
        # FIXME: memory allocation is too expensive
        self.events = np.concatenate([x[:, np.newaxis], y[:, np.newaxis], ts[:, np.newaxis], pol[:, np.newaxis]], axis=1)
        
        del ts, x, y, pol
        
        # undistort
        with open(os.path.join(self.basedir, "camera-calibrationA.json"), 'r') as f:
            calibdata = json.load(f)
            rgb_cam_id = 0
            eve_cam_id = 2
            # loop all 4 cameras (0=left, 1=right, 2=left events, 3=right events)
            self.K = np.zeros((3,3))
            self.K[0,0] = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["fx"]
            self.K[0,2] = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["cx"]
            self.K[1,1] = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["fy"]
            self.K[1,2] = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["cy"]
            self.K[2, 2] = 1

            k1 = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["k1"]
            k2 = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["k2"]
            k3 = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["k3"]
            k4 = calibdata["value0"]["intrinsics"][eve_cam_id]["intrinsics"]["k4"]
            self.dist_coeffs = np.asarray([k1, k2, k3, k4]).astype(np.float32)
            
            # size of event camera
            self.W = calibdata["value0"]["resolution"][eve_cam_id][0]
            self.H = calibdata["value0"]["resolution"][eve_cam_id][1]
            
            self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.dist_coeffs, (self.W, self.H), np.eye(3), balance=0.5)
            self.K = self.K.astype(np.float32)
            self.K_new = self.K_new.astype(np.float32)
            
            # self.fx = self.K[0,0]
            # self.fy = self.K[1,1]
            # self.cx = self.K[0,2]
            # self.cy = self.K[1,2]
            self.fx = self.K_new[0,0]
            self.fy = self.K_new[1,1]
            self.cx = self.K_new[0,2]
            self.cy = self.K_new[1,2]
            
            # rgb camera
            self.K_rgb = np.zeros((3,3))
            self.K_rgb[0,0] = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["fx"]
            self.K_rgb[0,2] = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["cx"]
            self.K_rgb[1,1] = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["fy"]
            self.K_rgb[1,2] = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["cy"]
            self.K_rgb[2, 2] = 1

            k1_rgb = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["k1"]
            k2_rgb = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["k2"]
            k3_rgb = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["k3"]
            k4_rgb = calibdata["value0"]["intrinsics"][rgb_cam_id]["intrinsics"]["k4"]
            self.dist_coeffs_rgb = np.asarray([k1_rgb, k2_rgb, k3_rgb, k4_rgb]).astype(np.float32)
            
            self.K_rgb = self.K_rgb.astype(np.float32)

            # size of rgb camera
            self.W_rgb = calibdata["value0"]["resolution"][rgb_cam_id][0]
            self.H_rgb = calibdata["value0"]["resolution"][rgb_cam_id][1]
            # self.K_new_rgb = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.dist_coeffs, (self.W, self.H), np.eye(3), balance=0.5)
            # self.fx_rgb = self.K_new_rgb[0,0]
            # self.fy_rgb = self.K_new_rgb[1,1]
            # self.cx_rgb = self.K_new_rgb[0,2]
            # self.cy_rgb = self.K_new_rgb[1,2]
            
            self.fx_rgb = self.K_rgb[0,0]
            self.fy_rgb = self.K_rgb[1,1]
            self.cx_rgb = self.K_rgb[0,2]
            self.cy_rgb = self.K_rgb[1,2]
                
        # get downsampled camera intrinsics
        if self.downsample_factor > 1:
            # event camera
            self.H_old = self.H
            self.W_old = self.W
            self.K_old = self.K.copy()
            self.fx = self.K_old[0,0] / self.downsample_factor
            self.fy = self.K_old[1,1] / self.downsample_factor
            self.cx = self.K_old[0,2] / self.downsample_factor
            self.cy = self.K_old[1,2] / self.downsample_factor
            self.K[0,0] = self.K_old[0,0] / self.downsample_factor
            self.K[0,2] = self.K_old[0,2] / self.downsample_factor
            self.K[1,1] = self.K_old[1,1] / self.downsample_factor
            self.K[1,2] = self.K_old[1,2] / self.downsample_factor
            self.H = self.H_old // self.downsample_factor
            self.W = self.W_old // self.downsample_factor
            
            # rgb camera
            self.H_rgb_old = self.H_rgb
            self.W_rgb_old = self.W_rgb
            self.K_rgb_old = self.K_rgb.copy()
            self.fx_rgb = self.K_rgb_old[0,0] / self.downsample_factor
            self.fy_rgb = self.K_rgb_old[1,1] / self.downsample_factor
            self.cx_rgb = self.K_rgb_old[0,2] / self.downsample_factor
            self.cy_rgb = self.K_rgb_old[1,2] / self.downsample_factor
            self.K_rgb[0,0] = self.K_rgb_old[0,0] / self.downsample_factor
            self.K_rgb[0,2] = self.K_rgb_old[0,2] / self.downsample_factor
            self.K_rgb[1,1] = self.K_rgb_old[1,1] / self.downsample_factor
            self.K_rgb[1,2] = self.K_rgb_old[1,2] / self.downsample_factor
            self.H_rgb = self.H_rgb_old // self.downsample_factor
            self.W_rgb = self.W_rgb_old // self.downsample_factor
        
        self.rays_d = None
        self.tracking_mask = None
        self.num_frames = len(self.poses_ts_us)
        self.frame_ids = range(0, self.num_frames)
        self.poses_ts = self.poses_ts_us/1e6      # us to s
        
        if self.config["evaluate_img"]:
            print("copying gt images")
            eval_start_idx = self.config["eval_start_idx"]
            eval_end_idx = self.config["eval_end_idx"]
            eval_step_size = self.config["eval_step_size"]
            
            self.val_img_idx = list(range(eval_start_idx, eval_end_idx, eval_step_size))
            eval_num = len(self.val_img_idx)
            self.val_img_ts = []
            
            save_dir = os.path.join(self.config["data"]["output"], self.config["data"]["exp_name"], "img_eval/gt")
            for i in range(eval_num):
                eval_idx_tmp = self.val_img_idx[i]
                # copy the gt image
                src_img_path = self.original_img_file_paths[eval_idx_tmp]
                src_img_ts = self.original_gt_poses_ts[eval_idx_tmp]
                src_img_path_name = f"f{eval_idx_tmp}_{float(src_img_ts):07.3f}s.jpg"
                dst_img_path = os.path.join(save_dir, src_img_path_name)
                shutil.copy(src_img_path, dst_img_path)
                self.val_img_ts.append(src_img_ts)
            
            print(f"valuating image number: {eval_num}")
            print(f"timestamp: {self.val_img_ts}\n")
            assert self.val_img_ts[0]>=self.poses_ts[0]
            assert self.val_img_ts[-1]<=self.poses_ts[-1]
        
        print("finished event loading!")

    def reset_downsample_factor(self, downsample_factor):
        self.fx = self.K_old[0,0] / downsample_factor
        self.fy = self.K_old[1,1] / downsample_factor
        self.cx = self.K_old[0,2] / downsample_factor
        self.cy = self.K_old[1,2] / downsample_factor
        self.K[0,0] = self.K_old[0,0] / downsample_factor
        self.K[0,2] = self.K_old[0,2] / downsample_factor
        self.K[1,1] = self.K_old[1,1] / downsample_factor
        self.K[1,2] = self.K_old[1,2] / downsample_factor
        self.H = self.H_old // downsample_factor
        self.W = self.W_old // downsample_factor

    def __len__(self):
        return self.num_frames-1

    def __getitem__(self, index):
        # color_path = self.img_files[index]
        # depth_path = self.depth_paths[index]

        # color_data = cv2.imread(color_path)
        # if '.png' in depth_path:
        #     depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # elif '.exr' in depth_path:
        #     depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # if self.distortion is not None:
        #     raise NotImplementedError()

        # color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        # color_data = color_data / 255.
        # depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor
        
        H, W = self.H_rgb, self.W_rgb
        # color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            # color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            # depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        # color_data = torch.from_numpy(color_data.astype(np.float64))
        # depth_data = torch.from_numpy(depth_data.astype(np.float64))
        
        selected_events = get_event_chunk(self.events, self.poses_ts[index], self.poses_ts[index+1])
        
        ret = {
            "frame_id": self.frame_ids[index+1],
            "img_frame_id": self.selected_frame_ids[index+1],
            "c2w": self.poses[index+1],
            # "rgb": color_data,
            # "depth": depth_data,
            "direction": self.rays_d,
            "pose_ts": self.poses_ts[index+1],
            "events": selected_events
        }
        return ret

    def load_poses_from_gt(self, gt_pose_path, imgs_ts, filter_imgs_num):
        # load groundtruth poses
        bottom = np.array([0, 0, 0, 1]).reshape([1, 4])
        extract_gt_pose = []
        extract_imgs_ts = []
        extract_frames_id = []
        gt_pose = np.loadtxt(gt_pose_path)
        for i in range(imgs_ts.shape[0]):
            idx = find_nearest_index(arr=gt_pose[:,0], value=imgs_ts[i])
            pose = gt_pose[idx, 1:]
            qvec = [pose[6], pose[3], pose[4], pose[5]]
            rot_mat = qvec2rotmat(qvec)
            tvec = pose[0:3]
            tvec = tvec.reshape(3, 1)
            m_c2w = np.concatenate([np.concatenate([rot_mat, tvec], 1), bottom], 0)
            extract_gt_pose.append(m_c2w)
            extract_imgs_ts.append(imgs_ts[i])
            extract_frames_id.append(i+filter_imgs_num)
                
        extract_gt_pose = torch.from_numpy(np.stack(extract_gt_pose, 0))
        extract_imgs_ts = np.stack(extract_imgs_ts)
        return extract_gt_pose, extract_imgs_ts, extract_frames_id
