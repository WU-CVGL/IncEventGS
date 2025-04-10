"""
Convert TUM trajectory to KITTI format.

Usage:
python datasets/tum_to_kitti.py \
    --input=data/MBA-VO/archviz_sharp1/groundtruth_synced.txt \
    --output=data/MBA-VO/archviz_sharp1/traj.txt
"""
from __future__ import annotations

import argparse
import pypose as pp
import torch
from io_utils import load_tum_trajectory, write_kitti_trajectory

EXTRINSICS_C2B = pp.SE3(torch.tensor([0, 0, 0, -0.5, 0.5, -0.5, 0.5]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="TUM trajectory file")
    parser.add_argument("--output", type=str, help="KITTI trajectory file")
    args = parser.parse_args()

    timestamps, poses = load_tum_trajectory(args.input)
    poses = poses @ EXTRINSICS_C2B
    write_kitti_trajectory(args.output, poses)
