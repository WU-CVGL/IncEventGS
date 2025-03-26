"""
Input / output utils.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor


def load_tum_trajectory(
        filename: Path
) -> Tuple[
    Float[Tensor, "num_poses"],
    Float[LieTensor, "num_poses 7"]
]:
    """load tum trajectory from file"""
    with open(filename, 'r', encoding="UTF-8") as f:
        lines = f.read().splitlines()
    if lines[0].startswith('#'):
        lines.pop(0)
    lines = [line.split() for line in lines]
    lines = [[float(val) for val in line] for line in lines]
    timestamps = []
    poses = []
    for line in lines:
        timestamps.append(torch.tensor(line[0]))
        poses.append(torch.tensor(line[1:]))
    timestamps = torch.stack(timestamps)
    poses = pp.SE3(torch.stack(poses))
    return timestamps, poses


def write_tum_trajectory(
        filename: Path,
        timestamps: Float[Tensor, "num_poses"],
        poses: Float[LieTensor, "num_poses 7"] | Float[Tensor, "num_poses 7"]
):
    """write tum trajectory to file"""
    with open(filename, 'w', encoding="UTF-8") as f:
        if pp.is_lietensor(poses):
            poses = poses.tensor()
        for timestamp, pose in zip(timestamps, poses):
            f.write(f'{timestamp.item()} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n')


def write_kitti_trajectory(
        filename: Path,
        poses: Float[LieTensor, "num_poses 7"] | Float[Tensor, "num_poses 7"]
):
    """write kitti trajectory to file"""
    with open(filename, 'w', encoding="UTF-8") as f:
        if pp.is_lietensor(poses):
            poses = poses.matrix()  # 4x4 matrix
        for pose in poses:
            f.write(f"{' '.join([str(p.item()) for p in pose.flatten()])}\n")
