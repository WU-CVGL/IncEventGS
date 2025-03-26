"""
SE(3) B-spline trajectory library

Created by lzzhao on 2023.09.19
"""
from __future__ import annotations

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor

_EPS = 1e-6

def se3_to_SE3(wu):
    Rt = pp.se3(wu).Exp().matrix()[:3,:4]
    return Rt

def se3_to_SE3_m44(wu):
    Rt = pp.se3(wu).Exp().matrix()
    return Rt

def SE3_to_se3(Rt):
    return pp.mat2SE3(Rt).Log()