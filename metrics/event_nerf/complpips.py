#!/usr/bin/env python3
#
# Compute and report the average LPIPS of two sets of images
#
import os
from os import path
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys
import torch

import lpips
loss_fn = lpips.LPIPS(net='alex')

inp_dir = sys.argv[1]  # prediction directory
gt_dir = sys.argv[2]  # ground-truth directory

def is_image(x):
    return path.splitext(x)[-1][1:].lower() in {'png', 'jpg'}

inp_files = [path.join(inp_dir, x) for x in sorted(os.listdir(inp_dir)) if is_image(x)]
gt_files = [path.join(gt_dir, x) for x in sorted(os.listdir(gt_dir)) if is_image(x)]
assert len(inp_files)==len(gt_files), f'number of images should match in both directories: {len(inp_files)} vs {len(gt_files)}'
assert len(inp_files)>0, 'no images found'

def preprocess_img(a):
    '''preprocess image from [0, 255] to [0, 1] and the correct channel order for LPIPS'''
    a = a / 255.
    # normalize to [-1, 1]
    a = a * 2 - 1
    # swap channels
    a = np.transpose(a, (2, 0, 1))[None, ...]
    a = torch.from_numpy(a).float()
    # print(a.shape)
    return a

res = []
for pred_fn, gt_fn in tqdm(list(zip(inp_files, gt_files))):
    pred = np.array(Image.open(pred_fn))
    pred = preprocess_img(pred)
    gt = np.array(Image.open(gt_fn))
    gt = preprocess_img(gt)
    val = loss_fn(pred, gt).item()
    res.append(val)

print(np.mean(res))

