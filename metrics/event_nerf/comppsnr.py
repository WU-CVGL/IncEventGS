#!/usr/bin/env python3
#
# Regress the tone-mapping as a linear transform in linear color space against
# the ground-truth, report the PSNR pre- and post-correction, and save
# the corrected images to a folder.
#
import numpy as np
from PIL import Image
import sys
import os
from os import path
from tqdm import tqdm


inp_dir = sys.argv[1]  # prediction directory
gt_dir = sys.argv[2]  # ground-truth directory

out_dir = inp_dir+'_corr'  # corrected output directory
os.makedirs(out_dir, exist_ok=True)

def is_image(x):
    return path.splitext(x)[-1][1:].lower() in {'png', 'jpg'}

inp_files = [path.join(inp_dir, x) for x in sorted(os.listdir(inp_dir)) if is_image(x)]
gt_files = [path.join(gt_dir, x) for x in sorted(os.listdir(gt_dir)) if is_image(x)]
assert len(inp_files)==len(gt_files), f'number of images should match in both directories: {len(inp_files)} vs {len(gt_files)}'
assert len(inp_files)>0, 'no images found'

res1 = 0.0
res2 = 0.0
count = 0

def img_preprocess(a):
    '''convert gamma to linear color'''
    a = a / 255.
    a = a ** 2.2
    a = np.log(a+0.1)
    return a

def img_unprocess(a):
    '''convert linear to gamma color'''
    a = np.maximum(0, np.exp(a)-0.1)
    a = a ** (1/2.2)
    return a

# load all files
gt = [img_preprocess(np.array(Image.open(x))) for x in tqdm(gt_files)]
pred = [img_preprocess(np.array(Image.open(x))) for x in tqdm(inp_files)]

gt = np.concatenate(gt)
pred = np.concatenate(pred)

assert gt.shape[-1] == 3
assert pred.shape[-1] == 3

X = gt.reshape((-1, 3))
Y = pred.reshape((-1, 3))

# regress A - slope, B - offset for the correction
A = (np.mean(X*Y, 0)-np.mean(X, 0)*np.mean(Y, 0))/(np.mean(Y*Y, 0)-np.mean(Y, 0)**2)
B = np.mean(X, 0)-A*np.mean(Y, 0)

# print the regressed values
print(A, B)
# print PSNR of the source uncorrected images in linear color
print(-10*np.log10(np.mean((X-Y)**2)))
# print PSNR of the source uncorrected images in gamma color
print(-10*np.log10(np.mean((img_unprocess(X)-img_unprocess(Y))**2)))

# apply the regressed correction
Y = Y * A + B
# print PSNR of the *corrected* images in linear color
print(-10*np.log10(np.mean((X-Y)**2)))
# print PSNR of the *corrected* images in gamma color (reported value)
print(-10*np.log10(np.mean((img_unprocess(X)-img_unprocess(Y))**2)))


# write corrected images to the output folder
for y_fn in tqdm(inp_files):
    y = np.array(Image.open(y_fn))
    y = img_preprocess(y)
    y = y * A + B
    y = img_unprocess(y)
    y = np.clip(y*255, 0, 255).astype(np.uint8)
    Image.fromarray(y).save(path.join(out_dir, path.basename(y_fn)))

