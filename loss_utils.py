import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def compute_white_balance_loss(x, alpha=20, epsilon=1e-6):
    """
    Custom loss function.
    x: Input value
    alpha: Parameter that controls the rate of gradient change
    epsilon: Smoothing parameter
    """
    smooth_abs = torch.sqrt((x - 0.5) ** 2 + epsilon)
    return torch.sigmoid(alpha * (smooth_abs - 0.25))


def compute_ssim_loss(X, Y, data_range=1.0, size_average=True, channel=3): # channel=1 for grayscale images
    if X.shape[-1]==3:
        X=X.permute(2, 0, 1).unsqueeze(0) # (H,W,3) to (1,3,H,W)
        Y=Y.permute(2, 0, 1).unsqueeze(0) # (H,W,3) to (1,3,H,W)
    else:
        X = X.unsqueeze(0).unsqueeze(0) # (H,W) to (1,1,H,W)
        Y = Y.unsqueeze(0).unsqueeze(0) # (H,W) to (1,1,H,W)
    
    ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)
    ssim_loss = 1 - ssim_module(X, Y)
    return ssim_loss