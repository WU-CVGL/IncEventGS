import torch
from spline.spline import se3_to_SE3


def reblur(results, deblur_num):
    batch_size = results.shape[0] // deblur_num
    results_sharp = results.reshape(deblur_num, batch_size, -1)
    results_blur = torch.mean(results_sharp, 0)
    return results_blur


def mid(results, deblur_num):
    batch_size = results.shape[0] // deblur_num

    j = (deblur_num // 2)
    idx = torch.arange(j*batch_size, (j+1)*batch_size)
    results_mid = results[idx]

    return results_mid


def mid_pose(poses, spline_num):
    frame_num = poses.shape[0] // spline_num
    idx = torch.arange(frame_num) * spline_num + spline_num // 2
    pose_mid = poses[idx]

    return pose_mid


def mid_start_end_poses(poses, spline_num):
    frame_num = poses.shape[0] // spline_num
    idx_mid = torch.arange(frame_num) * spline_num + spline_num // 2
    idx_start = torch.arange(frame_num) * spline_num
    idx_end = torch.arange(frame_num) * spline_num + (spline_num - 1)

    return poses[idx_mid].detach(), poses[idx_start].detach(), poses[idx_end].detach()


def noise_T():
    noise = torch.rand(1, 6) * 0.001
    noise_I = se3_to_SE3(noise).squeeze(0)
    bottom = torch.tensor([0., 0., 0., 1.]).reshape(1, 4)
    pose = torch.cat([noise_I, bottom], 0)

    return pose
