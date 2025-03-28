from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.core.transformations import transform_points
from torchgeometry.core.transformations import boxminus_transformation
from torchgeometry.core.conversions import convert_points_to_homogeneous
from torchgeometry.core.pinhole import PinholeCamera, PinholeCamerasList
from torchgeometry.core.pinhole import normalize_pixel_coordinates
from torchgeometry.core.pinhole import cam2pixel, pixel2cam
from torchgeometry.utils import create_meshgrid


__all__ = [
    "depth_warp",
    "DepthWarper",
]

class DepthWarper(nn.Module):
    r"""Warps a patch by depth.

    .. math::
        P_{src}^{\{dst\}} = K_{dst} * T_{src}^{\{dst\}}

        I_{src} = \\omega(I_{dst}, P_{src}^{\{dst\}}, D_{src})

    Args:
        pinholes_dst (PinholeCamera): the pinhole models for the destination
          frame.
        height (int): the height of the image to warp.
        width (int): the width of the image to warp.
        mode (Optional[str]): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (Optional[str]): padding mode for outside grid values
           'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    """

    def __init__(self,
                 pinhole_dst: PinholeCamera,
                 height: int, width: int,
                 mode: Optional[str] = 'bilinear',
                 padding_mode: Optional[str] = 'zeros'):
        super(DepthWarper, self).__init__()
        # constructor members
        self.width: int = width
        self.height: int = height
        self.mode: Optional[str] = mode
        self.padding_mode: Optional[str] = padding_mode
        self.eps = 1e-6

        # state members
        self._pinhole_dst: PinholeCamera = pinhole_dst
        self._pinhole_src: Union[None, PinholeCamera] = None
        self._dst_proj_src: Union[None, torch.Tensor] = None

        self.grid: torch.Tensor = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height: int, width: int) -> torch.Tensor:
        grid: torch.Tensor = create_meshgrid(
            height, width, normalized_coordinates=False)  # 1xHxWx2
        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix(
            self, pinhole_src: PinholeCamera) -> 'DepthWarper':
        r"""Computes the projection matrix from the source to destinaion frame.
        """
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError("Member self._pinhole_dst expected to be of class "
                            "PinholeCamera. Got {}"
                            .format(type(self._pinhole_dst)))
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError("Argument pinhole_src expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_src)))
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src: torch.Tensor = boxminus_transformation(
            self._pinhole_dst.extrinsics, pinhole_src.extrinsics)

        # compute the projection matrix between the non reference cameras and
        # the reference.
        dst_proj_src: torch.Tensor = torch.matmul(
            self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self


    def _compute_projection(self, x, y, invd):
        point = torch.FloatTensor([[[x], [y], [1.0], [invd]]])
        flow = torch.matmul(
            self._dst_proj_src, point.to(self._dst_proj_src.device))
        z = 1. / flow[:, 2]
        x = (flow[:, 0] * z)
        y = (flow[:, 1] * z)
        return torch.cat([x, y], 1)

    def compute_subpixel_step(self) -> torch.Tensor:
        """This computes the required inverse depth step to achieve sub pixel
        accurate sampling of the depth cost volume, per camera.

        Szeliski, Richard, and Daniel Scharstein.
        "Symmetric sub-pixel stereo matching." European Conference on Computer
        Vision. Springer Berlin Heidelberg, 2002.
        """
        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2,
                                         1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2,
                                         1.0 + delta_d)
        dx = torch.norm((xy_p1 - xy_m1), 2, dim=-1) / 2.0
        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras
        return torch.min(0.5 / dxdd)


    def warp_grid(self, depth_src: torch.Tensor) -> torch.Tensor:
        """Computes a grid for warping a given the depth from the reference
        pinhole camera.

        The function `compute_projection_matrix` has to be called beforehand in
        order to have precomputed the relative projection matrices encoding the
        relative pose and the intrinsics between the reference and a non
        reference camera.
        """
        # TODO: add type and value checkings
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError("Input depth_src has to be in the shape of "
                             "Bx1xHxW. Got {}".format(depth_src.shape))

        # unpack depth attributes
        batch_size, _, height, width = depth_src.shape
        device: torch.device = depth_src.device
        dtype: torch.dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords: torch.Tensor = self.grid.to(device).to(dtype).expand(
            batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src: torch.Tensor = pixel2cam(
            depth_src,
            self._pinhole_src.intrinsics_inverse().to(dtype),
            pixel_coords)  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src: torch.Tensor = cam2pixel(
            cam_coords_src, self._dst_proj_src.to(dtype))  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm: torch.Tensor = normalize_pixel_coordinates(
            pixel_coords_src, self.height, self.width)
        return pixel_coords_src_norm


    def forward(
            self,
            depth_src: torch.Tensor,
            patch_dst: torch.Tensor) -> torch.Tensor:
        """Warps a tensor from destination frame to reference given the depth
        in the reference frame.

        Args:
            depth_src (torch.Tensor): the depth in the reference frame. The
              tensor must have a shape :math:`(B, 1, H, W)`.
            patch_dst (torch.Tensor): the patch in the destination frame. The
              tensor must have a shape :math:`(B, C, H, W)`.

        Return:
            torch.Tensor: the warped patch from destination frame to reference.

        Shape:
            - Output: :math:`(N, C, H, W)` where C = number of channels.

        Example:
            >>> # pinholes camera models
            >>> pinhole_dst = tgm.PinholeCamera(...)
            >>> pinhole_src = tgm.PinholeCamera(...)
            >>> # create the depth warper, compute the projection matrix
            >>> warper = tgm.DepthWarper(pinhole_dst, height, width)
            >>> warper.compute_projection_matrix(pinhole_src)
            >>> # warp the destionation frame to reference by depth
            >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
            >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
            >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
        """
        return F.grid_sample(patch_dst, self.warp_grid(depth_src),
                             mode=self.mode, padding_mode=self.padding_mode)



# functional api
def depth_warp(pinhole_dst: PinholeCamera,
               pinhole_src: PinholeCamera,
               depth_src: torch.Tensor,
               patch_dst: torch.Tensor,
               height: int, width: int):
    r"""Function that warps a tensor from destination frame to reference
    given the depth in the reference frame.

    See :class:`~torchgeometry.DepthWarper` for details.

    Example:
        >>> # pinholes camera models
        >>> pinhole_dst = tgm.PinholeCamera(...)
        >>> pinhole_src = tgm.PinholeCamera(...)
        >>> # warp the destionation frame to reference by depth
        >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
        >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
        >>> image_src = tgm.depth_warp(pinhole_dst, pinhole_src,
        >>>     depth_src, image_dst, height, width)  # NxCxHxW
    """
    warper = DepthWarper(pinhole_dst, height, width)
    warper.compute_projection_matrix(pinhole_src)
    return warper(depth_src, patch_dst)

### Test
# import cv2
# import numpy as np
# import torchgeometry as tgm

# height = 480
# width = 768
# fx = 300
# fy = 300
# cx = 384
# cy = 240

# intrinsics = torch.zeros(1, 4, 4)
# intrinsics[..., 0, 0] += fx
# intrinsics[..., 1, 1] += fy
# intrinsics[..., 0, 2] += cx
# intrinsics[..., 1, 2] += cy
# intrinsics[..., 2, 2] += 1.0
# intrinsics[..., 3, 3] += 1.0

# T_src2wld_Rt = torch.eye(4).repeat(1, 1, 1)
# T_dst2wld_Rt = torch.eye(4).repeat(1, 1, 1)
# T_dst2wld_Rt[..., 0, 3] = 2

# # create image hegith and width
# height_tmp = torch.zeros(1)
# height_tmp[..., 0] += height
# width_tmp = torch.zeros(1)
# width_tmp[..., 0] += width

# # creat pinhole cameras 
# pinhole_src = tgm.PinholeCamera(intrinsics, T_src2wld_Rt, height_tmp, width_tmp)
# pinhole_dst = tgm.PinholeCamera(intrinsics, T_dst2wld_Rt, height_tmp, width_tmp)

# #
# depth_src = (torch.ones(1, 1, height, width) * 10).double()  # Nx1xHxW
# image_dst = cv2.imread('/media/peidong/Ventoy/replica_event_baseline-1000hz/room0/Gray/002.png')
# image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)
# image_dst = image_dst.astype(np.double) / 255.
# image_dst = torch.from_numpy(image_dst).unsqueeze(0).permute(0, 3, 1, 2)

# # 
# image_src = depth_warp(pinhole_dst, pinhole_src, depth_src, image_dst, height, width)  # NxCxHxW
# mask_src = (image_src > 0).double()
# print(mask_src)

# # display
# image_dst = image_dst.permute(0, 2, 3, 1)
# image_src = image_src.permute(0, 2, 3, 1)
# mask_src = mask_src.permute(0, 2, 3, 1)
# print(image_src.size())
# image = torch.cat([image_src, image_dst, mask_src], dim=2).squeeze(0).numpy()

# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()


