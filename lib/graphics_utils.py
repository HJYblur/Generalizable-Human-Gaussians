#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, K, h, w):
    znear = float(znear)
    zfar = float(zfar)
    near_fx = znear / float(K[0, 0])
    near_fy = znear / float(K[1, 1])
    left = - (float(w) - float(K[0, 2])) * near_fx
    right = float(K[0, 2]) * near_fx
    bottom = (float(K[1, 2]) - float(h)) * near_fy
    top = float(K[1, 2]) * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = float(2.0 * znear / (right - left))
    P[1, 1] = float(2.0 * znear / (top - bottom))
    P[0, 2] = float((right + left) / (right - left))
    P[1, 2] = float((top + bottom) / (top - bottom))
    P[3, 2] = z_sign
    P[2, 2] = float(z_sign * zfar / (zfar - znear))
    P[2, 3] = float(-(zfar * znear) / (zfar - znear))
    return P


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
