""" Tools for loss computation.
    Author: chenxi-wang
"""

import torch
import numpy as np

GRASP_MAX_WIDTH = 0.1
GRASP_MAX_TOLERANCE = 0.05
GRASPNESS_THRESHOLD = 0.1
THRESH_GOOD = 0.7
THRESH_BAD = 0.1

NUM_VIEW = 300
NUM_ANGLE = 12
NUM_DEPTH = 4
M_POINT = 1024

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [torch.FloatTensor, (N,3)]
                points in original coordinates
            transform: [torch.FloatTensor, (3,3)/(3,4)/(4,4)]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [torch.FloatTensor, (N,3)]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = torch.matmul(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
        cloud_ = torch.cat([cloud, ones], dim=1)
        cloud_transformed = torch.matmul(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def generate_grasp_views(N=300, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch
                
        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:,1], axis_x[:,0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y,1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Author: Charles R. Qi
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


# def batch_key_points(centers, Rs, widths):
#     center_shape = centers.size()
#     if len(centers.size()) == 3:
#         Bs, num_samples, _ = center_shape
#         centers = centers.view((-1, 3))
#         Rs = Rs.view((-1, 3, 3))
#         widths = widths.view(-1)
#     depth_base = 0.02
#     height = 0.02
#     tail_length = 0.04
#     R_sym = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], device=centers.device)
#     key_points = torch.zeros((centers.size(0), 5, 3), device=centers.device)

#     key_points[:, :, 0] -= height / 2
#     key_points[:, :2, 1] += widths.unsqueeze(-1) / 2
#     key_points[:, 2:4, 1] -= widths.unsqueeze(-1) / 2
#     key_points[:, 1:3, 0] += depth_base / 2
#     key_points[:, 0, 0] -= depth_base / 2
#     key_points[:, 3, 0] -= depth_base / 2
#     key_points[:, 4, 0] -= (depth_base / 2 + tail_length)
#     key_points_sym = key_points.detach()

#     key_points = torch.matmul(Rs, key_points.transpose(1, 2)).transpose(1, 2)
#     key_points_sym = torch.matmul(torch.matmul(Rs, R_sym), key_points_sym.transpose(1, 2)).transpose(1, 2)

#     key_points += centers.unsqueeze(1)
#     key_points_sym += centers.unsqueeze(1)
    
#     if len(center_shape) == 3:
#         key_points = key_points.view((Bs, num_samples, 5, 3))
#         key_points_sym = key_points_sym.view((Bs, num_samples, 5, 3))
#     return key_points, key_points_sym


def batch_get_key_points(centers, Rs, widths, depths):
    '''
        Input:
            centers: torch.Tensor of shape (-1, 3) for the translation.
            Rs: torch.Tensor of shape (-1, 3, 3) for the rotation matrix.
            widths: torch.Tensor of shape (-1) for the grasp width.
            depths: torch.Tensor of shape (-1) for the grasp depth.

        Output:
            key_points: torch.Tensor of shape (-1, 4, 3) for the key points of the grasp.
            key_points_sym: torch.Tensor of shape (-1, 4, 3) for the symmetric key points of the grasp.
    '''
    center_shape = centers.size()
    if len(center_shape) == 3:
        Bs, num_samples, _ = center_shape
        centers = centers.view((-1, 3))
        Rs = Rs.view((-1, 3, 3))
        widths = widths.view(-1)
        depths = depths.view(-1)

    height = 0.02

    R_sym = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], device=centers.device)
    key_points = torch.zeros((centers.size(0), 4, 3), device=centers.device)

    # Adjustments based on depth_base, widths, and height
    key_points[:, :, 0] -= depths.unsqueeze(1)  # Adjust x coordinates based on depth
    key_points[:, 1:, 1] -= widths.unsqueeze(1) / 2  # Adjust y coordinates based on width
    key_points[:, 2, 2] += height / 2  # Adjust the z coordinate for one of the height key points
    key_points[:, 3, 2] -= height / 2  # Adjust the z coordinate for the other height key point

    key_points_sym = key_points.detach()

    key_points = torch.matmul(Rs, key_points.transpose(1, 2)).transpose(1, 2)
    key_points_sym = torch.matmul(torch.matmul(Rs, R_sym), key_points_sym.transpose(1, 2)).transpose(1, 2)

    key_points += centers.unsqueeze(1)
    key_points_sym += centers.unsqueeze(1)
    if len(center_shape) == 3:
        key_points = key_points.view((Bs, num_samples, 4, 3))
        key_points_sym = key_points_sym.view((Bs, num_samples, 4, 3))
    return key_points, key_points_sym