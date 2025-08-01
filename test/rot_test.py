import numpy as np
import torch
import os
import sys
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from loss_utils import generate_grasp_views
from loss_utils import batch_viewpoint_params_to_matrix as batch_viewpoint_params_to_matrix_torch

# Constants
NUM_ANGLE = 12
NUM_VIEW = 300

# Torch implementation
angles_torch = torch.tensor([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)])
views_torch = generate_grasp_views(NUM_VIEW)
angles_repeat_torch = angles_torch.tile((NUM_VIEW,))
views_repeat_torch = views_torch.repeat_interleave(NUM_ANGLE, dim=0)
grasp_rot_torch = batch_viewpoint_params_to_matrix_torch(
    -views_repeat_torch, angles_repeat_torch
)

# NumPy implementation
angles_numpy = np.array([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)], dtype=np.float32)
views_numpy = generate_views(NUM_VIEW)
angles_repeat_numpy = angles_numpy.reshape(1, NUM_ANGLE).repeat(NUM_VIEW, axis=0).reshape(-1)
views_repeat_numpy = views_numpy.repeat(NUM_ANGLE, axis=0)
grasp_rot_numpy = batch_viewpoint_params_to_matrix(
    -views_repeat_numpy, angles_repeat_numpy
)

# Reshape to match final Torch output
grasp_rot_numpy = grasp_rot_numpy.reshape(NUM_VIEW, NUM_ANGLE, -1)
grasp_rot_torch = grasp_rot_torch.reshape(NUM_VIEW, NUM_ANGLE, -1)
grasp_rot_torch_numpy = grasp_rot_torch.numpy()

# Comparison
print("Shape comparison:")
print(f"Torch: {grasp_rot_torch.shape}, NumPy: {grasp_rot_numpy.shape}")
print("\nElement-wise comparison:")
print(np.allclose(grasp_rot_torch_numpy, grasp_rot_numpy, atol=1e-3))