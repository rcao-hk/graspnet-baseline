import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np
import open3d as o3d
import scipy.io as scio
import cv2
import argparse
from PIL import Image
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='render', help='builder')
parser.add_argument('--is_table', type=bool, default=False,
                    help='whether to render table, if True render with fixed mesh and dynamic camera, else render with dynamic mesh and camera without extrinsic')
parser.add_argument('--meshes_path', type=str, default='/media/gpuadmin/rcao/dataset/graspnet/models/',
                    help='path to one example')
parser.add_argument('--meshes_name', type=str, default='nontextured.ply', help='obj_xx.ply')
parser.add_argument('--is_offscreen', type=bool, default=True, help='is offscreen render')
parser.add_argument('--camera', type=str, default='realsense', help='type of camera, kinect or realsense')
parser.add_argument('--depth_scale', type=int, default=1000, help='depth scale of camera')
parser.add_argument('--output_width', type=int, default=1280, help='640 for linemod and 1280 for graspnet')
parser.add_argument('--output_height', type=int, default=720, help='480 for linemod and 720 for graspnet')
parser.add_argument('--root_path', type=str, default='/media/gpuadmin/rcao/dataset/graspnet/scenes/',
                    help='path to the output folder')
parser.add_argument('--output_path', type=str, default='augment_test',
                    help='path to the output folder')
opt = parser.parse_args()

img_width = 720
img_length = 1280


def get_bbox(label, obj_t, intrinsic):
    # bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(label)
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    center_2d = [
        int((fx * obj_t[0] / obj_t[2]) + cx),  # Projected x
        int((fy * obj_t[1] / obj_t[2]) + cy)   # Projected y
    ]
    
    # Compute mask's maximum width and height
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1

    # Compute patch size based on max distance to the mask boundary
    max_dist_top = center_2d[1] - rmin
    max_dist_bottom = rmax - center_2d[1]
    max_dist_left = center_2d[0] - cmin
    max_dist_right = cmax - center_2d[0]

    patch_size = 2 * max(max_dist_top, max_dist_bottom, max_dist_left, max_dist_right)

    # Compute bounding box centered on `center_2d`
    rmin = center_2d[1] - patch_size // 2
    rmax = center_2d[1] + patch_size // 2
    cmin = center_2d[0] - patch_size // 2
    cmax = center_2d[0] + patch_size // 2

    # # Adjust bounding box to center around `center_2d` while ensuring it covers the mask
    # cmin = center_2d[0] - (center_2d[0] - cmin)
    # cmax = center_2d[0] + (cmax - center_2d[0])
    # rmin = center_2d[1] - (center_2d[1] - rmin)
    # rmax = center_2d[1] + (rmax - center_2d[1])

    # # Zero padding if bbox goes out of bounds
    pad_top = max(0, -rmin)
    pad_bottom = max(0, rmax - img_width)
    pad_left = max(0, -cmin)
    pad_right = max(0, cmax - img_length)

    # Clip the bbox to fit within the image
    rmin = max(0, rmin)
    rmax = min(img_width, rmax)
    cmin = max(0, cmin)
    cmax = min(img_length, cmax)

    # return rmin, rmax, cmin, cmax, center_2d
    return (rmin, rmax, cmin, cmax), (pad_top, pad_bottom, pad_left, pad_right), center_2d


def axangle2mat(axis, angle, is_normalized=False):
    """
    Convert axis-angle representation to rotation matrix (NumPy).
    
    Parameters:
    - axis: np.ndarray, shape (B, 3), rotation axes.
    - angle: np.ndarray, shape (B,), rotation angles in radians.
    - is_normalized: bool, whether the axes are already normalized.

    Returns:
    - rot_matrix: np.ndarray, shape (B, 3, 3), rotation matrices.
    """
    if not is_normalized:
        norm_axis = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = axis / norm_axis

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    cos = np.cos(angle)
    sin = np.sin(angle)
    one_minus_cos = 1 - cos

    xs = x * sin
    ys = y * sin
    zs = z * sin
    xC = x * one_minus_cos
    yC = y * one_minus_cos
    zC = z * one_minus_cos
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot_matrix = np.stack(
        [
            x * xC + cos,
            xyC - zs,
            zxC + ys,
            xyC + zs,
            y * yC + cos,
            yzC - xs,
            zxC - ys,
            yzC + xs,
            z * zC + cos,
        ],
        axis=1,
    ).reshape(-1, 3, 3)  # Batch x 3 x 3
    return rot_matrix


def update_pose_with_ground_truth_translation(pose, rotation_angle):
    """
    Update the 6D pose with 2D rotation applied to the patch image, using ground truth pose translation.

    Parameters:
    - pose: np.ndarray, shape (4, 4), initial 6D pose in homogeneous coordinates.
    - rotation_angle: float, rotation angle in radians (2D image rotation).

    Returns:
    - updated_pose: np.ndarray, shape (4, 4), updated 6D pose in homogeneous coordinates.
    """
    # Extract rotation (R) and translation (t) from the 4x4 pose matrix
    R = pose[:3, :3]  # 3x3 rotation matrix
    t = pose[:3, 3]   # Translation vector (3,)

    # Calculate alpha_x and alpha_y using ground truth translation
    alpha_x = -np.arctan2(t[1], t[2])  # YZ plane
    alpha_y = np.arctan2(t[0], np.linalg.norm(t[1:3]))  # ZX plane

    # Generate compensation rotation matrices
    Rx = axangle2mat(np.array([[1.0, 0.0, 0.0]]), np.array([alpha_x]), is_normalized=True)[0]
    Ry = axangle2mat(np.array([[0.0, 1.0, 0.0]]), np.array([alpha_y]), is_normalized=True)[0]

    # Generate 2D rotation matrix (around Z-axis)
    Rz = axangle2mat(np.array([[0.0, 0.0, 1.0]]), np.array([rotation_angle]), is_normalized=True)[0]

    # Final rotation update
    R_new = Ry @ Rx @ Rz @ Rx.T @ Ry.T @ R

    # Translation remains unchanged
    t_new = t

    # Assemble the updated 4x4 pose matrix
    updated_pose = np.eye(4)
    updated_pose[:3, :3] = R_new
    updated_pose[:3, 3] = t_new

    return updated_pose


def rotate_image(image, rotation_angle, center=None, fill_color=(0, 0, 0)):
    """
    Rotate the patch image around its center.

    Parameters:
    - image: np.ndarray, the input image (H x W x C).
    - rotation_angle: float, rotation angle in degrees.
    - center: tuple, center of rotation (x, y). If None, use image center.
    - fill_color: tuple, fill color for areas outside the original image.

    Returns:
    - rotated_image: np.ndarray, the rotated image.
    """
    h, w = image.shape[:2]
    rotation_angle = np.rad2deg(rotation_angle)
    
    # Default center is the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill_color
    )

    return rotated_image


def image_render(meshes_list, cam_param, output_name='Data/test', width=1280, height=720, is_offscreen=False):
    if is_offscreen:
        vis = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
        vis.setup_camera(cam_param.intrinsic, cam_param.extrinsic)
        vis.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1]))
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        ctr = vis.get_view_control()

    if is_offscreen:
        material = o3d.visualization.rendering.MaterialRecord()
        for i_mesh in range(len(meshes_list)):
            vis.scene.add_geometry('model_' + str(i_mesh), meshes_list[i_mesh], material)
    else:
        for mesh in meshes_list:
            vis.add_geometry(mesh)

    rgb_name = output_name + '_rgb.png'
    if is_offscreen:
        rgb_image = vis.render_to_image()
        # rgb_image = (rgb_image * 255).astype(np.uint8)
        # cv2.imwrite(rgb_name, rgb_image)
        o3d.io.write_image(rgb_name, rgb_image)
    else:
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        ctr.set_constant_z_far(3000)  # important step
        vis.poll_events()
        vis.capture_screen_image(rgb_name, do_render=False)
    return np.asarray(rgb_image)


def depth_render(meshes_list, cam_param, output_name='Data/test',
                 depth_scale=1000, width=1280, height=720, is_offscreen=False):
    if is_offscreen:
        vis = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
        vis.setup_camera(cam_param.intrinsic, cam_param.extrinsic)
        vis.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1]))
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        ctr = vis.get_view_control()

    if is_offscreen:
        material = o3d.visualization.rendering.MaterialRecord()
        for i_mesh in range(len(meshes_list)):
            vis.scene.add_geometry('model_' + str(i_mesh), meshes_list[i_mesh], material)
    else:
        for mesh in meshes_list:
            vis.add_geometry(mesh)

    depth_name = output_name + '_depth.png'
    depth_name_offscreen = output_name + '_depth.tif'
    rgb_name = output_name + '_rgb.png'
    if is_offscreen:
        depth_image = vis.render_to_depth_image(
            z_in_view_space=True)  # Pixels range from 0 (near plane) to 1 (far plane);
        # different from commonly used depth map
        depth_image_np = np.asarray(depth_image)
        depth_image_np = depth_image_np * depth_scale
        depth_image_np = depth_image_np.astype(np.uint16)

        cv2.imwrite(depth_name, depth_image_np)
        # depth_image_pil = Image.fromarray(depth_image_np)
        # depth_image_pil.save(depth_name)

        rgb_image = vis.render_to_image()
        # rgb_image = (rgb_image * 255).astype(np.uint8)
        # cv2.imwrite(rgb_name, rgb_image)
        o3d.io.write_image(rgb_name, rgb_image)
    else:
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        ctr.set_constant_z_far(3000)  # important step
        vis.poll_events()
        vis.capture_depth_image(depth_name, do_render=False, depth_scale=depth_scale)
        vis.capture_screen_image(rgb_name, do_render=False)


def add_table_mesh(meshes_list, trans=np.array([None]), scale=1.):
    table_mesh = o3d.geometry.TriangleMesh()

    '''
    ^y   0___1
    |    |  /|
    |    | / |
    |    |/__|
    |    2   3
    ----->x
    '''
    vertices = np.array([[-scale, scale, 0.],
                         [scale, scale, 0.],
                         [-scale, -scale, 0.],
                         [scale, -scale, 0.]])
    vertex_colors = np.array([[0., 0.447, 0.451],
                              [0., 0.447, 0.451],
                              [0., 0.447, 0.451],
                              [0., 0.447, 0.451]])
    faces = np.array([[0, 2, 1],
                      [1, 2, 3]])

    table_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    table_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    table_mesh.triangles = o3d.utility.Vector3iVector(faces)

    if not trans.any() == None:
        table_mesh = table_mesh.transform(trans)

    meshes_list.append(table_mesh)
    return meshes_list


def load_mesh(mesh_path, is_print=False):
    if is_print:
        print('Loading %s'%mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh


def place_meshes_graspnet(meshes, poses, cam_pos=np.array([None])):
    meshes_list = []

    for i_mesh in range(len(meshes)):
        pose = poses[:, :, i_mesh]
        mesh = copy.deepcopy(meshes[i_mesh])

        homog_term = np.array([[0., 0., 0., 1.]])
        pose = np.concatenate((pose, homog_term), axis=0)
        mesh = mesh.transform(pose)

        if not cam_pos.any() == None:
            mesh = mesh.transform(cam_pos)

        meshes_list.append(mesh)

    return meshes_list


def get_type_camera_parameters(extrinsic_mat=None, camera=''):
    param = o3d.camera.PinholeCameraParameters()

    if extrinsic_mat == None:
        param.extrinsic = np.eye(4, dtype=np.float64)
    else:
        param.extrinsic = extrinsic_mat

    # param.intrinsic = o3d.camera.PinholeCameraIntrinsic()

    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    else:
        print("Unknow camera type")
        exit(0)

    return param


def get_patch_point_cloud(patch_depth, intrinsics, bbox, cam_scale):
    """
    Get the 3D point cloud from a patch depth map, dynamically creating xmap and ymap based on crop size.

    Parameters:
    - patch_depth: np.ndarray, shape (H, W), depth map of the patch (in meters).
    - intrinsics: np.ndarray, shape (3, 3), camera intrinsic matrix.
    - rmin, rmax, cmin, cmax: int, patch boundary in the full image.
    - choose: np.ndarray, indices of valid points in the patch.
    - cam_scale: float, depth scale factor.

    Returns:
    - point_cloud: np.ndarray, shape (N, 3), 3D point cloud.
    """
    rmin, rmax, cmin, cmax = bbox
    # Intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Get the size of the cropped patch
    patch_height = rmax - rmin
    patch_width = cmax - cmin

    # Dynamically generate xmap and ymap for the cropped patch
    xmap = np.tile(np.arange(cmin, cmax), (patch_height, 1))  # X coordinates for crop
    ymap = np.tile(np.arange(rmin, rmax).reshape(-1, 1), (1, patch_width))  # Y coordinates for crop

    # Masked depth and pixel coordinates
    depth_masked = patch_depth.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)

    # Scale depth
    pt2 = depth_masked / cam_scale

    # Compute 3D coordinates
    pt0 = (xmap_masked - cx) * pt2 / fx
    pt1 = (ymap_masked - cy) * pt2 / fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    return cloud


def get_K_crop(K, crop_xy):
    """
    Parameters:
        K: [3,3]
        crop_xy: [2]  left top of crop boxes
    """
    assert K.shape == (3, 3)
    assert crop_xy.shape == (2,)

    new_K = K.copy()
    new_K[[0, 1], 2] = K[[0, 1], 2] - crop_xy  # [b, 2]
    new_K[[0, 1]] = new_K[[0, 1]]
    return new_K


def project_rgb_point_cloud_to_image(points_xyz, points_rgb, K, patch_shape):
    """
    Project a 3D RGB point cloud onto a 2D image plane.

    Parameters:
    - point_cloud_rgb: np.ndarray, shape (N, 6), 3D points and RGB colors.
                       The first 3 columns are XYZ coordinates, and the last 3 are RGB values.
    - K: np.ndarray, shape (3, 3), camera intrinsic matrix.
    - img_width: int, width of the image.
    - img_height: int, height of the image.

    Returns:
    - projected_image: np.ndarray, shape (img_height, img_width, 3), projected RGB image.
    """
    patch_height, patch_width = patch_shape

    # Filter points with Z <= 0 (behind the camera)
    valid_mask = points_xyz[:, 2] > 0
    points_xyz = points_xyz[valid_mask]
    points_rgb = points_rgb[valid_mask]

    # Project 3D points to 2D image plane
    uvw = (K @ points_xyz.T).T  # Shape: (N_valid, 3)
    u = uvw[:, 0] / uvw[:, 2]  # Normalize by depth
    v = uvw[:, 1] / uvw[:, 2]

    # Round to nearest pixel and cast to integers
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Filter points outside the image bounds
    valid_mask = (u >= 0) & (u < patch_width) & (v >= 0) & (v < patch_height)
    u = u[valid_mask]
    v = v[valid_mask]
    points_rgb = points_rgb[valid_mask]

    # Initialize an empty image
    projected_image = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)

    # Assign RGB values to the image
    projected_image[v, u] = (points_rgb * 255).astype(np.uint8)

    return projected_image


from utils.data_utils import transform_point_cloud
def points_pose_augment(point_cloud, object_pose):
    R = object_pose[:3, :3]
    t = object_pose[:3, 3]
    aug_pose = object_pose.copy()
    point_cloud -= t
    # Flipping along the YZ plane
    if np.random.random() > 0.5:
        flip_mat = np.array([[-1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
        point_cloud = transform_point_cloud(point_cloud, flip_mat, '3x3')
        R = np.dot(flip_mat, R).astype(np.float32)

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[1, 0, 0],
                        [0, c,-s],
                        [0, s, c]])
    point_cloud = transform_point_cloud(point_cloud, rot_mat, '3x3')
    R = np.dot(rot_mat, R).astype(np.float32)
    point_cloud += t
    aug_pose[:3, :3] = R
    return point_cloud, aug_pose

    
scene_id = 12
anno_id = 180
scene_path = os.path.join(opt.root_path, 'scene_{:04d}'.format(scene_id))

rgb_path = os.path.join(scene_path, '{}/rgb/{:04d}.png'.format(opt.camera, anno_id))
mask_path = os.path.join(scene_path, '{}/label/{:04d}.png'.format(opt.camera, anno_id))
depth_path = os.path.join(scene_path, '{}/depth/{:04d}.png'.format(opt.camera, anno_id))
meta_path = os.path.join(scene_path, '{}/meta/{:04d}.mat'.format(opt.camera, anno_id))
meta = scio.loadmat(meta_path)
cam_pos = np.load(os.path.join(scene_path, opt.camera, 'cam0_wrt_table.npy'))
cam_trans_poses = np.load(os.path.join(scene_path, opt.camera, 'camera_poses.npy'))
extrinsic_mat = np.linalg.inv(cam_pos).tolist()
obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
poses = meta['poses']
intrinsics = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
choose_idx = 0
output_name = os.path.join(opt.output_path, '{}_{}'.format(anno_id, choose_idx))

color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open(depth_path))
seg = np.array(Image.open(mask_path))

depth_mask = (depth > 0)
seg_masked = seg * depth_mask
inst_mask = seg_masked == obj_idxs[choose_idx]
object_pose = poses[:, :, choose_idx]

bbox, pad_bbox, center = get_bbox(inst_mask, object_pose[:3, 3], intrinsics)
(rmin, rmax, cmin, cmax) = bbox
(pad_top, pad_bottom, pad_left, pad_right) = pad_bbox

patch_color = color[rmin:rmax, cmin:cmax, :]
patch_depth = depth[rmin:rmax, cmin:cmax]
patch_mask = inst_mask[rmin:rmax, cmin:cmax]

patch_color[~patch_mask, :] = 0
patch_depth[~patch_mask] = 0

patch_color = np.pad(patch_color, 
    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Pad for H, W, and no padding for C
    mode='constant', constant_values=0)
patch_depth = np.pad(patch_depth, 
    ((pad_top, pad_bottom), (pad_left, pad_right)), 
    mode='constant', constant_values=0)
patch_mask = np.pad(patch_mask, 
    ((pad_top, pad_bottom), (pad_left, pad_right)), 
    mode='constant', constant_values=0)

rmin -= pad_top
rmax += pad_bottom
cmin -= pad_left
cmax += pad_right
bbox = (rmin, rmax, cmin, cmax)

inst_cloud = get_patch_point_cloud(patch_depth, intrinsics, bbox, factor_depth)
inst_color = patch_color.reshape(-1, 3)
 
pc_obj = o3d.geometry.PointCloud()
pc_obj.points = o3d.utility.Vector3dVector(inst_cloud)
pc_obj.colors = o3d.utility.Vector3dVector(inst_color[:, ::-1])
o3d.io.write_point_cloud('{}.ply'.format(scene_id), pc_obj)

inst_cloud, object_pose = points_pose_augment(inst_cloud, object_pose)

crop_k = get_K_crop(intrinsics, np.array([cmin, rmin]))
proj_patch_color = project_rgb_point_cloud_to_image(inst_cloud, inst_color, crop_k, (rmax - rmin, cmax - cmin))

# print(proj_patch_color.shape)
cv2.imwrite('patch_color.png', (patch_color * 255).astype(np.uint8))
cv2.imwrite('proj_patch_color.png', proj_patch_color)

# import torch
# device = torch.device("cuda:0")
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVOrthographicCameras, 
#     PointsRasterizationSettings,
#     PointsRenderer,
#     PulsarPointsRenderer,
#     PointsRasterizer,
#     AlphaCompositor,
#     NormWeightedCompositor
# )

# import matplotlib.pyplot as plt
# from pytorch3d.structures import Pointclouds
# verts = torch.Tensor(inst_cloud).to(device)
        
# rgb = torch.Tensor(inst_color).to(device)

# point_cloud = Pointclouds(points=[verts], features=[rgb])

# # Initialize a camera.
# # R, T = look_at_view_transform(20, 10, 0)
# R = torch.eye(3).view(1, 3, 3).to(device)
# T = torch.tensor([[0.0, 0.0, 0.0]]).to(device)

# cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

# # Define the settings for rasterization and shading. Here we set the output image to be of size
# # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
# raster_settings = PointsRasterizationSettings(
#     image_size=512, 
#     radius = 0.005,
#     points_per_pixel = 5
# )

# # Create a points renderer by compositing points using an alpha compositor (nearer points
# # are weighted more heavily). See [1] for an explanation.
# rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
# renderer = PointsRenderer(
#     rasterizer=rasterizer,
#     compositor=AlphaCompositor()
# )
# images = renderer(point_cloud)
# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.savefig('test.png')