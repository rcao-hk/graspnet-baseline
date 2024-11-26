import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import numpy as np
import open3d as o3d
import scipy.io as scio
import cv2
import argparse
from PIL import Image
import copy
from tqdm import tqdm

from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points, sample_points, points_denoise
                            
img_width = 720
img_length = 1280


def get_bbox(label, obj_t, intrinsic):
    # bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(label)
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    center_2d = [
        round((fx * obj_t[0] / obj_t[2]) + cx),  # Projected x
        round((fy * obj_t[1] / obj_t[2]) + cy)   # Projected y
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
    # R_new = Ry @ Rx @ Rz @ Rx.T @ Ry.T @ R
    R_new = Rz @ R

    # Translation remains unchanged
    t_new = t

    # Assemble the updated 4x4 pose matrix
    updated_pose = np.eye(4)
    updated_pose[:3, :3] = R_new
    updated_pose[:3, 3] = t_new

    return updated_pose


def rotate_image(color, depth, mask, rotation_angle, center=None, fill_color=(0, 0, 0)):
    """
    Rotate the patch image around its center.

    Parameters:
    - image: np.ndarray, the input image (H x W x C).
    - rotation_angle: float, rotation angle in radius.
    - center: tuple, center of rotation (x, y). If None, use image center.
    - fill_color: tuple, fill color for areas outside the original image.

    Returns:
    - rotated_image: np.ndarray, the rotated image.
    """
    h, w = color.shape[:2]
    rotation_angle = np.rad2deg(rotation_angle)
    
    # Default center is the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)

    # Perform the rotation
    rotated_color = cv2.warpAffine(
        color, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill_color
    )

    rotated_depth = cv2.warpAffine(
        depth, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )

    rotated_mask = cv2.warpAffine(
        mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )
    return rotated_color, rotated_depth, rotated_mask



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


def flip_image(rgb, depth, mask, flip_code):
    """
    Flip RGB, depth, and mask images.

    Parameters:
    - rgb_image: np.ndarray, shape (H, W, 3), RGB image.
    - depth_image: np.ndarray, shape (H, W), depth image.
    - mask_image: np.ndarray, shape (H, W), mask image.
    - flip_code: int, flip direction.
        0: Flip vertically.
        1: Flip horizontally.
       -1: Flip both vertically and horizontally.

    Returns:
    - flipped_rgb: np.ndarray, flipped RGB image.
    - flipped_depth: np.ndarray, flipped depth image.
    - flipped_mask: np.ndarray, flipped mask image.
    """
    # Flip images using OpenCV's flip function
    flipped_rgb = cv2.flip(rgb, flip_code)
    flipped_depth = cv2.flip(depth, flip_code)
    flipped_mask = cv2.flip(mask, flip_code)

    return flipped_rgb, flipped_depth, flipped_mask


def load_grasp_labels(root):
    obj_names = list(range(88))
    # obj_names = [0, 2, 5, 14, 15, 20, 21, 22, 41, 43, 44, 46, 48, 52, 60, 62, 66, 70]
    # obj_names = [0, 2, 5, 20, 26, 37, 38, 51, 66]
    # obj_names = [ 8, 20, 26, 30, 41, 46, 56, 57, 60, 63, 66]
    obj_names = [ 0, 9, 17, 51, 58, 61, 69, 70,]
    valid_obj_idxs = []
    grasp_labels = {}
    for obj_idx in tqdm(obj_names, desc='Loading grasping labels...'):
        # if i == 18: continue
        valid_obj_idxs.append(obj_idx+1) #here align with label png
        # tolerance = np.load(os.path.join(root, 'tolerance', '{}_tolerance.npy'.format(str(obj_idx).zfill(3))))
        # label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        # grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
        #                         label['scores'].astype(np.float32), tolerance)
        # label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(obj_idx).zfill(3))))
        # grasp_labels[obj_idx+1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
        #                           label['scores'].astype(np.float32), tolerance)
        # label = h5py.File(os.path.join(root, 'grasp_label_simplified_hdf5', '{}_labels.hdf5'.format(str(obj_idx).zfill(3))), "r")
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_idx).zfill(3))))
        grasp_labels[obj_idx+1] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))
    return valid_obj_idxs, grasp_labels


scene_id = 14
anno_id = 135
root = '/media/gpuadmin/rcao/dataset/graspnet'
valid_obj_idxs, grasp_labels = load_grasp_labels(root)

scene_path = os.path.join(root, 'scenes', 'scene_{:04d}'.format(scene_id))
camera = 'realsense'
rgb_path = os.path.join(scene_path, '{}/rgb/{:04d}.png'.format(camera, anno_id))
mask_path = os.path.join(scene_path, '{}/label/{:04d}.png'.format(camera, anno_id))
depth_path = os.path.join(scene_path, '{}/depth/{:04d}.png'.format(camera, anno_id))
meta_path = os.path.join(scene_path, '{}/meta/{:04d}.mat'.format(camera, anno_id))
meta = scio.loadmat(meta_path)
cam_pos = np.load(os.path.join(scene_path, camera, 'cam0_wrt_table.npy'))
cam_trans_poses = np.load(os.path.join(scene_path, camera, 'camera_poses.npy'))

obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
poses = meta['poses']
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
choose_idx = 6
output_name = os.path.join('augment_test', '{}_{}'.format(anno_id, choose_idx))

color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
# color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open(depth_path))
seg = np.array(Image.open(mask_path))

depth_mask = (depth > 0)
seg_masked_org = seg * depth_mask
# inst_mask = seg_masked == obj_idxs[choose_idx]
inst_mask = seg_masked_org == obj_idxs[choose_idx]
inst_mask = inst_mask.astype(np.uint8)

camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

# generate cloud
cloud = create_point_cloud_from_depth_image(depth, camera, organized=False)

scene = o3d.geometry.PointCloud()
scene.points = o3d.utility.Vector3dVector(cloud.astype(np.float32))
scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float32))

color, depth, seg = flip_image(color, depth, seg, 1)
flip_mat = np.array([[-1, 0, 0],
                    [ 0, 1, 0],
                    [ 0, 0, 1]])
filp_cloud = transform_point_cloud(cloud, flip_mat, '3x3')
flip_pose = np.eye(4)
flip_pose[:3, :3] = flip_mat

# flip_mat = np.eye(4)

degree = 40
angle = np.deg2rad(degree)
c, s = np.cos(angle), np.sin(angle)

rot_pose = np.eye(4)
rot_pose[:3, :3] = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])
trans_pose = np.eye(4)
trans_pose[:3, :3] = rot_pose[:3, :3] @ flip_mat[:3, :3]

color, depth, seg = rotate_image(color, depth, seg, -angle)

trans_cloud = create_point_cloud_from_depth_image(depth, camera, organized=False)

# scene = o3d.geometry.PointCloud()
# scene.points = o3d.utility.Vector3dVector(cloud.astype(np.float32))
# scene.paint_uniform_color([1.0, 0.0, 0.0])
# rot_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float32))
# scene.transform(trans_pose)
# scene.transform(rot_pose)

# inst_pose = np.eye(4)
# inst_pose[:3, :] = poses[:, :, choose_idx]
# inst_pose = trans_pose @ inst_pose
# points, offsets, scores = grasp_labels[obj_idxs[choose_idx]]
# inst_point = transform_point_cloud(points, inst_pose, '4x4')

inst_pose = poses[:, :, choose_idx]
points, offsets, scores = grasp_labels[obj_idxs[choose_idx]]
inst_point = transform_point_cloud(points, inst_pose, '3x4')

inst_vis = o3d.geometry.PointCloud()
inst_vis.points = o3d.utility.Vector3dVector(inst_point.astype(np.float32))
inst_vis.paint_uniform_color([0.0, 1.0, 0.0])
# inst_vis.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float32))

proj_scene = o3d.geometry.PointCloud()
proj_scene.points = o3d.utility.Vector3dVector(trans_cloud.astype(np.float32))
proj_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float32))

axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
axis_pcd = axis_mesh.sample_points_uniformly(number_of_points=1000)

scene_vis = inst_vis + scene + axis_pcd
# scene_vis = inst_vis + proj_scene + axis_pcd
o3d.io.write_point_cloud('{}_{}_scene_gt.ply'.format(scene_id, anno_id), scene_vis)