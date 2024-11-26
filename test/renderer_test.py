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


def get_bbox_from_pose_and_mask(label, pose, intrinsics, img_shape):
    """
    Generate a patch bounding box based on 6D pose projection and mask width.

    Parameters:
    - label: numpy array, binary mask of the object (H, W).
    - pose: dict, containing 'R' (3x3 rotation matrix) and 't' (3x1 translation vector).
    - intrinsics: 3x3 numpy array, camera intrinsic matrix.
    - img_shape: tuple, (height, width) of the original image.

    Returns:
    - pad_params: numpy array, cropped patch with zero padding if necessary.
    - bbox: tuple, (rmin, rmax, cmin, cmax) of the bounding box.
    """
    img_height, img_width = img_shape

    # Project the 3D center to 2D image using camera intrinsics
    R, t = pose[:3, :3], pose[:3, 3]
    center_3d = t  # Assuming t is the 3D center of the object
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    center_2d = [
        round((fx * center_3d[0] / center_3d[2]) + cx),  # Projected x
        round((fy * center_3d[1] / center_3d[2]) + cy)   # Projected y
    ]

    # Compute mask's maximum width and height
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin_mask, rmax_mask = np.where(rows)[0][[0, -1]]
    cmin_mask, cmax_mask = np.where(cols)[0][[0, -1]]
    max_width = cmax_mask - cmin_mask
    max_height = rmax_mask - rmin_mask

    # Determine patch size (square, using max dimension)
    patch_size = max(max_width, max_height)

    # Compute bounding box centered on the projected 2D center
    rmin = center_2d[1] - patch_size // 2
    rmax = center_2d[1] + patch_size // 2
    cmin = center_2d[0] - patch_size // 2
    cmax = center_2d[0] + patch_size // 2

    # Zero padding if bbox goes out of bounds
    pad_top = max(0, -rmin)
    pad_bottom = max(0, rmax - img_height)
    pad_left = max(0, -cmin)
    pad_right = max(0, cmax - img_width)

    # Clip the bbox to fit within the image
    rmin_clipped = max(0, rmin)
    rmax_clipped = min(img_height, rmax)
    cmin_clipped = max(0, cmin)
    cmax_clipped = min(img_width, cmax)

    pad_params = (pad_top, pad_bottom, pad_left, pad_right)
    bbox = (rmin_clipped, rmax_clipped, cmin_clipped, cmax_clipped)
    return pad_params, bbox


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


scene_id = 0
anno_id = 0
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
choose_idx = 1
output_name = os.path.join(opt.output_path, '{}_{}'.format(anno_id, choose_idx))

color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open(depth_path))
seg = np.array(Image.open(mask_path))

depth_mask = (depth > 0)
seg_masked_org = seg * depth_mask
# inst_mask = seg_masked == obj_idxs[choose_idx]
inst_mask_org = seg_masked_org == obj_idxs[choose_idx]
object_pose = poses[:, :, choose_idx]
# rmin, rmax, cmin, cmax = get_bbox(inst_mask_org.astype(np.uint8))

pad_params, bbox = get_bbox_from_pose_and_mask(inst_mask_org.astype(np.uint8), object_pose, intrinsics, (img_width, img_length))
rmin, rmax, cmin, cmax = bbox
pad_top, pad_bottom, pad_left, pad_right = pad_params

img = color[rmin:rmax, cmin:cmax, :]
depth = depth[rmin:rmax, cmin:cmax]

# Apply zero padding
img = np.pad(
    img, 
    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Pad for H, W, and no padding for C
    mode='constant', constant_values=0
)
depth = np.pad(
    depth, 
    ((pad_top, pad_bottom), (pad_left, pad_right)), 
    mode='constant', constant_values=0
)


cam_param = get_type_camera_parameters(camera=opt.camera)

meshes_transed = []
meshes_labels_list = []
meshes = []

cam_param.intrinsic.set_intrinsics(img_length, img_width,
                                    intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2])
# cam_param.extrinsic = np.linalg.inv(np.matmul(cam_pos, cam_trans_poses[scene_id])).tolist()
num_meshes = poses.shape[2]

# for i_mesh in range(num_meshes):
#     idx_mesh = str(obj_idxs[i_mesh] - 1).zfill(3)
#     mesh_path = opt.meshes_path + idx_mesh + '/' + opt.meshes_name
#     mesh = load_mesh(mesh_path)
#     meshes.append(mesh)

idx_mesh = str(obj_idxs[choose_idx] - 1).zfill(3)
mesh_path = opt.meshes_path + idx_mesh + '/' + opt.meshes_name
mesh = load_mesh(mesh_path)
meshes.append(mesh)

render_poses = np.zeros((3, 4, 1))
render_poses[:, :, 0] = object_pose
meshes_transed = place_meshes_graspnet(meshes, render_poses)
rendered_img = image_render(meshes_transed, cam_param, output_name=output_name, width=img_length, height=img_width, is_offscreen=opt.is_offscreen)

rendered_img = rendered_img[rmin:rmax, cmin:cmax, :]
rendered_img = np.pad(
    rendered_img, 
    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
    mode='constant', constant_values=0
)

rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(opt.output_path, "{}_{}_raw.png".format(anno_id, choose_idx)), color*255.0)
cv2.imwrite(os.path.join(opt.output_path, "{}_{}_input.png".format(anno_id, choose_idx)), img*255.0)
cv2.imwrite(os.path.join(opt.output_path, "{}_{}_rendered.png".format(anno_id, choose_idx)), rendered_img)

for angle in [30, 60, 90]:
    rotation_angle = np.deg2rad(angle)

    augment_pose = update_pose_with_ground_truth_translation(object_pose, rotation_angle)
    # augment_poses = copy.deepcopy(poses)
    # augment_poses[:, :, choose_idx] = augment_pose[:3, :]
    render_poses = np.zeros((3, 4, 1))
    render_poses[:3, :, 0] = augment_pose[:3, :]

    meshes_aug_transed = place_meshes_graspnet(meshes, render_poses)
    rendered_aug_img = image_render(meshes_aug_transed, cam_param, output_name=output_name, width=img_length, height=img_width, is_offscreen=opt.is_offscreen)

    rendered_aug_img = rendered_aug_img[rmin:rmax, cmin:cmax, :]
    rendered_aug_img = np.pad(
        rendered_aug_img, 
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
        mode='constant', constant_values=0
    )

    rendered_rotated_img = rotate_image(rendered_img, -rotation_angle, fill_color=(0, 0, 0))
    cv2.imwrite(os.path.join(opt.output_path, "{}_{}_rendered_rot_{}.png".format(anno_id, choose_idx, angle)), rendered_rotated_img)

    rendered_aug_img = cv2.cvtColor(rendered_aug_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(opt.output_path, "{}_{}_rendered_aug_{}.png".format(anno_id, choose_idx, angle)), rendered_aug_img)

