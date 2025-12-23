import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

import numpy as np
import open3d as o3d
import cv2
import glob
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

import torch
import MinkowskiEngine as ME

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)

# from uois_segmentation.interface import uoisSegmentation
# from uois_sam.interface import uois_sam
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

from transforms3d._gohlketransforms import euler_matrix
from torchvision import transforms
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import plot_gripper_pro_max

# num_pt = 1024
num_point = 15000
voxel_size = 0.005
camera = 'realsense'
seed_feat_dim = 512
inst_denoise = False
multi_modal = False
rotation_filtering = True
num_view = 300
img_width = 1280
img_height = 720
checkpoint_path = 'log/gsnet_base/checkpoint.tar'

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320]
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_resized_idxs(idxs, orig_shape, resize_shape):
    orig_width, orig_length = orig_shape
    scale_x = resize_shape[1] / orig_length
    scale_y = resize_shape[0] / orig_width
    coords = np.unravel_index(idxs, (orig_width, orig_length))
    new_coords_y = np.clip((coords[0] * scale_y).astype(int), 0, resize_shape[0]-1)
    new_coords_x = np.clip((coords[1] * scale_x).astype(int), 0, resize_shape[1]-1)
    new_idxs = np.ravel_multi_index((new_coords_y, new_coords_x), resize_shape)
    return new_idxs

# camera_crop_x_left = 296
# camera_crop_x_right = 1000
# camera_crop_y_top = 60
# camera_crop_y_bottom = 508

# camera_crop_x_left = 200
# camera_crop_x_right = 1000
# camera_crop_y_top = 0
# camera_crop_y_bottom = 720

# camera_crop_x_left = 200  # 332
# camera_crop_x_right = 1000
# camera_crop_y_top = 148
# camera_crop_y_bottom = 660

camera_crop_x_left = 300  # 332
camera_crop_x_right = 1200
camera_crop_y_top = 148
camera_crop_y_bottom = 692 # 660

intrinsics = np.array([[927.17, 0., 651.32],
                       [0., 927.37, 349.62],
                       [0., 0., 1.]])
factor_depth = 1000

minimum_num_pt = 100
collision_thresh = 0.01
collision_voxel_size = 0.005
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOP_K = 5
TOP_K_execution = 1
eps = 1e-8
prepick_offset = 0.1
eef_offset = 0.04  # 0.03
trial_num = 10
filter_angle = 25
show_result_flag = True
mutli_modal = False

gripper_type = 2
velscale = 0.3
duration = 6.0  # unit:sec

init_state = [0.08929606817213488, -0.5517877865004958, -0.0970434279809019,
              -2.7017820761695006, -0.042292510363922446, 2.154521837007285, 0.8866391988343355]
place_state = [1.0489017273082568, 0.13238522675526387, -0.12584883247068349,
               -2.348927095346283, -0.04447002920508385, 2.5819659553103977, 0.8877465909168951]

init_rpy = [0.01996647412517154, -3.122374496976272, 3.071237423984786]
init_rot_base = euler_matrix(init_rpy[0], init_rpy[1], init_rpy[2])[:3, :3]

resize_shape = (224, 224)
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize(array):
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min + eps)
    return array


def compute_rot_distance(RT_1, RT_2):
    R1 = RT_1 / np.cbrt(np.linalg.det(RT_1))
    R2 = RT_2 / np.cbrt(np.linalg.det(RT_2))
    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
    return theta

def batch_matrix_to_viewpoint(batch_matrix):
    '''
    **Input:**

    - batch_matrix: numpy array of the rotation matrix with shape (n, 3, 3).

    **Output:**

    - towards: numpy array towards vectors with shape (n, 3).
    - angle: numpy array of in-plane rotations (n, ).
    '''
    # Extract the 'towards' vector, which is the first column of the rotation matrices
    towards = batch_matrix[:, :, 0]

    # Normalize the 'towards' vector to ensure it's a unit vector
    towards = towards / np.linalg.norm(towards, axis=1, keepdims=True)

    # We use the second column of R to extract the in-plane rotation 'angle'.
    # The second column is the y-axis of the camera frame in the world coordinates.
    y_axis = batch_matrix[:, :, 1]

    # For the angle, consider the original y-axis in camera frame, which after rotation aligns with 'y_axis'.
    # Since we don't rotate around the z-axis, the z-component should ideally be zero, and thus we ignore it.
    # The angle can be calculated from the y-axis component of the rotated y-axis.
    cos_angle = y_axis[:, 1]
    sin_angle = -y_axis[:, 0]

    # Compute the angle using atan2 which gives the angle relative to positive y-axis
    angle = np.arctan2(sin_angle, cos_angle)

    return towards, angle


def filter_rotation(approach, threshold=30):
    normal_dists = np.rad2deg(np.arccos((np.dot(approach,
                                                np.array([0, 0, 1]))) / (np.linalg.norm(approach, axis=1))))
    selected_indices = [i for i, d in enumerate(normal_dists) if d < threshold]
    return selected_indices


if mutli_modal:
    from models.GSNet import GraspNet_multimodal, pred_decode
    # from dataset.graspnet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
    net = GraspNet_multimodal(seed_feat_dim=seed_feat_dim, img_feat_dim=64, is_training=False)
else:
    from models.GSNet import GraspNet, pred_decode
    from dataset.graspnet_dataset import GraspNetDataset, GraspNetTransDataset, load_grasp_labels, minkowski_collate_fn
    net = GraspNet(seed_feat_dim=seed_feat_dim, is_training=False)

checkpoint = torch.load(checkpoint_path)
try:
    net.load_state_dict(checkpoint)
except:
    net.load_state_dict(checkpoint['model_state_dict'])
net.to(device)
net.eval()


def get_scene(img, cloud):
    scene = o3d.geometry.PointCloud()
    scene.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.0)
    scene.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))

    # crop_bb_rot_mat = euler_matrix(np.deg2rad(-12.0), np.deg2rad(0.0), np.deg2rad(0.0))[:3, :3]
    # crop_obb = o3d.geometry.OrientedBoundingBox(center=(0, 0, 0.6), R=crop_bb_rot_mat, extent=(0.6, 0.6, 0.14))
    crop_bb_rot_mat = np.eye(3)
    crop_obb = o3d.geometry.OrientedBoundingBox(center=(0, 0, 0.445), R=crop_bb_rot_mat, extent=(0.6, 0.6, 0.14))
    crop_obb.color = np.array([1, 0, 0])

    # cropped_scene = scene.crop(crop_obb)
    cropped_indices = crop_obb.get_point_indices_within_bounding_box(scene.points)
    cropped_scene = scene.select_by_index(cropped_indices, invert=False)  # select outside points

    # crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    # cropped_scene = scene.crop(crop_bbox)
    # o3d.visualization.draw_geometries([cropped_scene, crop_obb], width=1536, height=864)

    # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.01), fast_normal_computation=True)
    # # scene.estimate_normals(fast_normal_computation=True)
    # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    # scene.normalize_normals()
    return scene, cropped_scene, cropped_indices


def normalize_depth_image(depth_image):
    # 假设 depth_image 是一个包含原始深度值的 NumPy 数组

    # 计算5分位数和95分位数
    p5 = np.percentile(depth_image, 10)
    p95 = np.percentile(depth_image, 80)

    # 使用5分位数和95分位数作为深度值的范围
    # 将深度值限制在p5到p95之间
    depth_clipped = np.clip(depth_image, p5, p95)

    # 将限制后的深度值线性映射到0-255
    normalized_depth = 255 * (depth_clipped - p5) / (p95 - p5)

    # 将结果转换为整数类型
    normalized_depth = normalized_depth.astype(np.uint8)

    return normalized_depth


def main(color, depth):
    
    camera_info = CameraInfo(img_width, img_height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2],
                             intrinsics[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # scene = o3d.geometry.PointCloud()
    # scene.points = o3d.utility.Vector3dVector(cloud)
    # scene.colors = o3d.utility.Vector3dVector(color)

    crop_color = color[camera_crop_y_top:camera_crop_y_bottom, camera_crop_x_left:camera_crop_x_right, :]
    crop_cloud = cloud[camera_crop_y_top:camera_crop_y_bottom, camera_crop_x_left:camera_crop_x_right, :]
    crop_depth = depth[camera_crop_y_top:camera_crop_y_bottom, camera_crop_x_left:camera_crop_x_right]
    vis_color = cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB)
    # vis_depth = crop_depth * 200.0

    # vis_depth = normalize_depth_image(crop_depth)
    # vis_depth = cv2.normalize(vis_depth, None, 50, 200, cv2.NORM_MINMAX).astype(np.uint8)
    # # vis_depth = cv2.cvtColor(crop_depth, cv2.COLOR_GRAY2RGB)
    # cv2.imwrite('scene_rgb.png', vis_color)
    # cv2.imwrite('scene_depth.png', vis_depth)

    scene, cropped_scene, cropped_indices = get_scene(crop_color, crop_cloud)
    # o3d.visualization.draw_geometries([cropped_scene])

    # seg_result = uois_seg.inference(crop_color, crop_cloud)
    # seg_result = uois_seg.inference(crop_color)
    # uois_seg.show_result(save_path=os.path.join('save_data', '{}.png'.format(anno_idx)))

    cloud_masked = np.asarray(crop_cloud.reshape(-1, 3))
    color_masked = np.asarray(crop_color.reshape(-1, 3))

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
        
    cloud_tensor = torch.tensor(cloud_sampled, dtype=torch.float32, device=device)
    color_tensor = torch.tensor(color_sampled, dtype=torch.float32, device=device)


    coors_tensor = torch.tensor(cloud_sampled / voxel_size, dtype=torch.int32, device=device)
    feats_tensor = torch.ones_like(cloud_tensor).float().to(device)
    
    coordinates_batch, features_batch = ME.utils.sparse_collate([coors_tensor], [feats_tensor],
                                                                dtype=torch.float32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

    if multi_modal:
        inst_mask_org = inst_mask_org[camera_crop_y_top:camera_crop_y_bottom, camera_crop_x_left:camera_crop_x_right]
        inst_mask_choose = inst_mask_org.flatten().nonzero()[0]
        orig_width, orig_length, _ = crop_color.shape
            
        resized_idxs = get_resized_idxs(inst_mask_choose[idxs], (orig_width, orig_length), resize_shape)
        crop_color = img_transforms(crop_color)
        img_tensor = torch.tensor(crop_color, dtype=torch.int64, device=device)
        img_idxs_tensor = torch.tensor(resized_idxs, dtype=torch.int64, device=device)
    
        batch_data_label = {"point_clouds": cloud_tensor.unsqueeze(0),
                            "cloud_colors": color_tensor.unsqueeze(0),
                            "img": img_tensor,
                            "img_idxs": img_idxs_tensor,
                            "coors": coordinates_batch,
                            "feats": features_batch,
                            "quantize2original": quantize2original,
                            }
    else:
        batch_data_label = {"point_clouds": cloud_tensor.unsqueeze(0),
                            "cloud_colors": color_tensor.unsqueeze(0),
                            "coors": coordinates_batch,
                            "feats": features_batch,
                            "quantize2original": quantize2original,
                            }

    end_points = net(batch_data_label)
    grasp_preds = pred_decode(end_points)
    preds = torch.stack(grasp_preds).reshape(-1, 17).detach().cpu().numpy()
    gg = GraspGroup(preds)

    # collision detection
    if collision_thresh > 0:
        # cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
        # mfcdetector = ModelFreeCollisionDetector(crop_cloud.reshape(-1, 3), voxel_size=collision_voxel_size)
        # collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        # gg = gg[~collision_mask]

        mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=collision_voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        collision_mask = collision_mask.detach().cpu().numpy()
        gg = gg[~collision_mask]

    downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
    gg = gg.sort_by_score()
    # gg_vis = gg.random_sample(100)
    gg = gg.nms()
    gg_vis = gg[:50]
    gg_vis_geo = gg_vis.to_open3d_geometry_list()
    pcd_vis = o3d.geometry.PointCloud(downsampled_scene)
    for g in gg_vis_geo:
        pcd_vis += g.sample_points_uniformly(number_of_points=2000)
    
    o3d.io.write_point_cloud('vis.ply', pcd_vis)

    # o3d.visualization.draw_geometries([downsampled_scene] + gg_vis_geo)

    grasps_score = np.array(gg.scores)
    grasps_rotation = np.array(gg.rotation_matrices)
    grasps_trans = np.array(gg.translations)
    grasps_width = np.array(gg.widths)
    grasps_depth = np.array(gg.depths)
    grasps_direction, grasps_angle = batch_matrix_to_viewpoint(grasps_rotation)

    if rotation_filtering:
        filtered_indices = filter_rotation(grasps_direction, filter_angle)
        grasps_score = grasps_score[filtered_indices]
        grasps_width = grasps_width[filtered_indices]
        grasps_depth = grasps_depth[filtered_indices]
        grasps_rotation = grasps_rotation[filtered_indices]
        grasps_trans = grasps_trans[filtered_indices]

    # if len(suction_scores) == 0:
    #     print('All graps are filtered out, stop the trial')
    #     return 0

    try:
        execu_idx = np.argsort(grasps_score)[::-1][:TOP_K_execution]
    except:
        print('All grasp filtered out, continue')

    # execu_t = grasps_trans[execu_idx]
    # execu_r = grasps_rotation[execu_idx]
    #
    # minus_y_90_R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # execu_pose = np.identity(4)
    # execu_pose[:3, :3] = np.dot(execu_r, minus_y_90_R)
    # execu_pose[:3, 3] = execu_t

    downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
    grippers = []
    for sampled_idx, sampled_point in enumerate(grasps_trans):
        if sampled_idx not in execu_idx:
            continue
        R = grasps_rotation[sampled_idx]
        t = sampled_point
        gripper = plot_gripper_pro_max(t, R, grasps_width[sampled_idx], grasps_depth[sampled_idx], grasps_score[sampled_idx])
        grippers.append(gripper)

    if show_result_flag:
        o3d.visualization.draw_geometries([scene, *grippers], width=1536, height=864)

    vis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    prepick_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

    # vis_pose = np.identity(4)
    # vis_pose[:3, :3] = np.dot(R, minus_y_90_R)
    # vis_pose[:3, 3] = t
    # vis_frame.transform(vis_pose)
    # prepick_offset_mat = [[1., 0, 0, 0],
    #                       [0, 1., 0, 0],
    #                       [0, 0, 1., prepick_offset],
    #                       [0, 0, 0, 1.]]
    # prepick_pose = np.dot(vis_pose, prepick_offset_mat)
    # prepick_frame.transform(prepick_pose)
    if show_result_flag:
        o3d.visualization.draw_geometries([downsampled_scene, vis_frame, prepick_frame, *grippers], width=1536,
                                          height=864)


if __name__ == '__main__':
    
    data_root = '/mnt/ssd/robotarm/object_depth_percetion/assets/sample1'
    anno_idx = 0
    color = np.array(Image.open(os.path.join(data_root, 'color_{:04d}.png'.format(anno_idx))))
    depth = np.array(Image.open(os.path.join(data_root, 'depth_{:04d}.png'.format(anno_idx))))
    
    main(color=color, depth=depth)