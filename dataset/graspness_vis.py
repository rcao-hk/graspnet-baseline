# visualize_graspness.py
import os
import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list

def load_depth_and_meta(dataset_root, virtual_dataset_root, scene_id, ann_id, camera_type, depth_type):
    if depth_type == 'virtual':
        depth_path = os.path.join(virtual_dataset_root, f'scene_{scene_id:04d}', camera_type, f'{ann_id:04d}_depth.png')
    else:  # 'real'
        depth_path = os.path.join(dataset_root, 'scenes', f'scene_{scene_id:04d}', camera_type, 'depth', f'{ann_id:04d}.png')

    seg_path = os.path.join(dataset_root, 'scenes', f'scene_{scene_id:04d}', camera_type, 'label', f'{ann_id:04d}.png')
    meta_path = os.path.join(dataset_root, 'scenes', f'scene_{scene_id:04d}', camera_type, 'meta', f'{ann_id:04d}.mat')
    poses_path = os.path.join(dataset_root, 'scenes', f'scene_{scene_id:04d}', camera_type, 'camera_poses.npy')
    align_path = os.path.join(dataset_root, 'scenes', f'scene_{scene_id:04d}', camera_type, 'cam0_wrt_table.npy')

    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(seg_path))
    meta = scio.loadmat(meta_path)
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    camera_poses = np.load(poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat = np.load(align_path)
    trans = align_mat @ camera_pose

    return depth, seg, camera, trans

def rebuild_mask_and_cloud(depth, seg, camera, trans):
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    depth_mask = depth > 0
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = depth_mask & workspace_mask
    cloud_masked = cloud[mask]  # (N,3)
    return cloud, mask, cloud_masked

def colormap_vals(vals_0_1, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(vals_0_1.flatten())[:, :3]  # RGB, drop alpha
    return colors.astype(np.float64)

def main(args):

    for scene_id in range(130):
        # 1) 读取深度/标注/位姿，重建与生成阶段一致的 mask 与点云
        depth, seg, camera, trans = load_depth_and_meta(
            args.dataset_root, args.virtual_dataset_root, scene_id, args.ann_id,
            args.camera_type, args.depth_type
        )
        cloud, mask, cloud_masked = rebuild_mask_and_cloud(depth, seg, camera, trans)

        # 2) 读取已保存的 graspness（已在生成阶段归一化到[0,1]）
        if args.depth_type == 'virtual':
            g_root = os.path.join(args.dataset_root, 'virtual_graspness')
        else:
            g_root = os.path.join(args.dataset_root, 'graspness')
        g_path = os.path.join(g_root, f'scene_{scene_id:04d}', args.camera_type, f'{args.ann_id:04d}.npy')
        assert os.path.exists(g_path), f'Graspness file not found: {g_path}'
        graspness = np.load(g_path).astype(np.float32)  # shape (N,1) aligned to cloud_masked
        # assert graspness.ndim == 2 and graspness.shape[1] == 1, f'Unexpected shape: {graspness.shape}'
        # assert graspness.shape[0] == cloud_masked.shape[0], \
        #     f'Count mismatch: graspness {graspness.shape[0]} vs points {cloud_masked.shape[0]}'

        # if args.depth_type == 'virtual':
        #     # 3) 3D 可视化（Open3D）并导出带颜色的 PLY
        #     colors = colormap_vals(graspness, 'viridis')
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))
        #     pcd.colors = o3d.utility.Vector3dVector(colors)
        # elif args.depth_type == 'real':
        #     # 3) 3D 可视化（Open3D）并导出带颜色的 PLY
        #     colors = colormap_vals(graspness.flatten(), 'viridis')
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3).astype(np.float64))
        #     pcd.colors = o3d.utility.Vector3dVector(colors)
            
        # os.makedirs(args.save_dir, exist_ok=True)
        # ply_out = os.path.join(args.save_dir, f'scene_{scene_id:04d}_{args.camera_type}_{args.ann_id:04d}_{args.depth_type}.ply')
        # o3d.io.write_point_cloud(ply_out, pcd, write_ascii=False, compressed=False, print_progress=False)
        # print(f'[Saved] colored point cloud -> {ply_out}')

        # 4) 2D 热力图叠加与保存
        H, W = depth.shape
        heat = np.zeros((H, W), dtype=np.float32)
        if args.depth_type == 'virtual':
            heat[mask] = graspness.squeeze()  # 映射回像素
        elif args.depth_type == 'real':
            heat = graspness.reshape(H, W)  # 已与原始深度对齐
        # 归一化以防万一（生成阶段已归一化，这里稳妥再来一次）
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min())
        heat_uint8 = (heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_VIRIDIS)  # BGR

        # 将深度归一化做灰度底图，再叠加热力图
        # depth_vis = depth.copy().astype(np.float32)
        # valid = depth_vis > 0
        # if valid.any():
        #     dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
        #     depth_vis[~valid] = dmax
        #     depth_vis = (depth_vis - dmin) / (dmax - dmin + 1e-8)
        # depth_gray = (depth_vis * 255).astype(np.uint8)
        # depth_gray_3c = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

        # overlay = cv2.addWeighted(depth_gray_3c, 0.6, heat_color, 0.4, 0.0)
        # img_out = os.path.join(args.save_dir, f'scene_{scene_id:04d}_{args.camera_type}_{args.ann_id:04d}_{args.depth_type}_overlay.png')
        # cv2.imwrite(img_out, overlay)
        # print(f'[Saved] heatmap overlay -> {img_out}')

        heat_out = os.path.join(args.save_dir, f'scene_{scene_id:04d}_{args.camera_type}_{args.ann_id:04d}_{args.depth_type}_heat.png')
        cv2.imwrite(heat_out, heat_color)
        print(f'[Saved] raw heatmap -> {heat_out}')

        # 5) 可选：弹出 3D 视窗
        # if not args.no_viewer:
        #     o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet')
    parser.add_argument('--virtual_dataset_root', default='/data/robotarm/dataset/graspnet/virtual_scenes')
    parser.add_argument('--camera_type', default='realsense', choices=['realsense', 'kinect'])
    parser.add_argument('--depth_type', default='virtual', choices=['virtual', 'real'])
    parser.add_argument('--ann_id', type=int, default=125)
    parser.add_argument('--save_dir', default='vis')
    parser.add_argument('--no_viewer', action='store_true', help='不弹出Open3D窗口，只保存文件')
    args = parser.parse_args()
    
    main(args)
