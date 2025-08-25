import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import copy
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.io as scio
import open3d as o3d
from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from PIL import Image
from graspnetAPI import GraspGroup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
cfgs = parser.parse_args()

data_root = '/media/2TB/result/mmgnet/experiment'
dataset_root = '/media/2TB/dataset/graspnet'

vis_root = '/media/2TB/result/mmgnet/vis'
os.makedirs(os.path.join(vis_root, 'feature_vis'), exist_ok=True)

geo_feat_root = os.path.join(data_root, 'ignet_v0.6.2')
rgb_feat_root = os.path.join(data_root, 'ignet_v0.8.2.26.2')
camera_type = 'realsense'

perplexity = 50
n_iter = 1000
random_seed = 0
width = 1280
height = 720
sample_ratio = 0.01
if cfgs.split == 'test_seen':
    scene_list = list(range(100, 130))  # 100-129
elif cfgs.split == 'test_similar':
    scene_list = list(range(130, 160))
elif cfgs.split == 'test_novel':
    scene_list = list(range(160, 190))

for scene_idx in tqdm(scene_list):
    for anno_idx in range(0, 255, int(1/sample_ratio)):
        geo_feat_path = os.path.join(geo_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_middle_feats.npy'.format(anno_idx))
        geo_feat = np.load(geo_feat_path, allow_pickle=True)

        rgb_feat_path = os.path.join(rgb_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_middle_feats.npy'.format(anno_idx))
        rgb_feat = np.load(rgb_feat_path, allow_pickle=True)

        geo_graspness_path = os.path.join(geo_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_graspness.npy.npz'.format(anno_idx))
        geo_graspness = np.load(geo_graspness_path)['graspness'].astype(np.float32)

        rgb_graspness_path = os.path.join(rgb_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_graspness.npy.npz'.format(anno_idx))
        rgb_graspness = np.load(rgb_graspness_path)['graspness'].astype(np.float32)

        B, C, N = rgb_feat.shape
        
        rgb_path = os.path.join(dataset_root,
                                'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera_type, anno_idx))
        depth_path = os.path.join(dataset_root,
                                  'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera_type, anno_idx))

        meta_path = os.path.join(dataset_root,
                                 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera_type, anno_idx))
        
        mask_path = os.path.join(dataset_root,
                                 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera_type, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        
        meta = scio.loadmat(meta_path)
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)

        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2],
                                 factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        depth_mask = (depth > 0)
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera_type)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera_type)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        
        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(cloud_masked)
        # scene.colors = o3d.utility.Vector3dVector(color_masked)
        # downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
        
        # gg_geo_numpy = np.load(os.path.join(geo_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_raw.npy'.format(anno_idx)), allow_pickle=True)

        # gg_rgb_numpy = np.load(os.path.join(rgb_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_raw.npy'.format(anno_idx)), allow_pickle=True)

        for batch_idx in range(B):
            rgb_inst_feat = rgb_feat[batch_idx]
            geo_inst_feat = geo_feat[batch_idx]

            rgb_inst_graspness = np.mean(rgb_graspness[batch_idx], axis=1)
            geo_inst_graspness = np.mean(geo_graspness[batch_idx], axis=1)

            rgb_inst_feat = rgb_inst_feat.T  # [N, C]
            geo_inst_feat = geo_inst_feat.T  # [N, C]

            # rgb_inst_gg = gg_rgb_numpy[batch_idx]
            # rgb_inst_gg = GraspGroup(rgb_inst_gg)
            # geo_inst_gg = gg_geo_numpy[batch_idx]
            # geo_inst_gg = GraspGroup(geo_inst_gg)

            # inst_mask = seg_masked == obj_idxs[batch_idx]

            # inst_pc_vis = o3d.geometry.PointCloud()
            # inst_pc_vis.points = o3d.utility.Vector3dVector(cloud_masked[inst_mask].astype(np.float32))
            # inst_pc_vis.colors = o3d.utility.Vector3dVector(color_masked[inst_mask].astype(np.float32))
            # geo_inst_vis = inst_pc_vis.voxel_down_sample(voxel_size=0.001)
            # rgb_inst_vis = copy.deepcopy(geo_inst_vis)
            
            # mfcdetector = ModelFreeCollisionDetectorTorch(cloud_masked[inst_mask], voxel_size=0.01)
            # collision_mask = mfcdetector.detect(rgb_inst_gg, approach_dist=0.05, collision_thresh=0.01)
            # collision_mask = collision_mask.detach().cpu().numpy()
            # rgb_inst_gg = rgb_inst_gg[~collision_mask]

            # rgb_inst_gg = rgb_inst_gg.sort_by_score()
            # rgb_inst_gg = rgb_inst_gg.nms()
            # rgb_inst_gg = rgb_inst_gg[:5]
            # rgb_inst_gg_vis = rgb_inst_gg.to_open3d_geometry_list()
            
            # for rgb_inst_g in rgb_inst_gg_vis:
            #     rgb_inst_vis += rgb_inst_g.sample_points_uniformly(number_of_points=1000)
            
            # collision_mask = mfcdetector.detect(geo_inst_gg, approach_dist=0.05, collision_thresh=0.01)
            # collision_mask = collision_mask.detach().cpu().numpy()
            # geo_inst_gg = geo_inst_gg[~collision_mask]
            
            # geo_inst_gg = geo_inst_gg.sort_by_score()
            # geo_inst_gg = geo_inst_gg.nms()
            # geo_inst_gg = geo_inst_gg[:5]
            # geo_inst_gg_vis = geo_inst_gg.to_open3d_geometry_list()

            # for geo_inst_g in geo_inst_gg_vis:
            #     geo_inst_vis += geo_inst_g.sample_points_uniformly(number_of_points=1000)
            
            # o3d.io.write_point_cloud(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_geo_grasps.ply'.format(scene_idx, anno_idx, batch_idx)), geo_inst_vis)
            
            # o3d.io.write_point_cloud(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_rgb_grasps.ply'.format(scene_idx, anno_idx, batch_idx)), rgb_inst_vis)
            
            # features = np.concatenate([rgb_inst_feat, geo_inst_feat], axis=0)
            # tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42)
            # tsne_emb = tsne.fit_transform(features)  # [N, 2]

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # tsne_geo_emb = tsne.fit_transform(geo_inst_feat)  # [N, 2]

            # plt.figure(figsize=(8, 6))
            # sc = plt.scatter(tsne_geo_emb[:, 0], tsne_geo_emb[:, 1], s=12, c=geo_inst_graspness, cmap='viridis', alpha=0.8)
            # plt.colorbar(sc, label='Graspness')
            # plt.title('t-SNE Features Colored by Graspness Value')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_point_graspness.png'.format(scene_idx, anno_idx, batch_idx)), dpi=300)
            # plt.close()

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # tsne_rgb_emb = tsne.fit_transform(rgb_inst_feat)  # [N, 2]

            # plt.figure(figsize=(8, 6))
            # sc = plt.scatter(tsne_rgb_emb[:, 0], tsne_rgb_emb[:, 1], s=12, c=rgb_inst_graspness, cmap='viridis', alpha=0.8)
            # plt.colorbar(sc, label='Graspness')
            # plt.title('t-SNE Features Colored by Graspness Value')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_rgb_graspness.png'.format(scene_idx, anno_idx, batch_idx)), dpi=300)
            # plt.close()
            
            # # # 合并特征，统一进行t-SNE降维
            # combined_features = np.concatenate([rgb_inst_feat, geo_inst_feat], axis=0)

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # combined_embedded = tsne.fit_transform(combined_features)

            # N = rgb_inst_feat.shape[0]
            # geo_embedded = combined_embedded[:N]
            # rgb_embedded = combined_embedded[N:]

            # # 可视化
            # plt.figure(figsize=(8, 6))
            # plt.scatter(geo_embedded[:, 0], geo_embedded[:, 1], s=8, c=geo_inst_graspness, alpha=0.6, label='Point-only')
            # plt.scatter(rgb_embedded[:, 0], rgb_embedded[:, 1], s=8, c=rgb_inst_graspness, alpha=0.6, label='RGB Enhanced')
            # plt.title('t-SNE of Point Features: Point-only vs RGB Enhanced')
            # plt.axis('off')
            # plt.legend()
            # # plt.savefig('ap_mean_vs_{}_noise.svg'.format(noise_type), format='svg', dpi=800)
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}.svg'.format(scene_idx, anno_idx, batch_idx)), format='svg', dpi=600)
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}.png'.format(scene_idx, anno_idx, batch_idx)), dpi=600)
            # plt.close()

            geo_scaler = StandardScaler()
            geo_inst_feat_norm = geo_scaler.fit_transform(geo_inst_feat)  # shape [N, C]

            rgb_scaler = StandardScaler()
            rgb_inst_feat_norm = rgb_scaler.fit_transform(rgb_inst_feat)  # shape [N, C]

            tsne_geo = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            tsne_geo_emb = tsne_geo.fit_transform(geo_inst_feat_norm)  # [N, 2]
            tsne_rgb = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            tsne_rgb_emb = tsne_rgb.fit_transform(rgb_inst_feat_norm)  # [N, 2]

            # --- 并排可视化 ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

            sc0 = axes[0].scatter(tsne_geo_emb[:, 0], tsne_geo_emb[:, 1], s=12, c=geo_inst_graspness, cmap='viridis', alpha=0.8)
            axes[0].set_title('Point-only Baseline Feature', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            sc1 = axes[1].scatter(tsne_rgb_emb[:, 0], tsne_rgb_emb[:, 1], s=12, c=rgb_inst_graspness, cmap='viridis', alpha=0.8)
            axes[1].set_title('MMGNet Feature', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # colorbar共用，放右侧
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(axes[1], width="4%", height="100%", loc='lower left',
                            bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=axes[1].transAxes, borderpad=0)
            cb = fig.colorbar(sc1, cax=cax)
            cb.ax.tick_params(labelsize=10)

            # fig.colorbar(sc1, ax=axes.ravel().tolist(), label='Graspness', shrink=0.7, pad=0.03)
            # plt.tight_layout()
            plt.subplots_adjust(right=0.85)  # 让整体更紧凑
            plt.savefig(os.path.join(
                vis_root, 'feature_vis', 
                f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_batch_{batch_idx}_tsne_sidebyside.png'
            ), dpi=600)
            plt.savefig(os.path.join(
                vis_root, 'feature_vis', 
                f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_batch_{batch_idx}_tsne_sidebyside.svg'
            ), format='svg', dpi=600)
            plt.close()