import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

import cv2
import time
import re
import glob
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME

from graspnetAPI import GraspGroup, GraspNetEval

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

import resource
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Where dataset is')
parser.add_argument('--ckpt_root', default='/media/gpuadmin/rcao/result/ignet', help='Where checkpoint is')
parser.add_argument('--network_ver', type=str, default='v0.6.4', help='Network version')
parser.add_argument('--dump_dir', type=str, default='ignet_v0.6.4', help='Dump dir to save outputs')
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--ckpt_epoch', type=int, default=46, help='Checkpoint epoch name of trained model')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
cfgs = parser.parse_args()

minimum_num_pt = 50
width = 1280
height = 720
# voxel_size = 0.005
# TOP_K = 300

data_type = 'real' # syn
restored_depth = False
use_gt_mask = False
seg_model = 'uois'
num_pt = cfgs.inst_pt_num
split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
ckpt_root = cfgs.ckpt_root
voxel_size = cfgs.voxel_size
network_ver = cfgs.network_ver
dump_dir = os.path.join('experiment', cfgs.dump_dir)
ckpt_epoch = cfgs.ckpt_epoch

device = torch.device("cuda:"+cfgs.gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if network_ver.startswith('v0.6'):
    from models.IGNet_v0_6 import IGNet, pred_decode
elif network_ver.startswith('v0.7'):
    from models.IGNet_v0_7 import IGNet, pred_decode
else:
    from models.GSNet import IGNet, pred_decode
# from models.GSNet_v0_4 import IGNet, pred_decode

pattern = re.compile(rf'(epoch_{ckpt_epoch}_.+\.tar|checkpoint_{ckpt_epoch}\.tar)$')
ckpt_files = glob.glob(os.path.join(ckpt_root, 'ignet_' + network_ver, '*.tar'))

ckpt_name = None
for ckpt_path in ckpt_files:
    if pattern.search(os.path.basename(ckpt_path)):
        ckpt_name = ckpt_path
        break

try :
    assert ckpt_name is not None
    print('Load checkpoint from {}'.format(ckpt_name))
except :
    print('No such checkpoint file.')
    sys.exit(1)

net = IGNet(num_view=300, seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
net.to(device)
net.eval()
checkpoint = torch.load(ckpt_name, map_location=device)

net.load_state_dict(checkpoint['model_state_dict'])
eps = 1e-8

def inference(scene_idx):
    for anno_idx in range(256):
        if data_type == 'real':
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            if restored_depth:
                depth_path = os.path.join(dataset_root,
                                          'restored_depth/scene_{:04d}/{}/{:04d}.png'.format(scene_idx, camera, anno_idx))
            else:
                depth_path = os.path.join(dataset_root,
                                          'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))   
            mask_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        elif data_type == 'syn':
            rgb_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_rgb.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
            
        meta_path = os.path.join(dataset_root,
                                'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        seg_mask_path = os.path.join(dataset_root,
                                '{}_mask/scene_{:04d}/{}/{:04d}.png'.format(seg_model, scene_idx, camera, anno_idx))

        # suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))
        # normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHAdNGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        net_seg = np.array(Image.open(seg_mask_path))
        # normal = np.load(normal_path)['normals']

        if use_gt_mask:
            net_seg = seg
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2],
                                factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        depth_mask = (depth > 0)
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = net_seg[mask]
        # normal_masked = normal

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked)
        scene.colors = o3d.utility.Vector3dVector(color_masked)
        # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
        # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        # normal_masked = np.asarray(scene.normals)

        inst_cloud_list = []
        inst_color_list = []
        inst_coors_list = []
        inst_feats_list = []
        inst_normals_list = []
        seg_idxs = np.unique(net_seg)
        for obj_idx in seg_idxs:
            if obj_idx == 0:
                continue

            inst_mask = seg_masked == obj_idx
            inst_mask_len = inst_mask.sum()
            if inst_mask_len < minimum_num_pt:
                continue
            if inst_mask_len >= num_pt:
                idxs = np.random.choice(inst_mask_len, num_pt, replace=False)
            else:
                idxs1 = np.arange(inst_mask_len)
                idxs2 = np.random.choice(inst_mask_len, num_pt - inst_mask_len, replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            inst_cloud_list.append(cloud_masked[inst_mask][idxs].astype(np.float32))
            inst_color_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_coors_list.append(cloud_masked[inst_mask][idxs].astype(np.float32) / voxel_size)
            # inst_feats_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_feats_list.append(np.ones_like(cloud_masked[inst_mask][idxs]).astype(np.float32))            
            # inst_normals_list.append(normal_masked[inst_mask][idxs].astype(np.float32))
            
            # inst_seal_score_list.append(seal_score_gt[inst_mask][idxs].astype(np.float32))
            # inst_wrench_score_list.append(wrench_score_gt[inst_mask][idxs].astype(np.float32))

        inst_cloud_tensor = torch.tensor(np.array(inst_cloud_list), dtype=torch.float32, device=device)
        inst_colors_tensor = torch.tensor(np.array(inst_color_list), dtype=torch.float32, device=device)
        # inst_normals_tensor = torch.tensor(np.array(inst_normals_list), dtype=torch.float32, device=device)
        
        coordinates_batch, features_batch = ME.utils.sparse_collate(inst_coors_list, inst_feats_list,
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

        batch_data_label = {"point_clouds": inst_cloud_tensor,
                            "cloud_colors": inst_colors_tensor,
                            # "cloud_normals": inst_normals_tensor,
                            "coors": coordinates_batch,
                            "feats": features_batch,
                            "quantize2original": quantize2original}
        
        with torch.no_grad():
            end_points = net(batch_data_label)
            grasp_preds = pred_decode(end_points, normalize=False)
            preds = np.stack(grasp_preds).reshape(-1, 17)
            gg = GraspGroup(preds)
            # torch.cuda.empty_cache()
        
        # collision detection
        if cfgs.collision_thresh > 0:
            # cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
            # mfcdetector = ModelFreeCollisionDetector(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            # collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
        # gg = gg.sort_by_score()
        # gg_vis = gg.random_sample(100)
        # gg_vis = gg[:500]
        # gg_vis_geo = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([scene] + gg_vis_geo)

        # save grasps
        save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
        gg.save_npy(save_path)
        print('Saving {}, {}'.format(scene_idx, anno_idx))
        
    # res = GraspNetEval.eval_scene(scene_id=scene_idx, dump_folder=dump_dir)
    # return res


scene_list = []
if split == 'test':
    for i in range(100, 190):
        scene_list.append(i)
elif split == 'test_seen':
    for i in range(100, 130):
        scene_list.append(i)
elif split == 'test_similar':
    for i in range(130, 160):
        scene_list.append(i)
elif split == 'test_novel':
    for i in range(160, 190):
        scene_list.append(i)
else:
    print('invalid split')

# scene_list = [100]
# res = []
for scene_idx in scene_list:
    inference(scene_idx)
    # res.append(results)

# def eval_from_array(res, split, camera):
#     ap_top50 = np.mean(res[:, :, :50, :])
#     print('\nEvaluation Result of Top 50 Suctions:\n----------\n{}, AP {}={:6f}'.format(camera, split, ap_top50))

#     ap_top50_0dot2 = np.mean(res[..., :50, 0])
#     print('----------\n{}, AP0.2 {}={:6f}'.format(camera, split, ap_top50_0dot2))

#     ap_top50_0dot4 = np.mean(res[..., :50, 1])
#     print('----------\n{}, AP0.4 {}={:6f}'.format(camera, split, ap_top50_0dot4))

#     ap_top50_0dot6 = np.mean(res[..., :50, 2])
#     print('----------\n{}, AP0.6 {}={:6f}'.format(camera, split, ap_top50_0dot6))

#     ap_top50_0dot8 = np.mean(res[..., :50, 3])
#     print('----------\n{}, AP0.8 {}={:6f}'.format(camera, split, ap_top50_0dot8))

#     ap_top1 = np.mean(res[:, :, :1, :])
#     print('\nEvaluation Result of Top 1 Suction:\n----------\n{}, AP {}={:6f}'.format(camera, split, ap_top1))

#     ap_top1_0dot2 = np.mean(res[..., :1, 0])
#     print('----------\n{}, AP0.2 {}={:6f}'.format(camera, split, ap_top1_0dot2))

#     ap_top1_0dot4 = np.mean(res[..., :1, 1])
#     print('----------\n{}, AP0.4 {}={:6f}'.format(camera, split, ap_top1_0dot4))

#     ap_top1_0dot6 = np.mean(res[..., :1, 2])
#     print('----------\n{}, AP0.6 {}={:6f}'.format(camera, split, ap_top1_0dot6))

#     ap_top1_0dot8 = np.mean(res[..., :1, 3])
#     print('----------\n{}, AP0.8 {}={:6f}'.format(camera, split, ap_top1_0dot8))


# res = np.array(res)
# eval_from_array(res, split, camera)