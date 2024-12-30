import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import resource
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

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

from graspnetAPI import GraspGroup

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_depth_map, apply_smoothing, random_point_dropout
from torchvision import transforms

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
parser.add_argument('--img_feat_dim', default=64, type=int, help='Image feature dim')
parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Where dataset is')
parser.add_argument('--ckpt_root', default='/media/user/data1/rcao/result/ignet/checkpoint', help='Where checkpoint is')
parser.add_argument('--network_ver', type=str, default='v0.8.2', help='Network version')
parser.add_argument('--dump_dir', type=str, default='ignet_v0.8.2.x', help='Dump dir to save outputs')
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
parser.add_argument('--ckpt_epoch', type=int, default=53, help='Checkpoint epoch name of trained model')
parser.add_argument('--seg_root',type=str, default='/media/user/data1/rcao/result/uois/graspnet', help='Segmentation results [default: uois]')
parser.add_argument('--seg_model',type=str, default='GDS_v0.3.2', help='Segmentation results [default: uois]')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='Collision Threshold in collision detection [default: 0.0]')
parser.add_argument('--smooth_size', type=int, default=1, help='Blur size used for depth smoothing [default: 1]')
parser.add_argument('--dropout_num', type=int, default=0, help=' [default: 0]')
parser.add_argument('--downsample_voxel_size', type=float, default=0.01, help='Downsample point cloud [default: 0.0]')
# parser.add_argument('--scene_pt_num', type=int, default=0, help='Point number of each scene [default: 15000]')
cfgs = parser.parse_args()

print(cfgs)
minimum_num_pt = 30
img_width = 720
img_length = 1280

resize_shape = (224, 224)
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
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

seg_root = cfgs.seg_root
seg_model = cfgs.seg_model
if seg_model == 'gt':
    use_gt_mask = True
else:
    use_gt_mask = False
    
inst_num_pt = cfgs.inst_pt_num

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
voxel_size = cfgs.voxel_size
network_ver = cfgs.network_ver
ckpt_root = cfgs.ckpt_root
dump_dir = os.path.join(cfgs.dump_dir)
ckpt_epoch = cfgs.ckpt_epoch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if network_ver.startswith('v0.8'):
    from models.IGNet_v0_8 import IGNet, pred_decode
elif network_ver.startswith('v0.7'):
    from models.IGNet_v0_7 import IGNet, pred_decode
elif network_ver.startswith('v0.6'):
    from models.IGNet_v0_6 import IGNet, pred_decode
else:
    raise NotImplementedError
# from models.GSNet_v0_4 import IGNet, pred_decode

pattern = re.compile(rf'(epoch_{ckpt_epoch}_.+\.tar|checkpoint_{ckpt_epoch}\.tar)$')
ckpt_files = glob.glob(os.path.join(ckpt_root, 'ignet_' + network_ver, cfgs.camera, '*.tar'))

ckpt_name = None
for ckpt_path in ckpt_files:
    if pattern.search(os.path.basename(ckpt_path)):
        ckpt_name = ckpt_path
        break

try :
    assert ckpt_name is not None
    print('Load checkpoint from {}'.format(ckpt_name))
except :
    raise FileNotFoundError

if network_ver.startswith('v0.6'):
    net = IGNet(num_view=300, seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
else:
    net = IGNet(num_view=300, seed_feat_dim=cfgs.seed_feat_dim, img_feat_dim=cfgs.img_feat_dim, is_training=False, multi_scale_grouping=cfgs.multi_scale_grouping)
net.to(device)
net.eval()
checkpoint = torch.load(ckpt_name, map_location=device)

try:
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
except:
    net.load_state_dict(checkpoint, strict=True)
eps = 1e-8

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def inference(scene_idx):
    # elapsed_time_list = []
    for anno_idx in range(256):
        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

        depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
        mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
            
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        seg_mask_path = os.path.join(seg_root, '{}_mask/scene_{:04d}/{}/{:04d}.png'.format(seg_model, scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        # normal = np.load(normal_path)['normals']

        if use_gt_mask:
            net_seg = seg
        else:
            net_seg = np.array(Image.open(seg_mask_path))
            
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        
        if cfgs.smooth_size > 1:
            smooth_depth = apply_smoothing(depth, size=cfgs.smooth_size)
            smooth_cloud = create_point_cloud_from_depth_image(smooth_depth, camera_info, organized=True)
        
        if cfgs.gaussian_noise_level > 0:
            noisy_depth = add_gaussian_noise_depth_map(depth, factor_depth, level=cfgs.gaussian_noise_level, valid_min_depth=0.1)
            noisy_cloud = create_point_cloud_from_depth_image(noisy_depth, camera_info, organized=True)
            
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        depth_mask = (depth > 0)
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        if cfgs.smooth_size > 1:
            cloud_masked = smooth_cloud[mask]
        elif cfgs.gaussian_noise_level > 0:
            cloud_masked = noisy_cloud[mask]
        else:
            cloud_masked = cloud[mask]
            
        color_masked = color[mask]
        seg_masked = net_seg[mask]
        seg_masked_org = net_seg * mask

        if cfgs.downsample_voxel_size > 0.0:
            scene_masked = o3d.geometry.PointCloud()
            scene_masked.points = o3d.utility.Vector3dVector(cloud_masked)
            scene_masked.colors = o3d.utility.Vector3dVector(color_masked)
            max_bound = scene_masked.get_max_bound() + cfgs.downsample_voxel_size * 0.5
            min_bound = scene_masked.get_min_bound() - cfgs.downsample_voxel_size * 0.5
            out = scene_masked.voxel_down_sample_and_trace(cfgs.downsample_voxel_size, min_bound, max_bound, False)
            downsample_idx = [cubic_index[0] for cubic_index in out[2]]
            cloud_masked = cloud_masked[downsample_idx] 
            color_masked = color_masked[downsample_idx]
            seg_masked = seg_masked[downsample_idx]
            
            seg_masked_downsample = np.zeros_like(seg_masked_org)
            valid_indices = np.nonzero(mask)  # 获取原图像中所有有效点的 (row, col) 索引
            selected_rows = valid_indices[0][downsample_idx]  # 选中点的行坐标
            selected_cols = valid_indices[1][downsample_idx]  # 选中点的列坐标
            seg_masked_downsample[selected_rows, selected_cols] = seg_masked_org[selected_rows, selected_cols]
            seg_masked_org = seg_masked_downsample
        
        # if cfgs.scene_pt_num > 0.0:
        #     scene_sample_idx = sample_points(len(cloud_masked), cfgs.scene_pt_num)
        #     cloud_masked = cloud_masked[scene_sample_idx] 
        #     color_masked = color_masked[scene_sample_idx]
        #     seg_masked = seg_masked[scene_sample_idx]

        #     seg_masked_downsample = np.zeros_like(seg_masked_org)
        #     valid_indices = np.nonzero(mask)  # 获取原图像中所有有效点的 (row, col) 索引
        #     selected_rows = valid_indices[0][scene_sample_idx]  # 选中点的行坐标
        #     selected_cols = valid_indices[1][scene_sample_idx]  # 选中点的列坐标
        #     seg_masked_downsample[selected_rows, selected_cols] = seg_masked_org[selected_rows, selected_cols]
        #     seg_masked_org = seg_masked_downsample
        
        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(cloud_masked)
        # scene.colors = o3d.utility.Vector3dVector(color_masked)
        # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
        # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        # normal_masked = np.asarray(scene.normals)

        inst_cloud_list = []
        inst_color_list = []
        inst_coors_list = []
        inst_feats_list = []
        inst_imgs_list  = []
        inst_img_idxs_list = []
        seg_idxs = np.unique(net_seg)
        for obj_idx in seg_idxs:
            if obj_idx == 0:
                continue

            inst_mask = seg_masked == obj_idx
            inst_mask_len = inst_mask.sum()
            if inst_mask_len < minimum_num_pt:
                continue
            inst_mask_org = seg_masked_org == obj_idx

            inst_cloud = cloud_masked[inst_mask]
            inst_color = color_masked[inst_mask]

            if cfgs.dropout_num > 0:
                inst_cloud, select_idx, dropout_idx = random_point_dropout(inst_cloud, min_num=minimum_num_pt, num_points_to_drop=cfgs.dropout_num, radius_percent=0.1)
                inst_color = inst_color[select_idx]

                if len(dropout_idx) != 0:
                    # 将 inst_cloud 的索引转换回 cloud_masked 的索引
                    inst_global_idx = np.where(inst_mask_org.flatten())[0]  # inst_mask 对应的全局索引
                    dropped_global_idx = inst_global_idx[dropout_idx]  # 转换为全局索引

                    # 更新 inst_mask_org：将被丢弃的点对应的像素置为 False
                    inst_mask_org_flat = inst_mask_org.flatten()  # 展平 mask
                    inst_mask_org_flat[dropped_global_idx] = False  # 更新 mask
                    inst_mask_org = inst_mask_org_flat.reshape(inst_mask_org.shape)  # 恢复形状
                    
            idxs = sample_points(len(inst_cloud), inst_num_pt)
            
            rmin, rmax, cmin, cmax = get_bbox(inst_mask_org.astype(np.uint8))
            img = color[rmin:rmax, cmin:cmax, :]
            inst_mask_org = inst_mask_org[rmin:rmax, cmin:cmax]
            inst_mask_choose = inst_mask_org.flatten().nonzero()[0]
            orig_width, orig_length, _ = img.shape
            resized_idxs = get_resized_idxs(inst_mask_choose[idxs], (orig_width, orig_length), resize_shape)
            img = img_transforms(img)
                        
            sample_cloud = inst_cloud[idxs].astype(np.float32)
            sample_color = inst_color[idxs].astype(np.float32)
            sample_coors = inst_cloud[idxs].astype(np.float32) / voxel_size
            sample_feats = np.ones_like(inst_cloud[idxs]).astype(np.float32)

            # inst_save = o3d.geometry.PointCloud()
            # inst_save.points = o3d.utility.Vector3dVector(sample_cloud)
            # inst_save.colors = o3d.utility.Vector3dVector(sample_color)
            # o3d.io.write_point_cloud("{}_inst_input.ply".format(anno_idx), inst_save)
            
            # inst_idxs_img = np.zeros_like(img)
            # inst_idxs_img = inst_idxs_img.reshape(-1, 3)
            # inst_idxs_img[resized_idxs] = sample_color
            # inst_idxs_img = inst_idxs_img.reshape((224, 224, 3))
            # cv2.imwrite("{}_inst_input.png".format(anno_idx), inst_idxs_img*255.)
            
            inst_cloud_list.append(sample_cloud)
            inst_color_list.append(sample_color)
            inst_coors_list.append(sample_coors)
            # inst_feats_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_feats_list.append(sample_feats)            
            inst_imgs_list.append(img)
            inst_img_idxs_list.append(resized_idxs.astype(np.int64))

        inst_cloud_tensor = torch.tensor(np.array(inst_cloud_list), dtype=torch.float32, device=device)
        inst_colors_tensor = torch.tensor(np.array(inst_color_list), dtype=torch.float32, device=device)
        inst_imgs_tensor = torch.stack(inst_imgs_list, dim=0).to(device)
        inst_img_idxs_tensor = torch.tensor(np.array(inst_img_idxs_list), dtype=torch.int64, device=device)
        
        inst_coors_tensor = torch.tensor(np.array(inst_coors_list), dtype=torch.float32, device=device)
        inst_feats_tensor = torch.tensor(np.array(inst_feats_list), dtype=torch.float32, device=device)
        
        # coordinates_batch, features_batch = ME.utils.sparse_collate(inst_coors_list, inst_feats_list,
        #                                                             dtype=torch.float32)
        # coordinates_batch = coordinates_batch.to(device)
        # features_batch = features_batch.to(device)
        # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        #     coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

        batch_data_label = {"point_clouds": inst_cloud_tensor,
                            "cloud_colors": inst_colors_tensor,
                            # "cloud_normals": inst_normals_tensor,
                            "img": inst_imgs_tensor,
                            "img_idxs": inst_img_idxs_tensor,
                            "coors": inst_coors_tensor,
                            "feats": inst_feats_tensor,
                            # "coors": coordinates_batch,
                            # "feats": features_batch,
                            # "quantize2original": quantize2original,
                            }

        with torch.no_grad(): 
            end_points = net(batch_data_label)
            grasp_preds = pred_decode(end_points, normalize=False)
            preds = np.stack(grasp_preds).reshape(-1, 17)
            gg = GraspGroup(preds)
            
        # torch.cuda.empty_cache()
        # collision detection
        # 记录时间并执行前向传播
        # start.record()
        
        if cfgs.collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print('Inference Time:', elapsed_time)
        # elapsed_time_list.append(elapsed_time)

        # save grasps
        save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
        gg.save_npy(save_path)
        print('Saving {}, {}'.format(scene_idx, anno_idx))


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

for scene_idx in scene_list:
    inference(scene_idx)