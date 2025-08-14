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

import numpy as np
# compatible with numpy >= 1.24.4
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

import cv2
import time
import re
import glob
import argparse
import torch
from PIL import Image, ImageEnhance
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME

from graspnetAPI import GraspGroup

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise
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
parser.add_argument('--img_feat_dim', default=256, type=int, help='Image feature dim')
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Where dataset is')
parser.add_argument('--ckpt_root', default='/media/gpuadmin/rcao/result/ignet', help='Where checkpoint is')
parser.add_argument('--network_ver', type=str, default='v0.8.0', help='Network version')
parser.add_argument('--dump_dir', type=str, default='ignet_v0.8.0', help='Dump dir to save outputs')
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
parser.add_argument('--ckpt_epoch', type=int, default=48, help='Checkpoint epoch name of trained model')
parser.add_argument('--inst_denoise', action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--restored_depth', action='store_true', help='Flag to use restored depth [default: False]')
parser.add_argument('--depth_root',type=str, default='/media/gpuadmin/rcao/result/depth/v0.4', help='Restored depth path')
parser.add_argument('--seg_root',type=str, default='/media/gpuadmin/rcao/dataset/graspnet', help='Segmentation results [default: uois]')
parser.add_argument('--seg_model',type=str, default='uois', help='Segmentation results [default: uois]')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--rgb_noise', type=str, default='None', help=' [default: None]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
cfgs = parser.parse_args()

print(cfgs)
minimum_num_pt = 50
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
    

def defocus_blur(image, kernel_size=9):
    """
    Apply defocus blur (a type of Gaussian blur) to the image.
    
    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the kernel used for Gaussian blur (must be odd).
    
    Returns:
    - Defocus blurred image.
    """
    # Apply Gaussian blur to simulate defocus
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def cutout(img, size_min=0.1, size_max=0.3, ratio_1=0.3, ratio_2=1/0.3):
    img = np.array(img)  # 转换为 NumPy 数组
    img_h, img_w, img_c = img.shape  # 处理 RGB 图像，形状 (H, W, 3)

    while True:
        size = np.random.uniform(size_min, size_max) * img_h * img_w
        ratio = np.random.uniform(ratio_1, ratio_2)
        erase_w = int(np.sqrt(size / ratio))
        erase_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)

        if x + erase_w <= img_w and y + erase_h <= img_h:
            break

    # 生成全 0 遮挡区域
    value = np.zeros((erase_h, erase_w, img_c), dtype=img.dtype)

    # 应用遮挡
    img[y:y + erase_h, x:x + erase_w, :] = value

    return img


def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):

    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def colorjitter(img, brightness, contrast, saturation, hue):
    img = Image.fromarray((img * 255).astype(np.uint8))
    brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_factor = np.random.uniform(-hue, hue)

    img = adjust_brightness(img, brightness_factor)
    img = adjust_contrast(img, contrast_factor)
    img = adjust_saturation(img, saturation_factor)
    img = adjust_hue(img, hue_factor)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

data_type = 'real' # syn
restored_depth = cfgs.restored_depth
restored_depth_root = cfgs.depth_root
inst_denoise = cfgs.inst_denoise
seg_root = cfgs.seg_root
seg_model = cfgs.seg_model
if seg_model == 'gt':
    use_gt_mask = True
else:
    use_gt_mask = False
    
num_pt = cfgs.inst_pt_num
denoise_pre_sample_num = int(num_pt * 1.5)

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
voxel_size = cfgs.voxel_size
network_ver = cfgs.network_ver
ckpt_root = cfgs.ckpt_root
dump_dir = os.path.join('experiment', cfgs.dump_dir)
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
        if data_type == 'real':
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            if restored_depth:
                depth_path = os.path.join(restored_depth_root, '{}/scene_{:04d}/{:04d}.png'.format(camera, scene_idx, anno_idx))
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
        seg_mask_path = os.path.join(seg_root,
                                '{}_mask/scene_{:04d}/{}/{:04d}.png'.format(seg_model, scene_idx, camera, anno_idx))

        # suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))
        # normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHAdNGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

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
        seg_masked_org = net_seg * mask
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

            if inst_denoise:
                inst_cloud_clear_idx = points_denoise(inst_cloud, denoise_pre_sample_num)
                idxs = sample_points(len(inst_cloud_clear_idx), num_pt)
                idxs = inst_cloud_clear_idx[idxs]
            else:
                idxs = sample_points(len(inst_cloud), num_pt)
            
            rmin, rmax, cmin, cmax = get_bbox(inst_mask_org.astype(np.uint8))
            img = color[rmin:rmax, cmin:cmax, :]
            if cfgs.rgb_noise == 'blur':
                img = defocus_blur(img)
            elif cfgs.rgb_noise == 'cutout':
                img = cutout(img)
            elif cfgs.rgb_noise == 'colorjitter':
                img = colorjitter(img, 0.3, 0.3, 0.3, 0.3)
                
            # cv2.imwrite('test_{}_{}_{}.png'.format(scene_idx, anno_idx, obj_idx), img*255.0)
            inst_mask_org = inst_mask_org[rmin:rmax, cmin:cmax]
            inst_mask_choose = inst_mask_org.flatten().nonzero()[0]
            orig_width, orig_length, _ = img.shape
            resized_idxs = get_resized_idxs(inst_mask_choose[idxs], (orig_width, orig_length), resize_shape)
            img = img_transforms(img)
        
            inst_cloud_list.append(inst_cloud[idxs].astype(np.float32))
            inst_color_list.append(inst_color[idxs].astype(np.float32))
            inst_coors_list.append(inst_cloud[idxs].astype(np.float32) / voxel_size)
            # inst_feats_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_feats_list.append(np.ones_like(inst_cloud[idxs]).astype(np.float32))            
            inst_imgs_list.append(img)
            inst_img_idxs_list.append(resized_idxs.astype(np.int64))
        
        # if len(inst_cloud_list) == 0:
        #     print('No instance in scene {} anno {}'.format(scene_idx, anno_idx))
        #     save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
        #     gg = GraspGroup(np.zeros((0, 17)))
        #     gg.save_npy(save_path)
        #     continue
        
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
            # cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
            # mfcdetector = ModelFreeCollisionDetector(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            # collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            
            mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print('Inference Time:', elapsed_time)
        # elapsed_time_list.append(elapsed_time)
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
    
    # print(f"Mean Inference Time：{np.mean(elapsed_time_list[1:]):.3f} ms")


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