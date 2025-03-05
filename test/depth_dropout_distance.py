import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import argparse

import multiprocessing
from PIL import Image
# from skimage.morphology import binary_erosion, binary_dilation, disk
# from tqdm import tqdm
import torch
from pytorch3d.loss import chamfer_distance

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, create_point_cloud_from_depth_image, sample_points 

def find_large_missing_regions(depth, foreground_mask, min_size=50):
    """
    通过连通组件标记找到成块缺失的部分，滤除小的缺失区域，并仅考虑foreground mask上的缺失。

    输入:
    - depth: (H, W) 形状的 numpy 数组，表示深度图。
    - foreground_mask: (H, W) 形状的 numpy 数组，表示前景 mask（1 表示前景，0 表示背景）。
    - min_size: 连通区域的最小大小，滤除小于该大小的缺失区域。

    输出:
    - large_missing_regions: (H, W) 形状的 numpy 数组，标记出大的缺失区域。
    """
    
    # 1. 仅考虑前景区域的缺失点
    depth_mask = (depth == 0)  # 假设缺失点的深度值为 0
    valid_mask = depth_mask & foreground_mask  # 在前景区域内的缺失点

    # 2. 找到连通区域
    labeled, num_labels = ndimage.label(valid_mask)
    
    # 3. 获取各个区域的大小
    region_sizes = np.bincount(labeled.ravel())
    
    # 4. 创建一个新的 mask，标记大的缺失区域
    large_missing_regions = np.zeros_like(depth, dtype=np.int32)
    
    filtered_labels = []
    for label in range(1, num_labels + 1):
        if region_sizes[label] >= min_size:  # 如果区域的大小大于 min_size
            large_missing_regions[labeled == label] = label
            filtered_labels.append(label)
            
    return large_missing_regions, labeled, filtered_labels

def apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, dropout_rate):
    """
    根据 dropout_rate 随机选择部分区域，生成 dropout mask。

    输入:
    - large_missing_regions: (H, W) 形状的 numpy 数组，表示大缺失区域的标记。
    - labeled: (H, W) 形状的 numpy 数组，表示每个连通区域的标签。
    - filtered_labels: 连通区域标签的列表。
    - dropout_rate: 需要保留的区域比例。

    输出:
    - dropout_regions: (H, W) 形状的 numpy 数组，标记选择的 dropout 区域。
    """
    # 创建一个新的 dropout mask
    dropout_regions = np.zeros_like(large_missing_regions, dtype=np.int32)
    
    # 根据 dropout_rate 随机选择区域
    num_regions_to_keep = max(0, int(len(filtered_labels) * dropout_rate))  # 计算保留区域的数量
    if num_regions_to_keep == 0:
        return dropout_regions
    
    selected_labels = np.random.choice(filtered_labels, num_regions_to_keep, replace=False)  # 随机选择区域
    for label in selected_labels:
        dropout_regions[labeled == label] = label  # 标记选择的区域为 dropout 区域
    
    return dropout_regions

def visualize_large_missing_regions_with_labels(depth, large_missing_regions, num_labels, save_path='large_missing_regions.png'):
    """
    可视化成块缺失的部分，并为每个联通区域标上数字标签。

    输入:
    - depth: (H, W) 形状的 numpy 数组，表示深度图。
    - large_missing_regions: (H, W) 形状的 numpy 数组，表示成块缺失区域的二值掩码。
    - labeled: (H, W) 形状的 numpy 数组，包含标记的区域，每个区域有一个唯一的标签。
    - num_labels: 整数，联通区域的数量。

    输出:
    - 可视化图像。
    """
    
    # 创建一个深度图的彩色图像，用于叠加显示
    depth_color = cm.viridis(depth / np.max(depth))  # 使用 'viridis' colormap
    
    full_regions = large_missing_regions >= 1
    # 将成块缺失区域叠加为红色
    depth_color[full_regions, :] = [1, 0, 0, 1]  # 红色

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_color)
    plt.title('Large Missing Regions in Depth Map')
    plt.axis('off')
    
    label_count = 0
    # 在每个联通区域的中心位置标注标签（仅显示大于 min_size 的区域）
    for label in num_labels:
        label_count += 1
        # 找到每个区域的坐标
        region_coords = np.column_stack(np.where(large_missing_regions == label))
        
        # 计算区域的中心
        center = np.mean(region_coords, axis=0).astype(int)
        
        # 在中心位置标注区域编号
        plt.text(center[1], center[0], str(label_count), color='white', fontsize=12, ha='center', va='center')

    # 保存图像，去除白边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    return depth_color


# vis_save_root = 'dropout_depth_vis'
# os.makedirs(vis_save_root, exist_ok=True)

img_width = 720
img_length = 1280

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

intrinsics = np.array([[927.17, 0., 651.32],
                       [  0., 927.37, 349.62],
                       [  0., 0., 1.  ]])
camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 1000)
print_interval = 10

def distance_compute(scene_idx, cfgs):
    result = np.zeros((256, 1))
    dataset_root = cfgs.dataset_root
    camera = cfgs.camera_type
    with torch.no_grad():
        for anno_idx in range(256):
            real_depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
            clear_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            
            seg_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
            real_depth = np.array(Image.open(real_depth_path))
            clear_depth = np.array(Image.open(clear_depth_path))
            
            seg = np.array(Image.open(seg_path))
            foreground_mask = (seg > 0)  # 假设前景 mask（1 表示前景，0 表示背景）

            if cfgs.dropout_rate == 0:
                dropout_mask = np.zeros_like(real_depth, dtype=bool)
            else:
                # 查找成块缺失的部分
                large_missing_regions, labeled, filtered_labels = find_large_missing_regions(real_depth, foreground_mask, cfgs.min_size)

                # 根据 dropout_rate 随机选择区域
                dropout_regions = apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, cfgs.dropout_rate)
                dropout_mask = dropout_regions > 0

            real_depth_mask = (real_depth > 0) & foreground_mask
            clear_depth_mask = (clear_depth > 0) & foreground_mask & (~dropout_mask)
            real_cloud = create_point_cloud_from_depth_image(real_depth, camera_info, organized=True)
            clear_cloud = create_point_cloud_from_depth_image(clear_depth, camera_info, organized=True)
            noisy_cloud = clear_cloud[clear_depth_mask]
            real_cloud = real_cloud[real_depth_mask]
            
            noisy_idxs = sample_points(len(noisy_cloud), cfgs.match_num)
            noisy_cloud = noisy_cloud[noisy_idxs]
            real_idxs = sample_points(len(real_cloud), cfgs.match_num)
            real_cloud = real_cloud[real_idxs]
            
            noisy_cloud = torch.tensor(noisy_cloud, dtype=torch.float32, device=device)
            real_cloud = torch.tensor(real_cloud, dtype=torch.float32, device=device)
    
            # # 可视化成块缺失的区域并标注标签
            # visualize_large_missing_regions_with_labels(real_depth, dropout_regions, filtered_labels, os.path.join(vis_save_root, '{:04d}_{:04d}.png'.format(scene_idx, anno_idx)))
            
            noisy_cloud = noisy_cloud.view(1, cfgs.match_num, 3)
            real_cloud = real_cloud.view(1, cfgs.match_num, 3)

            noise_dis, _ = chamfer_distance(noisy_cloud, real_cloud)
                        
            # result[anno_idx, 0] = clear_dis.item()
            # result[anno_idx, 1] = noise_dis.item()
            # print(scene_idx, anno_idx, clear_dis.item(), noise_dis.item())
            
            result[anno_idx, 0] = noise_dis.item()
            if anno_idx % print_interval == 0:
                print(scene_idx, anno_idx, noise_dis.item())

    return result
        
def parallel(scene_ids, cfgs, proc = 2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes = proc)
    result_list = []
    for scene_id in scene_ids:
        scene_result = p.apply_async(distance_compute, (scene_id, cfgs))
        result_list.append(scene_result)
    p.close()
    p.join()
    return result_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_type', default='realsense', help='Camera to use [kinect | realsense]')
    parser.add_argument('--dataset_root', default='/data/jhpan/dataset/graspnet', help='Where dataset is')
    parser.add_argument('--min_size', type=int, default=200, help='Minimum size of missing region [default: 50]')
    parser.add_argument('--proc_num', type=int, default=10, help='Number of processes [default: 10]')
    parser.add_argument('--dropout_rate', type=float, default=0, help=' [default: 0]')
    parser.add_argument('--match_num', type=int, default=20000, help='Number of points to match [default: 20000]')
    cfgs = parser.parse_args()

    print(cfgs)
    # scene_list = list(range(100, 190))
    # scene_list = [100, 101, 102, 143, 144, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188]
    scene_list = list(range(100, 190))
    # scene_list = list(range(130, 160))
    # scene_list = list(range(160, 190))
    result_list = parallel(scene_list, cfgs=cfgs, proc=cfgs.proc_num)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    
    save_root = 'depth_distance'
    os.makedirs(save_root, exist_ok=True)
    np.save(os.path.join(save_root, 'dropout_rate_{}_depth_distance.npy'.format(cfgs.dropout_rate)), results)
    # np.save(os.path.join(save_root, 'clear_depth_distance.npy'.format(cfgs.dropout_rate)), results)