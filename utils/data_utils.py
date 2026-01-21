""" Tools for data processing.
    Author: Rui Cao
"""

import numpy as np
import open3d as o3d
import cv2
from scipy import ndimage

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask


def sample_points(points_len, sample_num):
    if points_len >= sample_num:
        idxs = np.random.choice(points_len, sample_num, replace=False)
    else:
        idxs1 = np.arange(points_len)
        idxs2 = np.random.choice(points_len, sample_num - points_len, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return idxs


# def points_denoise(points, pre_sample_num):
#     sampled_idxs = sample_points(len(points), pre_sample_num)
#     sampled_pcd = o3d.geometry.PointCloud()
#     sampled_pcd.points = o3d.utility.Vector3dVector(points[sampled_idxs])
    
#     cl, ind_1 = sampled_pcd.remove_statistical_outlier(nb_neighbors=80, std_ratio=2)  # default 80, 2.0
#     inst_inler1 = sampled_pcd.select_by_index(ind_1)
#     cl, ind_2 = inst_inler1.remove_statistical_outlier(nb_neighbors=1000, std_ratio=4.5) # 1000, 4.5
#     choose_idx = sampled_idxs[ind_1][ind_2]
#     return choose_idx


def points_denoise(points, pre_sample_num):
    sampled_idxs = sample_points(len(points), pre_sample_num)
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(points[sampled_idxs])
    
    cl, ind_1 = sampled_pcd.remove_statistical_outlier(nb_neighbors=80, std_ratio=3.5)  # default 80, 2.0
    choose_idx = sampled_idxs[ind_1]
    return choose_idx


def add_gaussian_noise_point_cloud(point_cloud, level=0.005, valid_min_z=0):
    """
    Adds Gaussian noise to point cloud data, suitable for point clouds with shape (N, 3), 
    where each point consists of (x, y, z) coordinates.

    Input:
    - point_cloud: numpy array, shape (N, 3), representing the point cloud data.
    - level: maximum noise intensity.
    - valid_min_z: minimum valid depth value (z-axis); noise is only added to points that meet this condition.

    Output:
    - noisy_point_cloud: point cloud data with added noise.
    """
    # 确定有效的点，仅对深度 z 大于 valid_min_z 的点添加噪声
    mask = point_cloud[:, 2] > valid_min_z
    noisy_point_cloud = point_cloud.copy()

    # 随机生成噪声级别
    noise_level = np.random.uniform(0, level)

    # 生成高斯噪声，并应用于 (x, y, z) 三个通道
    noise = noise_level * np.random.randn(*point_cloud.shape)  # (N, 3) 形状
    noisy_point_cloud[mask] += noise[mask]

    return noisy_point_cloud


def add_gaussian_noise_depth_map(depth_map, scale, level=0.005, valid_min_depth=0):
    """
    Adds Gaussian noise to a depth map, suitable for 2D depth maps with shape (H, W),
    where each value represents a depth measurement.

    Input:
    - depth_map: numpy array, shape (H, W), representing the depth map.
    - level: standard deviation of the Gaussian noise.
    - valid_min_depth: minimum valid depth value; noise is only added to pixels with depth greater than this value.

    Output:
    - noisy_depth_map: depth map with added Gaussian noise.
    """
    # 确定有效像素，仅对深度大于 valid_min_depth 的像素添加噪声
    depth_map = depth_map / scale
    mask = depth_map > valid_min_depth
    noisy_depth_map = depth_map.copy()

    # 生成高斯噪声，均值为 0，标准差为 level
    noise = np.random.normal(0, level, depth_map.shape)

    # 仅对有效像素添加噪声
    noisy_depth_map[mask] += noise[mask]
    noisy_depth_map = noisy_depth_map * scale
    return noisy_depth_map


def apply_smoothing(depth_map, size=3):
    smoothed_depth = cv2.blur(depth_map.astype(np.uint16), (size, size))
    return smoothed_depth

def random_point_dropout(point_cloud, min_num=50, num_points_to_drop=3, radius_percent=0.01):
    """
    Randomly selects a few center points in the point cloud and removes all points 
    within a spherical region centered on each selected point.

    Parameters:
    - point_cloud: numpy array of shape (N, 3), representing the point cloud data.
    - min_num: minimum acceptable number of points to retain.
    - num_points_to_drop: number of random center points to select.
    - radius_percent: percentage of the objects size to determine the radius of the spherical region (relative to the bounding box diagonal).

    Returns:
    - retained_point_cloud: the point cloud data with retained points.
    - retained_indices: indices of the retained points in the original point cloud.
    """
    num_points = point_cloud.shape[0]

    # Calculate object size using the bounding box diagonal length
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox_diagonal = np.linalg.norm(max_coords - min_coords)
    
    # Compute the radius based on the given percentage
    radius = radius_percent * bbox_diagonal

    # Initialize a mask to keep all points initially
    mask = np.ones(num_points, dtype=bool)

    # Randomly select `num_points_to_drop` points as the center points
    center_indices = np.random.choice(num_points, num_points_to_drop, replace=False)

    # For each selected center point, remove points within the radius
    for center_idx in center_indices:
        center = point_cloud[center_idx]

        # Calculate the distance from each point to the center
        distances = np.linalg.norm(point_cloud - center, axis=1)

        # Update the mask to set points within the radius to False
        mask &= distances > radius

    # Use the mask to get retained points and their indices
    retained_point_cloud = point_cloud[mask]
    
    # Ensure the retained points meet the minimum number requirement
    if len(retained_point_cloud) < min_num:
        return point_cloud, np.arange(num_points), np.array([])
    
    retained_indices = np.where(mask)[0]  # Indices of retained points
    dropped_indices = np.where(~mask)[0]  # Indices of dropped points
    return retained_point_cloud, retained_indices, dropped_indices


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


def fractal_smooth_noise(h, w, base_res=16, octaves=4, persistence=0.5, seed=0):
    """
    Perlin-like fractal smooth noise in [0,1], shape (h,w).
    base_res: 越小 -> 越大块；越大 -> 越细碎
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), np.float32)
    amp = 1.0
    total_amp = 0.0

    for o in range(octaves):
        res = max(2, int(base_res * (2 ** o)))
        gh = max(2, h // res)
        gw = max(2, w // res)

        grid = rng.random((gh, gw), dtype=np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)

        # 轻微平滑让边界更“随机形状”
        k = 2 * (o + 1) + 1
        up = cv2.GaussianBlur(up, (k, k), 0)

        noise += amp * up
        total_amp += amp
        amp *= persistence

    noise /= (total_amp + 1e-8)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def depthaware_perlin_dropout_masks(
    depth_raw, depth_clear, seg,
    dropout_rate,
    seed=0,
    base_res=16, octaves=4, persistence=0.5,
    strict_match=True,
    min_inst_pixels=50,
    use_bbox_local_noise=True
):
    """
    Return two masks:
      - drop_depth: pixels dropped due to depth missing (raw==0) (possibly subsampled if strict_match)
      - drop_perlin: pixels dropped by perlin fill to reach target rate
    Both are (H,W) bool and disjoint.
    """
    H, W = depth_clear.shape
    drop_depth = np.zeros((H, W), dtype=bool)
    drop_perlin = np.zeros((H, W), dtype=bool)

    inst_ids = np.unique(seg)
    inst_ids = inst_ids[inst_ids != 0]
    rng = np.random.default_rng(seed)

    for iid in inst_ids:
        omega = (seg == iid) & (depth_clear > 0)
        n_omega = int(omega.sum())
        if n_omega < min_inst_pixels:
            continue

        target = int(np.floor(dropout_rate * n_omega))

        # raw missing candidates
        miss = omega & (depth_raw == 0)
        miss_idx = np.flatnonzero(miss)

        # ---- stage 1: depth-guided ----
        if len(miss_idx) >= target:
            if strict_match:
                choose = rng.choice(miss_idx, size=target, replace=False)
                flat = drop_depth.reshape(-1)
                flat[choose] = True
                drop_depth = flat.reshape(H, W)
            else:
                drop_depth |= miss
            continue
        else:
            drop_depth |= miss
            need = target - len(miss_idx)
            if need <= 0:
                continue

        # ---- stage 2: perlin fill on remaining valid pixels ----
        already = drop_depth | drop_perlin
        remain = omega & (~already)
        remain_idx = np.flatnonzero(remain)
        if len(remain_idx) == 0:
            continue

        if use_bbox_local_noise:
            ys, xs = np.where(omega)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            hh, ww = (y1 - y0), (x1 - x0)
            noise = fractal_smooth_noise(
                hh, ww,
                base_res=max(4, min(base_res, max(hh, ww)//4)),
                octaves=octaves,
                persistence=persistence,
                seed=int(seed + iid * 131)
            )
            noise_map = np.zeros((H, W), np.float32)
            noise_map[y0:y1, x0:x1] = noise
        else:
            noise_map = fractal_smooth_noise(
                H, W, base_res=base_res, octaves=octaves,
                persistence=persistence, seed=int(seed + iid * 131)
            )

        scores = noise_map.reshape(-1)[remain_idx]
        need = min(need, len(remain_idx))
        pick_local = np.argpartition(scores, -need)[-need:]
        pick = remain_idx[pick_local]

        flat = drop_perlin.reshape(-1)
        flat[pick] = True
        drop_perlin = flat.reshape(H, W)

    # ensure disjoint
    drop_perlin &= (~drop_depth)
    return drop_depth, drop_perlin