""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points, add_gaussian_noise_point_cloud, apply_smoothing, random_point_dropout, add_gaussian_noise_depth_map, sample_points, find_large_missing_regions, apply_dropout_to_regions

img_width = 720
img_length = 1280
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
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


class GraspNetDataset(Dataset):
    def __init__(self, root, big_file_root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, voxel_size=0.005, gaussian_noise_level=0.0, smooth_size=1, dropout_num=0, dropout_rate=0, downsample_voxel_size=0.0, remove_invisible=True, augment=False, load_label=True, depth_type='real'):
        assert(num_points<=50000)
        self.root = root
        if big_file_root is None:
            self.big_file_root = big_file_root
        else:
            self.big_file_root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label    
        self.collision_labels = {}
        self.voxel_size = voxel_size
        self.gaussian_noise_level = gaussian_noise_level
        self.smooth_size = smooth_size
        self.dropout_num = dropout_num
        self.downsample_voxel_size = downsample_voxel_size
        self.depth_type = depth_type
        self.dropout_rate = dropout_rate
        self.dropout_min_size = 200
        assert self.depth_type in ['real', 'virtual']
        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,160) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = []
        self.depthpath = []
        self.realdepthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        # self.normalpath = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.realdepthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_depth.png'))
                if self.depth_type == 'real':
                    self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                elif self.depth_type == 'virtual':
                    self.labelpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_label.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                # self.normalpath.append(os.path.join(root, 'normals', x, camera, str(img_num).zfill(4)+'.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                if self.load_label:
                    self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
            if self.load_label:
                collision_labels = np.load(os.path.join(self.big_file_root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        real_depth = np.array(Image.open(self.realdepthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        # normal = np.load(self.normalpath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten()
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        if self.smooth_size > 1:
            smooth_depth = apply_smoothing(depth, size=self.smooth_size)
            smooth_cloud = create_point_cloud_from_depth_image(smooth_depth, camera, organized=True)

        if self.gaussian_noise_level > 0:
            noisy_depth = add_gaussian_noise_depth_map(depth, factor_depth, level=self.gaussian_noise_level, valid_min_depth=0.1)
            noisy_cloud = create_point_cloud_from_depth_image(noisy_depth, camera, organized=True)
            
        if self.dropout_rate > 0.0:
            foreground_mask = (seg > 0)
            large_missing_regions, labeled, filtered_labels = find_large_missing_regions(real_depth, foreground_mask, self.dropout_min_size)

            # 根据 dropout_rate 随机选择区域
            dropout_regions = apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, self.dropout_rate)
            dropout_mask = dropout_regions > 0
            
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
            
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        
        if return_raw_cloud:
            return cloud_masked, color_masked

        if self.smooth_size > 1:
            cloud_masked = smooth_cloud[mask]
        elif self.gaussian_noise_level > 0:
            cloud_masked = noisy_cloud[mask]
        elif self.dropout_rate > 0.0:
            mask = mask & ~dropout_mask
            cloud_masked = cloud[mask]
        else:
            cloud_masked = cloud[mask]
            
        if self.dropout_num > 0:
            inst_cloud = []
            inst_color = []
            background_mask = (seg_masked == 0)
            inst_cloud.append(cloud_masked[background_mask])
            inst_color.append(color_masked[background_mask])
            for obj_idx in obj_idxs:
                if (seg_masked == obj_idx).sum() < 50:
                    continue
                inst_mask = (seg_masked == obj_idx)
                inst_cloud_select, select_idx, dropout_idx = random_point_dropout(cloud_masked[inst_mask], min_num=50, num_points_to_drop=self.dropout_num, radius_percent=0.1)
                inst_cloud.append(inst_cloud_select)
                inst_color.append(color_masked[inst_mask][select_idx])
            cloud_masked = np.concatenate(inst_cloud, axis=0)
            color_masked = np.concatenate(inst_color, axis=0)
        
        if self.downsample_voxel_size > 0.0:
            scene_masked = o3d.geometry.PointCloud()
            scene_masked.points = o3d.utility.Vector3dVector(cloud_masked)
            scene_masked.colors = o3d.utility.Vector3dVector(color_masked)
            max_bound = scene_masked.get_max_bound() + self.downsample_voxel_size * 0.5
            min_bound = scene_masked.get_min_bound() - self.downsample_voxel_size * 0.5
            out = scene_masked.voxel_down_sample_and_trace(self.downsample_voxel_size, min_bound, max_bound, False)
            downsample_idx = [cubic_index[0] for cubic_index in out[2]]
            cloud_masked = cloud_masked[downsample_idx, :] 
            color_masked = color_masked[downsample_idx, :]
            
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            # idxs = np.concatenate([idxs1], axis=0)
        
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        # if self.gaussian_noise_level > 0.0:
        #     cloud_sampled = add_gaussian_noise_point_cloud(cloud_sampled, level=self.gaussian_noise_level, valid_min_z=0.1)
                
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['cloud_normals'] = normal_sampled.astype(np.float32)
        
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        # normal = np.load(self.normalpath[index])
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        if self.smooth_size > 1:
            depth = apply_smoothing(depth, size=self.smooth_size)
            
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
            
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        if self.dropout_num > 0:
            inst_cloud = []
            inst_color = []
            background_mask = (seg_masked == 0)
            inst_cloud.append(cloud_masked[background_mask])
            inst_color.append(color_masked[background_mask])
            for obj_idx in obj_idxs:
                if (seg_masked == obj_idx).sum() < 50:
                    continue
                inst_mask = (seg_masked == obj_idx)
                inst_cloud_select, select_idx, dropout_idx = random_point_dropout(cloud_masked[inst_mask], min_num=50, num_points_to_drop=self.dropout_num, radius_percent=0.1)
                inst_cloud.append(inst_cloud_select)
                inst_color.append(color_masked[inst_mask][select_idx])
            cloud_masked = np.concatenate(inst_cloud, axis=0)
            color_masked = np.concatenate(inst_color, axis=0)

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        # normal_sampled = normal[idxs]

        if self.gaussian_noise_level > 0.0:
            cloud_sampled = add_gaussian_noise_point_cloud(cloud_sampled, level=self.gaussian_noise_level, valid_min_z=0.1)
            
        graspness_sampled = graspness[idxs]

        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        # grasp_tolerance_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            # points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            points, offsets, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            # if self.remove_invisible:
            #     visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
            #     points = points[visible_mask]
            #     offsets = offsets[visible_mask]
            #     scores = scores[visible_mask]
            #     # tolerance = tolerance[visible_mask]
            #     collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            # tolerance = tolerance[idxs].copy()
            # tolerance[collision] = 0
            # grasp_tolerance_list.append(tolerance)
        
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['cloud_normals'] = normal_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)

        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        # ret_dict['grasp_tolerance_list'] = grasp_tolerance_list

        return ret_dict

from torchvision import transforms
class GraspNetMultiDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, voxel_size=0.005, remove_invisible=True, augment=False, load_label=True):
        assert(num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label    
        self.collision_labels = {}
        self.voxel_size = voxel_size
        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,160) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.resize_shape = (448, 448)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        # self.normalpath = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                # self.normalpath.append(os.path.join(root, 'normals', x, camera, str(img_num).zfill(4)+'.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                if self.load_label:
                    self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]


    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_resized_idxs(self, idxs, orig_shape):
        orig_width, orig_length = orig_shape
        scale_x = self.resize_shape[1] / orig_length
        scale_y = self.resize_shape[0] / orig_width
        coords = np.unravel_index(idxs, (orig_width, orig_length))
        new_coords_y = np.clip((coords[0] * scale_y).astype(int), 0, self.resize_shape[0]-1)
        new_coords_x = np.clip((coords[1] * scale_x).astype(int), 0, self.resize_shape[1]-1)
        new_idxs = np.ravel_multi_index((new_coords_y, new_coords_x), self.resize_shape)
        return new_idxs
    
    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        # normal = np.load(self.normalpath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten()
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        if return_raw_cloud:
            return cloud_masked, color_masked

        seg_masked = seg * mask
        
        rmin, rmax, cmin, cmax = get_bbox(mask)
        patch_color = color[rmin:rmax, cmin:cmax, :]
        patch_cloud = cloud[rmin:rmax, cmin:cmax, :]
        patch_seg = seg_masked[rmin:rmax, cmin:cmax]
        
        choose = patch_seg.flatten().nonzero()[0]
        sampled_idxs = sample_points(len(choose), self.num_points)
        choose = choose[sampled_idxs]

        cloud_sampled = patch_cloud.reshape(-1, 3)[choose]
        color_sampled = patch_color.reshape(-1, 3)[choose]

        orig_width, orig_length, _ = patch_color.shape
        resized_idxs = self.get_resized_idxs(choose, (orig_width, orig_length))
        img = self.img_transforms(patch_color)
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['cloud_normals'] = normal_sampled.astype(np.float32)
        
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['img'] = img
        ret_dict['img_idxs'] = resized_idxs.astype(np.int64)
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        # normal = np.load(self.normalpath[index])
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        graspness_2d = np.zeros(mask.shape, dtype=graspness.dtype)
        graspness_2d[mask] = graspness.squeeze()

        # cloud_masked = cloud[mask]
        # color_masked = color[mask]
        # seg_masked = seg[mask]
        seg_masked = seg * mask
        
        rmin, rmax, cmin, cmax = get_bbox(mask)
        
        bbox = (rmin, rmax, cmin, cmax)
        patch_color = color[rmin:rmax, cmin:cmax, :]
        patch_cloud = cloud[rmin:rmax, cmin:cmax, :]
        # cloud_masked = cloud[mask]
        # color_masked = color[mask]
        patch_seg = seg_masked[rmin:rmax, cmin:cmax]
        patch_graspness = graspness_2d[rmin:rmax, cmin:cmax]
        
        choose = patch_seg.flatten().nonzero()[0]
        sampled_idxs = sample_points(len(choose), self.num_points)
        choose = choose[sampled_idxs]

        cloud_sampled = patch_cloud.reshape(-1, 3)[choose]
        color_sampled = patch_color.reshape(-1, 3)[choose]
        seg_sampled = patch_seg.flatten()[choose]
        graspness_sampled = patch_graspness.flatten()[choose]
        
        # inst_pc_vis = o3d.geometry.PointCloud()
        # inst_pc_vis.points = o3d.utility.Vector3dVector(inst_cloud.astype(np.float32))
        # # inst_pc_vis.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32))
        # inst_pc_vis.paint_uniform_color([1.0, 0.0, 0.0])
        # combin_vis = scene_vis + inst_pc_vis
        # o3d.io.write_point_cloud('{0}_combine.ply'.format(index), combin_vis)

        orig_width, orig_length, _ = patch_color.shape
        resized_idxs = self.get_resized_idxs(choose, (orig_width, orig_length))
        img = self.img_transforms(patch_color)

        # cv2.imwrite("{}_after.png".format(index), patch_color[:, :, ::-1]*255.)
        # inst_idxs_img = np.zeros_like(img)
        # _, inst_img_width, inst_img_length = inst_idxs_img.shape
        # inst_idxs_img = inst_idxs_img.reshape(-1, 3)
        # inst_idxs_img[resized_idxs] = color_sampled[:, ::-1]
        # inst_idxs_img = inst_idxs_img.reshape((inst_img_width, inst_img_length, 3))
        # cv2.imwrite("{}_inst_input.png".format(index), inst_idxs_img*255.)
        
        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap('viridis')
        # cmap_rgb = cmap(graspness_sampled)[:, :3]
        # inst_pc_vis = o3d.geometry.PointCloud()
        # inst_pc_vis.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
        # inst_pc_vis.colors = o3d.utility.Vector3dVector(cmap_rgb)
        # o3d.io.write_point_cloud('{0}_input.ply'.format(index), inst_pc_vis)
        
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        # grasp_tolerance_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            # points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            points, offsets, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            # if self.remove_invisible:
            #     visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
            #     points = points[visible_mask]
            #     offsets = offsets[visible_mask]
            #     scores = scores[visible_mask]
            #     # tolerance = tolerance[visible_mask]
            #     collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            # tolerance = tolerance[idxs].copy()
            # tolerance[collision] = 0
            # grasp_tolerance_list.append(tolerance)
        
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        # ret_dict['cloud_normals'] = normal_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)

        ret_dict['img'] = img
        ret_dict['img_idxs'] = resized_idxs.astype(np.int64)
        
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        # ret_dict['grasp_tolerance_list'] = grasp_tolerance_list

        return ret_dict


def load_grasp_labels(root):
    obj_names = list(range(88))
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
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_idx).zfill(3))))
        grasp_labels[obj_idx+1] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))
    return valid_obj_idxs, grasp_labels


def collate_fn(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


import MinkowskiEngine as ME
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data], dtype=torch.float32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res


if __name__ == "__main__":
    import random
    def setup_seed(seed):
         torch.manual_seed(seed)
         torch.cuda.manual_seed_all(seed)
         np.random.seed(seed)
         random.seed(seed)
         torch.backends.cudnn.deterministic = True
    setup_seed(0)
    
    root = '/media/gpuadmin/rcao/dataset/graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetMultiDataset(root, valid_obj_idxs, grasp_labels, num_points=15000, camera='realsense', split='train', remove_outlier=True, voxel_size=0.002)
    # print(len(train_dataset))

    scene_list = list(range(len(train_dataset)))
    # np.random.shuffle(scene_list)
    for scene_id in scene_list[:10]:
        end_points = train_dataset[scene_id]
