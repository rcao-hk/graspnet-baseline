""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import open3d as o3d
import torch
# from torch._six import container_abcs
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points, add_gaussian_noise_point_cloud, apply_smoothing, random_point_dropout, add_gaussian_noise_depth_map, find_large_missing_regions, apply_dropout_to_regions

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, gaussian_noise_level=0.0, smooth_size=1, dropout_num=0, dropout_rate=0, downsample_voxel_size=0.0, remove_invisible=True, augment=False, load_label=True, depth_type='virtual'):
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
        self.dropout_rate = dropout_rate
        self.dropout_min_size = 200
        # self.step = 100
        self.depth_type = depth_type
        assert self.depth_type in ['real', 'virtual']
        self.gaussian_noise_level = gaussian_noise_level
        self.smooth_size = smooth_size
        self.dropout_num = dropout_num
        self.downsample_voxel_size = downsample_voxel_size
        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190))
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,190))
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190))
        elif split == 'test_train':
            self.sceneIds = list(range(0,10))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = []
        self.depthpath = []
        self.realdepthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
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
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]


    def scene_list(self):
        return self.scenename

    def __len__(self):
        # return int(len(self.depthpath) / self.step)
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans,flip_mat.T)


        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans,rot_mat.T)

        return point_clouds, object_poses_list, aug_trans

    def __getitem__(self, index):
        # index = index * self.step
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
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
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
            
        if self.dropout_rate > 0:
            foreground_mask = (seg > 0)
            large_missing_regions, labeled, filtered_labels = find_large_missing_regions(real_depth, foreground_mask, self.dropout_min_size)

            # 根据 dropout_rate 随机选择区域
            dropout_regions = apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, self.dropout_rate)
            dropout_mask = dropout_regions > 0
            
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
            
        # sample points
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
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        # if self.gaussian_noise_level > 0.0:
        #     cloud_sampled = add_gaussian_noise_point_cloud(cloud_sampled, level=self.gaussian_noise_level, valid_min_z=0.1)
            
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['instance_mask'] = seg_sampled
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)

        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
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
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1

        if self.gaussian_noise_level > 0.0:
            cloud_sampled = add_gaussian_noise_point_cloud(cloud_sampled, level=self.gaussian_noise_level, valid_min_z=0.1)
            
        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        ret_dict['instance_mask'] = seg_sampled

        return ret_dict

class GraspNetSegDataset(GraspNetDataset):

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

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

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1
        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        # collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])

            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[:, :, i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3, :3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3, :3], object_poses_list[i]).astype(np.float32)

        # Compute object centers and directions
        offsets = np.zeros((len(seg_sampled), 3), dtype=np.float32)
        cf_3D_centers = np.zeros((100, 3), dtype=np.float32)  # 100 max object centers
        for i, k in enumerate(np.unique(seg_sampled)):
            mask = seg_sampled == k
            if k == 0:
                offsets[mask, ...] = 0
                continue
            # Compute 3D center
            center = np.average(cloud_sampled[mask],axis=0)
            cf_3D_centers[i - 1] = center

            # Compute directions
            object_center_offsets = (center - cloud_sampled).astype(np.float32)
            offsets[mask, ...] = object_center_offsets[mask, ...]

        foreground_mask = (seg_sampled > 0).astype(np.int64)
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        ret_dict['foreground_mask'] = foreground_mask
        ret_dict['instance_mask'] = seg_sampled
        ret_dict['cf_3D_centers'] = cf_3D_centers
        ret_dict['3D_offsets'] = offsets
        ret_dict['num_3D_centers'] = np.array(len(np.unique(seg_sampled) - 1))

        return ret_dict



def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels




def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))