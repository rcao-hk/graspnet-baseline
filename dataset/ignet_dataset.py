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

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=1024,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True, real_data=True, 
                 syn_data=False, visib_threshold=0.0, voxel_size=0.005):
        # assert(num_points<=50000)
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
        self.minimum_num_pt = 50
        self.real_data = real_data
        self.syn_data = syn_data
        self.visib_threshold = visib_threshold
        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.visibpath = []
        # self.graspnesspath = []
        # self.normalpath = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                if self.real_data:
                    self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                    self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                    # self.depthpath.append(os.path.join(root, 'restored_depth',  x, camera, str(img_num).zfill(4)+'.png'))
                    self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                    self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                    self.visibpath.append(os.path.join(root, 'visib_info', x, camera, str(img_num).zfill(4)+'.mat'))
                    # self.normalpath.append(os.path.join(root, 'normals', x, camera, str(img_num).zfill(4)+'.npz'))
                    self.scenename.append(x.strip())
                    self.frameid.append(img_num)
                                    
                if self.syn_data:
                    self.colorpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_rgb.png'))
                    self.depthpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_depth.png'))
                    self.labelpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_label.png'))
                    self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                    self.visibpath.append(os.path.join(root, 'visib_info', x, camera, str(img_num).zfill(4)+'.mat'))                    
                    # self.normalpath.append(os.path.join(root, 'normals', x, camera, str(img_num).zfill(4)+'.npz'))
                    self.scenename.append(x.strip())
                    self.frameid.append(img_num)
                # if self.load_label:
                #     self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
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

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        visib_info = scio.loadmat(self.visibpath[index])
        # normal = np.load(self.normalpath[index])['normals']
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
        # normal_masked = normal
        
        while 1:
            choose_idx = np.random.choice(np.arange(len(obj_idxs)))
            inst_mask = seg_masked == obj_idxs[choose_idx]
            inst_mask_len = inst_mask.sum()
            inst_visib_fract = float(visib_info[str(obj_idxs[choose_idx])]['visib_fract'])
            if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
                break
        
        # if return_raw_cloud:
        #     return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        inst_cloud = cloud_masked[inst_mask][idxs]
        inst_color = color_masked[inst_mask][idxs]
        # inst_normal = normal_masked[inst_mask][idxs]
        # inst_seal_score = seal_score[inst_mask][idxs]
        # cloud_sampled = cloud_masked[idxs]
        # color_sampled = color_masked[idxs]
        # inst_normal = normal[idxs]
        
        ret_dict = {}
        ret_dict['point_clouds'] = inst_cloud.astype(np.float32)
        ret_dict['cloud_colors'] = inst_color.astype(np.float32)
        # ret_dict['cloud_normals'] = inst_normal.astype(np.float32)
        
        ret_dict['coors'] = inst_cloud.astype(np.float32) / self.voxel_size
        # ret_dict['feats'] = inst_color.astype(np.float32)
        ret_dict['feats'] = np.ones_like(inst_cloud).astype(np.float32)
        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        visib_info = scio.loadmat(self.visibpath[index])
        scene = self.scenename[index]
        # graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        # normal = np.load(self.normalpath[index])['normals']
        
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
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        # normal_masked = normal
        
        while 1:
            choose_idx = np.random.choice(np.arange(len(obj_idxs)))
            inst_mask = seg_masked == obj_idxs[choose_idx]
            inst_mask_len = inst_mask.sum()
            inst_visib_fract = float(visib_info[str(obj_idxs[choose_idx])]['visib_fract'])
            if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
                break
            
        # sample points
        if inst_mask_len >= self.num_points:
            idxs = np.random.choice(inst_mask_len, self.num_points, replace=False)
        else:
            idxs1 = np.arange(inst_mask_len)
            idxs2 = np.random.choice(inst_mask_len, self.num_points - inst_mask_len, replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            
        inst_cloud = cloud_masked[inst_mask][idxs]
        inst_color = color_masked[inst_mask][idxs]
        # inst_normal = normal_masked[inst_mask][idxs]
        
        points, offsets, scores = self.grasp_labels[obj_idxs[choose_idx]]
        object_pose = poses[:, :, choose_idx]
        collision = self.collision_labels[scene][choose_idx] #(Np, V, A, D)
        # grasp_idxs = np.random.choice(len(points), min(max(int(len(points)/4), 300),len(points)), replace=False)
        
        if self.augment:
            inst_cloud, object_poses_list = self.augment_data(inst_cloud, [object_pose])
            object_pose = object_poses_list[0]
        
        grasp_idxs = np.random.choice(len(points), 350, replace=False)
        # grasp_idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 350), len(points)), replace=False)
        grasp_points = points[grasp_idxs]
        grasp_offsets = offsets[grasp_idxs]
        collision = collision[grasp_idxs].copy()
        scores = scores[grasp_idxs].copy()
        scores[collision] = 0
        grasp_scores = scores
        
        # seg_sampled = seg_masked[idxs]
        # graspness_sampled = graspness[idxs]

        # objectness_label = seg_sampled.copy()
        # objectness_label[objectness_label>1] = 1
        
        # object_poses_list = []
        # grasp_points_list = []
        # grasp_offsets_list = []
        # grasp_scores_list = []
        # grasp_tolerance_list = []
        # for i, obj_idx in enumerate(obj_idxs):
        #     # if obj_idx not in self.valid_obj_idxs:
        #     #     continue
        #     # if (seg_sampled == obj_idx).sum() < 50:
        #     #     continue
        #     # object_poses_list.append(poses[:, :, i])
        #     # points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
        #     points, offsets, scores = self.grasp_labels[obj_idx]
        #     collision = self.collision_labels[scene][i] #(Np, V, A, D)

        #     # remove invisible grasp points
        #     # if self.remove_invisible:
        #     #     visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
        #     #     points = points[visible_mask]
        #     #     offsets = offsets[visible_mask]
        #     #     scores = scores[visible_mask]
        #     #     # tolerance = tolerance[visible_mask]
        #     #     collision = collision[visible_mask]

        #     # idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
        #     grasp_points_list.append(points[idxs])
        #     grasp_offsets_list.append(offsets[idxs])
        #     collision = collision[idxs].copy()
        #     scores = scores[idxs].copy()
        #     scores[collision] = 0
        #     grasp_scores_list.append(scores)
        #     # tolerance = tolerance[idxs].copy()
        #     # tolerance[collision] = 0
        #     # grasp_tolerance_list.append(tolerance)
        ret_dict = {}
        ret_dict['point_clouds'] = inst_cloud.astype(np.float32)
        ret_dict['cloud_colors'] = inst_color.astype(np.float32)
        
        # ret_dict['cloud_normals'] = inst_normal.astype(np.float32)
        ret_dict['coors'] = inst_cloud.astype(np.float32) / self.voxel_size
        # ret_dict['feats'] = inst_color.astype(np.float32)
        ret_dict['feats'] = np.ones_like(inst_cloud).astype(np.float32)
        
        # ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        # ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        # ret_dict['object_poses_list'] = object_poses_list
        # ret_dict['grasp_points_list'] = grasp_points_list
        # ret_dict['grasp_offsets_list'] = grasp_offsets_list
        # ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['object_pose'] = object_pose.astype(np.float32)
        ret_dict['grasp_points'] = grasp_points.astype(np.float32)
        ret_dict['grasp_offsets'] = grasp_offsets.astype(np.float32)
        ret_dict['grasp_labels'] = grasp_scores.astype(np.float32)
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
    if type(batch[0]).__module__ == 'numpy':
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


def pt_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data], dtype=torch.float32)
    # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
    #     coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        # "quantize2original": quantize2original
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
    root = '/data/Benchmark/graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, split='train', remove_outlier=True, remove_invisible=True, num_points=20000)
    print(len(train_dataset))

    end_points = train_dataset[233]
    cloud = end_points['point_clouds']
    seg = end_points['objectness_label']
    print(cloud.shape)
    print(cloud.dtype)
    print(cloud[:,0].min(), cloud[:,0].max())
    print(cloud[:,1].min(), cloud[:,1].max())
    print(cloud[:,2].min(), cloud[:,2].max())
    print(seg.shape)
    print((seg>0).sum())
    print(seg.dtype)
    print(np.unique(seg))