import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging purposes
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
import open3d as o3d

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from utils.collision_detector import ModelFreeCollisionDetectorTorch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/2TB/dataset/graspnet')
parser.add_argument('--sim_dataset_root', default='/media/2TB/dataset/graspnet_sim/graspnet_trans_full', help='Root directory for simulated dataset')
parser.add_argument('--split', default='test_novel', help='Dataset split [default: test_seen]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default='/media/2TB/result/mmgnet/checkpoint/gsnet_virtual/realsense/epoch18.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/media/2TB/result/grasp/graspnet_trans_full/15000/gsnet_virtual_ours_restored')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--remove_outlier', action='store_true', default=False)
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--worker_num', type=int, default=18, help='Worker number for dataloader [default: 4]')
parser.add_argument('--depth_type', default='restored', help='Depth type [real/virtual]')
parser.add_argument('--depth_result_root', default='/media/2TB/result/depth/graspnet_trans_full/dreds_dav2_complete_obs_iter_unc_cali_convgru_l1_only_0.5_l1+grad_sigma_conf_320x180/vitl', help='Root directory for depth results')
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for restored depth')
parser.add_argument('--mutli_modal', action='store_true', default=False)
cfgs = parser.parse_args()
print(cfgs)
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)

if cfgs.mutli_modal:
    from models.GSNet import GraspNet_multimodal, pred_decode
    # from dataset.graspnet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
else:
    from models.GSNet import GraspNet, pred_decode
    from dataset.graspnet_dataset import GraspNetDataset, GraspNetTransDataset, load_grasp_labels, minkowski_collate_fn

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    if cfgs.mutli_modal:
        valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
        test_dataset = GraspNetMultiDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, load_label=False, augment=False)
        print('Test dataset length: ', len(test_dataset))
        scene_list = test_dataset.scene_list()
        test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                     num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, 
                                     collate_fn=collate_fn)
        print('Test dataloader length: ', len(test_dataloader))
    else:
        # test_dataset = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, split=cfgs.split, camera=cfgs.camera, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, gaussian_noise_level=cfgs.gaussian_noise_level, smooth_size=cfgs.smooth_size, dropout_num=cfgs.dropout_num, dropout_rate=cfgs.dropout_rate, downsample_voxel_size=cfgs.downsample_voxel_size, remove_outlier=cfgs.remove_outlier, augment=False, load_label=False, depth_type=cfgs.depth_type, depth_result_root=cfgs.depth_result_root)
        test_dataset = GraspNetTransDataset(cfgs.dataset_root, cfgs.sim_dataset_root, camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=cfgs.remove_outlier, augment=False, depth_type=cfgs.depth_type, depth_result_root=cfgs.depth_result_root, conf_threshold=cfgs.conf_threshold)
        print('Test dataset length: ', len(test_dataset))
        scene_list = test_dataset.scene_list()
        test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                    num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
        print('Test dataloader length: ', len(test_dataloader))
            
    # Init the model
    if cfgs.mutli_modal:
        net = GraspNet_multimodal(seed_feat_dim=cfgs.seed_feat_dim, img_feat_dim=64, is_training=False)
    else:
        net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    batch_interval = 100
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(test_dataloader):
        
        # if batch_idx <= 3056 or batch_idx >3060:
        #     continue
        # cloud, _ = test_dataset.get_data(batch_idx, return_raw_cloud=True)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
        # o3d.io.write_point_cloud(os.path.join('vis', 'cloud_{}.ply'.format(batch_idx)), pcd)
        
        for key in batch_data:
            # try:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
            # except:
            #     # print(f"Error in batch data: {key}, {batch_data[key]}")
            #     print(f"Error during load data: {scene_list[data_idx]}, {data_idx % 256}")
            #     gg = GraspGroup()
            #     gg.save_npy(os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera, str(data_idx % 256).zfill(4) + '.npy'))
            #     continue
                
        # Forward pass
        with torch.no_grad():
            # try:
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)
            # except:
            #     data_idx = batch_idx * cfgs.batch_size
            #     print(f"Error during forward pass: {scene_list[data_idx]}, {data_idx % 256}")
            #     gg = GraspGroup()
            #     gg.save_npy(os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera, str(data_idx % 256).zfill(4) + '.npy'))
            #     continue
            
        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = test_dataset.get_data(data_idx, return_raw_cloud=True)
                # mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()


if __name__ == '__main__':
    inference()