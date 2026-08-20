import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset
from graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

import torch.nn.functional as F

from utils.collision_detector import ModelFreeCollisionDetectorTorch
from models.scale_graspnet import GraspNet_MSCQ, pred_decode
from models.dsn import DSN, cluster
from dataset.scale_grasp_dataset import GraspNetDataset, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test]')
parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet', help='Dataset root')
parser.add_argument('--checkpoint_path', default='log/scale_grasp/log_full_model/checkpoint.tar', help='Model checkpoint path')
parser.add_argument('--seg_checkpoint_path', default='log/scale_grasp/log_insseg/checkpoint.tar', help='Segmentation Model checkpoint path')
parser.add_argument('--dump_dir', default='experiment/scale_grasp.512', help='Dump dir to save outputs')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--remove_outlier', action='store_true', default=True)
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='Noise level for scene points')
parser.add_argument('--smooth_size', type=int, default=0, help='Smooth size for scene points')
parser.add_argument('--dropout_num', type=int, default=0, help='Gaussian noise level for scene points')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate for scene points')
parser.add_argument('--downsample_voxel_size', type=float, default=0.0, help='Voxel Size for scene points downsample')
parser.add_argument('--depth_type', default='virtual', help='Depth type [real/virtual]')
parser.add_argument('--obs', action='store_true', default=True, help='Whether to use observation point clouds')

parser.add_argument('--enable_inference_timer', action='store_true',
                    help='Measure mean inference latency [default: False]')
parser.add_argument('--timer_warmup', type=int, default=20,
                    help='Number of initial batches excluded from timing [default: 20]')
parser.add_argument(
    '--num_inference',
    type=int,
    default=-1,
    help='Number of samples to infer; -1 means no limit [default: -1]'
)

cfgs = parser.parse_args()
print(cfgs)

if not os.path.exists(cfgs.dump_dir):
    os.makedirs(cfgs.dump_dir)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

FULL_TEST_DATASET = GraspNetDataset(
    cfgs.dataset_root, None, None,
    split=cfgs.split,
    camera=cfgs.camera,
    num_points=cfgs.num_point,
    gaussian_noise_level=cfgs.gaussian_noise_level,
    smooth_size=cfgs.smooth_size,
    dropout_num=cfgs.dropout_num,
    downsample_voxel_size=cfgs.downsample_voxel_size,
    dropout_rate=cfgs.dropout_rate,
    remove_outlier=cfgs.remove_outlier,
    augment=False,
    load_label=False,
    depth_type=cfgs.depth_type
)

full_num_samples = len(FULL_TEST_DATASET)
if cfgs.num_inference < 0:
    num_inference = full_num_samples
else:
    num_inference = min(int(cfgs.num_inference), full_num_samples)

if num_inference <= 0:
    raise ValueError(
        f"--num_inference must be -1 or a positive integer, got {cfgs.num_inference}"
    )

# Keep the original dataset for scene metadata and raw-cloud loading.
# The Subset restricts DataLoader inference to the first num_inference samples.
TEST_DATASET = Subset(FULL_TEST_DATASET, range(num_inference))
SCENE_LIST = FULL_TEST_DATASET.scene_list()

print(
    f"Inference samples: {num_inference}"
    + (" (unlimited)" if cfgs.num_inference < 0 else f" / {full_num_samples}")
)

TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=cfgs.batch_size,
    shuffle=False,
    num_workers=4,
    worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn
)
print(len(TEST_DATALOADER))

net = GraspNet_MSCQ(
    input_feature_dim=0,
    num_view=cfgs.num_view,
    num_angle=12,
    num_depth=4,
    cylinder_radius=0.08,
    hmin=-0.02,
    hmax_list=[0.01, 0.02, 0.03, 0.04],
    is_training=False,
    obs=cfgs.obs
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

checkpoint = torch.load(cfgs.checkpoint_path, weights_only=False)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

if cfgs.obs:
    seg_net = DSN(input_feature_dim=0)
    seg_net.to(device)
    checkpoint = torch.load(cfgs.seg_checkpoint_path, weights_only=False)
    seg_net.load_state_dict(checkpoint['model_state_dict'])


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def inference():
    batch_interval = 100
    net.eval()
    if cfgs.obs:
        seg_net.eval()

    inference_total_ms = 0.0
    inference_sample_count = 0

    tic = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        should_time = (
            cfgs.enable_inference_timer
            and batch_idx >= max(0, cfgs.timer_warmup)
        )

        if should_time:
            _sync_cuda()
            infer_start = time.perf_counter()

        with torch.inference_mode():
            if cfgs.obs:
                end_points = seg_net(batch_data)
                batch_xyz_img = end_points["point_clouds"]
                B, _, N = batch_xyz_img.shape
                batch_offsets = end_points["center_offsets"]
                batch_fg = end_points["foreground_logits"]
                batch_fg = F.softmax(batch_fg, dim=1)
                batch_fg = torch.argmax(batch_fg, dim=1)

                clustered_imgs = []
                for i in range(B):
                    clustered_img, uniq_cluster_centers = cluster(
                        batch_xyz_img[i],
                        batch_offsets[i].permute(1, 0),
                        batch_fg[i]
                    )
                    clustered_imgs.append(clustered_img.unsqueeze(0))
                end_points['seed_cluster'] = torch.cat(clustered_imgs, dim=0)

            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        if should_time:
            _sync_cuda()
            elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
            actual_batch_size = len(grasp_preds)
            inference_total_ms += elapsed_ms
            inference_sample_count += actual_batch_size

        actual_batch_size = len(grasp_preds)
        for i in range(actual_batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            if cfgs.collision_thresh > 0:
                cloud, _ = FULL_TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetectorTorch(
                    cloud.reshape(-1, 3),
                    voxel_size=cfgs.voxel_size
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh
                )
                collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~collision_mask]

            save_dir = os.path.join(
                cfgs.dump_dir,
                SCENE_LIST[data_idx],
                cfgs.camera
            )
            save_path = os.path.join(
                save_dir,
                str(data_idx % 256).zfill(4) + '.npy'
            )
            os.makedirs(save_dir, exist_ok=True)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            denom = batch_interval if batch_idx > 0 else 1
            print(
                'Eval batch: %d, time: %fs'
                % (batch_idx, (toc - tic) / denom)
            )
            tic = time.time()

    if cfgs.enable_inference_timer:
        if inference_sample_count > 0:
            mean_inference_ms = inference_total_ms / inference_sample_count
            print(
                "\n[INFERENCE-TIMER] "
                f"Mean inference time: {mean_inference_ms:.3f} ms/sample "
                f"(samples={inference_sample_count}, "
                f"warmup_batches={max(0, cfgs.timer_warmup)}, "
                f"batch_size={cfgs.batch_size})"
            )
        else:
            print(
                "\n[INFERENCE-TIMER][WARN] No timed samples. "
                "Reduce --timer_warmup or process more batches."
            )


if __name__ == '__main__':
    inference()
