#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import torch
import argparse

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

# 需要 MinkowskiEngine
import MinkowskiEngine as ME
from models.IGNet_v0_9 import IGNet, pred_decode
from dataset.ignet_multi_dataset import GraspNetDataset, GraspNetMultiDataset, minkowski_collate_fn, collate_fn, load_grasp_labels


def move_to_device(x, device):
    """递归把 batch 中的 tensor 搬到 device；numpy 会转 torch"""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [move_to_device(v, device) for v in x]
    # numpy array -> torch
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            return t.to(device, non_blocking=True)
    except Exception:
        pass
    return x


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

def summarize_end_points(end_points):
    """打印 end_points 的结构和形状（含 list-of-list）"""
    print("\n[end_points summary]")
    for k, v in end_points.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {tuple(v.shape)} {v.dtype} {v.device}")
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                print(f"  {k}: {type(v).__name__}(len=0)")
                continue
            # list(B) of arrays/tensors?
            if isinstance(v[0], (np.ndarray, torch.Tensor)):
                shapes = []
                for i in range(min(2, len(v))):
                    a = v[i]
                    shapes.append(tuple(a.shape) if hasattr(a, "shape") else type(a))
                print(f"  {k}: {type(v).__name__}(len={len(v)}), e.g. {shapes}")
            # list(B) of list(instances)?
            elif isinstance(v[0], (list, tuple)):
                lens = [len(x) for x in v[:min(4, len(v))]]
                print(f"  {k}: {type(v).__name__}(len={len(v)}), per-sample lens={lens}")
                # 打印第一条 sample 的一个 instance 形状
                if len(v[0]) > 0 and isinstance(v[0][0], (np.ndarray, torch.Tensor)):
                    a = v[0][0]
                    print(f"      first instance shape: {tuple(a.shape)} {('torch' if torch.is_tensor(a) else 'np')}")
            else:
                print(f"  {k}: {type(v).__name__}(len={len(v)}), elem0={type(v[0])}")
        else:
            print(f"  {k}: {type(v).__name__}")


@torch.no_grad()
def infer_img_feat_hw(model, img, device):
    """如果模型有 img_backbone，试跑一次拿到 (C,Hf,Wf)，用于生成合法 img_idxs"""
    if not hasattr(model, "img_backbone"):
        return None
    try:
        model.eval()
        feat = model.img_backbone(img.to(device))
        assert feat.dim() == 4, f"img_backbone output must be 4D, got {feat.shape}"
        _, C, Hf, Wf = feat.shape
        return (C, Hf, Wf)
    except Exception as e:
        print("[WARN] cannot infer img_backbone feature size:", repr(e))
        return None
from torch.utils.data import DataLoader

def get_one_batch_from_graspnet(cfgs, split="test_seen"):
    # 1) labels
    valid_obj_idxs, grasp_labels = load_grasp_labels(
        cfgs.big_file_root if getattr(cfgs, "big_file_root", None) is not None else cfgs.dataset_root
    )

    # 2) dataset
    if split == "train":
        dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            valid_obj_idxs,
            grasp_labels,
            camera=cfgs.camera,
            split="train",
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=cfgs.augment,
            voxel_size=cfgs.voxel_size,
        )
        shuffle = True
    else:
        dataset = GraspNetMultiDataset(
            cfgs.dataset_root,
            valid_obj_idxs,
            grasp_labels,
            camera=cfgs.camera,
            split=split,
            num_points=cfgs.num_point,
            remove_outlier=True,
            augment=False,
            voxel_size=cfgs.voxel_size,
        )
        shuffle = False

    print(f"[INFO] dataset({split}) len = {len(dataset)}")

    # 3) dataloader
    loader = DataLoader(
        dataset,
        batch_size=cfgs.batch_size,
        shuffle=shuffle,
        num_workers=cfgs.worker_num,
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn,          # 你自己的 collate_fn
        pin_memory=cfgs.pin_memory,
        drop_last=True,                 # debug 时更稳
        persistent_workers=(cfgs.worker_num > 0),
    )
    print(f"[INFO] loader({split}) len = {len(loader)}")

    # 4) get one batch
    batch = next(iter(loader))
    return batch


def summarize_output(out, prefix=""):
    if torch.is_tensor(out):
        print(f"{prefix}Tensor {tuple(out.shape)} {out.dtype} {out.device}")
    elif isinstance(out, dict):
        print(f"{prefix}dict(keys={list(out.keys())})")
        for k, v in out.items():
            summarize_output(v, prefix + f"  [{k}] ")
    elif isinstance(out, (list, tuple)):
        print(f"{prefix}{type(out).__name__}(len={len(out)})")
        for i, v in enumerate(out):
            summarize_output(v, prefix + f"  ({i}) ")
    else:
        print(f"{prefix}{type(out).__name__}: {out}")


# CUDA_VISIBLE_DEVICES=0 python train_ignet.py --method_id 'mmgnet_scene_aug_new' --batch_size 6 --worker_num 3 --camera 'realsense' --dataset_root '/data/robotarm/dataset/graspnet' --big_file_root '/data/robotarm/dataset/graspnet' --ckpt_root 'log' --num_point 20000 --match_point_num 350 --learning_rate 0.0015  --lr_sched_period 6 --max_epoch 24 --seed_feat_dim 256 --img_feat_dim 64 --lr_sched --weight_decay 1e-3 --eval_start_epoch 6 --pin_memory --voxel_size 0.002 --m_point 1024  --fuse_type 'early' --augment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet', help='Dataset root')
    parser.add_argument('--big_file_root', default='/data/robotarm/dataset/graspnet', help='Big file root')
    parser.add_argument('--camera', default='realsense', help='Big file root')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
    parser.add_argument('--augment', action='store_true', help='Set point_augment for point cloud augmentation [default: False]')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 2]')
    parser.add_argument('--worker_num', type=int, default=3, help='Worker number for dataloader [default: 4]')
    parser.add_argument('--pin_memory', action='store_true', help='Set pin_memory for faster training [default: False]')
    
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--N", type=int, default=1024, help="points per sample (keep small for debug)")
    parser.add_argument("--voxel_grid", type=int, default=128)
    parser.add_argument("--H", type=int, default=448)
    parser.add_argument("--W", type=int, default=448)
    parser.add_argument("--auto_img_idx", action="store_true", help="use img_backbone output Hf*Wf for img_idxs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    # ==========================
    model = IGNet(m_point=1024, num_view=300, seed_feat_dim=256, img_feat_dim=64, is_training=False, multi_scale_grouping=False, fuse_type='intermediate')
    model = model.to(device).eval()

    # optional：根据 img_backbone 输出大小自动生成 img_idxs（避免 gather 越界）
    auto_hw = None
    if args.auto_img_idx:
        dummy_img = torch.rand((args.B, 3, args.H, args.W), dtype=torch.float32, device=device)
        auto_hw = infer_img_feat_hw(model, dummy_img, device)
        if auto_hw is not None:
            print(f"[INFO] img_backbone feat: C={auto_hw[0]}, Hf={auto_hw[1]}, Wf={auto_hw[2]}")

    # ====== get real batch ======
    end_points = get_one_batch_from_graspnet(args, split="test_seen")

    # 搬到 GPU（兼容 list-of-list）
    end_points = move_to_device(end_points, device)
    
    # summarize_end_points(end_points)

    print("\n[INFO] running forward ...")
    with torch.no_grad():
        out = model(end_points)

    print("\n[INFO] output summary:")
    summarize_output(out)


if __name__ == "__main__":
    main()
