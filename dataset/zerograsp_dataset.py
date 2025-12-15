import webdataset as wds
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import io
import random
import cv2
import torch as th
from torchvision import transforms
import yaml
import argparse


def parse_config(config_file_path=None):
    parser = argparse.ArgumentParser('Train a network for 3D reconstruction from a single stereo image.')
    parser.add_argument('--config', default='configs/mirage/config.yaml', help='config file')

    # General parameters
    parser.add_argument('--project_name', type=str, default='mirage')
    parser.add_argument('--model_name', type=str, default='mirage')
    parser.add_argument('--run_name', type=str, help='Run name of WandB')
    parser.add_argument('--train_dataset_name', type=str, default='mirage', help='Evaluation dataset name')
    parser.add_argument('--val_dataset_name', type=str, default='mirage', help='Validation dataset name')
    parser.add_argument('--eval_dataset_name', type=str, default=None, help='Evaluation dataset name')

    # Training parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint file')
    parser.add_argument('--train_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/wds_graspnet_fix_graspness/shard-{000000..009999}.tar', help='URL to a webdataset for training')
    parser.add_argument('--val_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/wds_graspnet_fix_graspness/shard-{010000..010001}.tar', help='URL to a webdataset for validation')
    # parser.add_argument('--val_dataset_url', type=str, default='s3://tri-ml-datasets/mirage_stereo_datasets/eval_datasets/woven_hard/shard-{000000..000003}.tar', help='URL to a webdataset for validation')
    parser.add_argument('--train_dataset_size', type=int, default=1000000)
    parser.add_argument('--val_dataset_size', type=int, default=200)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--scheduler_step', type=int, default=3000)
    parser.add_argument('--scheduler_decay', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--mode',
                        type=str,
                        choices=['training', 'overfitting', 'validation', 'test', 'training_viz', 'validation_viz'])
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--valid_frame_name', type=str, default='valid_frames')

    # General Network parameters
    parser.add_argument('--backbone_model', type=str, default=None, choices=['dinov2', 'resnext', 'featup_dinov2', None])
    parser.add_argument('--input_feature_type', type=str, default='F', choices=['F', 'P', 'L', 'FL'])
    parser.add_argument('--single_obj', default=False, action='store_true', help='Enable the demo mode')
    parser.add_argument('--predict_grasp', default=False, action='store_true', help='Enable grasp prediction')

    # For demo
    parser.add_argument('--img_path', type=str, help='Image path for demo')
    parser.add_argument('--depth_path', type=str, help='Depth map path for demo')
    parser.add_argument('--mask_path', type=str, help='Mask path for demo')
    parser.add_argument('--camera_info_path', type=str, help='Camera info path for demo')

    # Mirage
    parser.add_argument('--depth_scale', type=float, default=0.1, help='Depth scale of depth images')
    parser.add_argument('--grid_size', type=float, default=0.2, help='Grid size in meter')
    parser.add_argument('--num_local_enc_layers', type=int, default=2, help='Number of layers for the local encoder')
    parser.add_argument('--num_enc_layers', type=int, default=2, help='Number of layers for PIVOT encoder')
    parser.add_argument('--num_dec_layers', type=int, default=2, help='Number of layers for PIVOT decoder')
    # parser.add_argument('--update_lod_freq', type=int, default=3, help='Frequency of updating LoD in epochs')
    parser.add_argument('--update_octree', default=False, action='store_true', help='Should update an octree for prediction?')
    parser.add_argument('--mode_sample', default=False, action='store_true', help='Should update an octree for prediction?')
    # parser.add_argument('--init_lod', type=int, default=6, help='Initial LoD')
    parser.add_argument('--max_lod', type=int, default=9, help='Number of maximum LoD')
    parser.add_argument('--min_lod', type=int, default=6, help='Number of maximum LoD')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads for multi-head attention (MHA)')
    parser.add_argument('--max_freq', type=int, default=32, help='Max freq (resolution) of positional encoding')
    parser.add_argument('--xpos_scale_base_denom', type=int, default=2, help='Denominator for xpos_scale_base')
    parser.add_argument('--pos_emb_dim', type=int, default=32, help='Positional embedding dimension for Transformer')
    parser.add_argument('--dim_model', type=int, default=32, help='Latent feature dimension for Backbone')
    parser.add_argument('--dim_mae', type=int, default=64, help='Latent feature dimension for Transformer')
    parser.add_argument('--resid_dropout', type=float, default=0.0, help='Dropout rate for MHA')
    parser.add_argument('--ff_dropout', type=float, default=0.1, help='Dropout rate for the feedforward network')
    parser.add_argument('--ff_activation',
                        type=str,
                        default='gelu',
                        help='Activation function for the feedforward network')
    parser.add_argument('--ff_hidden_layer_multiplier',
                        type=int,
                        default=4,
                        help='Hidden layer multiplier for the feedforward network')
    parser.add_argument('--head_mlp', type=str, default='siren', choices=['mlp', 'siren'])
    # parser.add_argument('--sampling_alg', type=str, default='surface', choices=['depth', 'surface'])
    # parser.add_argument('--sample_var', type=float, default=0.1, help='Variance for sampling points around surfaces')
    parser.add_argument('--pe_type', type=str, default='rope', choices=['wo', 'rope', 'cpe', 'ape', 'rpe'], help='Positional encoding type')
    parser.add_argument('--use_cpe', default=False, action='store_true', help='Use conditional positional encoding')
    parser.add_argument('--use_rpe', default=False, action='store_true', help='Use relative positional encoding')
    parser.add_argument('--use_rope', default=True, action='store_true', help='Use rotational positional encoding')
    parser.add_argument('--enc_patch_size', type=int, default=256, help='Size of patches')
    parser.add_argument('--dec_patch_size', type=int, default=256, help='Size of patches')
    parser.add_argument('--attn_type', type=str, default='self', choices=['self', 'cross'], help='Attention type')
    parser.add_argument('--mae_type', type=str, default='full', choices=['full', 'octree', 'dsa'], help='MAE type')
    parser.add_argument('--vot_scale_factor', type=int, default=2, help='Scale factor of an image for a voxel occlusion tester')
    parser.add_argument('--ofe_scale_factor', type=int, default=2, help='Scale factor of an image for an octree feature extractor')
    parser.add_argument('--kl_weight', type=float, default=10.0, help='Weight for KL divergence loss')
    parser.add_argument('--kl_loss_cycle_len', type=float, default=20000, help='Cycle length for KL divergence loss')
    # parser.add_argument('--use_mask3d', default=False, action='store_true', help='Use Mask3D for instance segmentation')
    # parser.add_argument('--use_mask3d_scheduler', default=False, action='store_true', help='Enable the demo mode')
    # parser.add_argument('--use_mask3d_rope', default=False, action='store_true', help='Enable the demo mode')
    # parser.add_argument('--mask3d_pred_disp', default=False, action='store_true', help='Enable the demo mode')
    parser.add_argument('--oneformer3d_dim', type=int, default=48, help='Latent feature dimension for Mask3D')
    parser.add_argument('--oneformer3d_num_heads', type=int, default=8, help='Number of heads for Mask3D multi-head attention')
    parser.add_argument('--oneformer3d_num_queries', type=int, default=25, help='Number of queries for Mask3D')
    parser.add_argument('--use_mult_obj_enc', default=False, action='store_true', help='Use multi-object encoding')
    parser.add_argument('--use_sing_obj_ref', default=False, action='store_true', help='Use single object refinement')
    parser.add_argument('--use_per_obj_latent', default=False, action='store_true', help='Use a per-object latent feature z')
    parser.add_argument('--concat_obj_latent', default=False, action='store_true', help='Concatenate features to compute z')
    parser.add_argument('--use_gt_depth', default=False, action='store_true', help='Use ground-truth depth maps for training')
    parser.add_argument('--use_full_layer', default=False, action='store_true', help='Use a full layer')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent feature dimension for the MLP head network')
    parser.add_argument('--ofe_use_conv3d', default=False, action='store_true', help='Use Conv3DEncoder for OFE')
    parser.add_argument('--use_pool_and_concat', default=False, action='store_true', help='Use a pool and concate strategy')
    parser.add_argument('--use_only_visible', default=False, action='store_true', help='Use only visible regions for grasp pose prediction')
    parser.add_argument('--use_aug', default=False, action='store_true', help='Use augmentation')
    parser.add_argument('--fine_tuning', default=False, action='store_true', help='Enable fine tuning (freeze an encoder)')
    parser.add_argument('--fine_tuning_decoder', default=False, action='store_true', help='Enable fine tuning (freeze an decoder)')
    parser.add_argument('--use_collision_constraints', default=False, action='store_true', help='Use collision constraint')
    parser.add_argument('--use_collision_detection', default=False, action='store_true', help='Use collision detector')
    parser.add_argument('--use_collision_detection_only_with_depth_map', default=False, action='store_true', help='Use collision detector')


    # Convolutional Occupancy Networks and StereoPiFu
    parser.add_argument('--latent_feature_dim',
                        type=int,
                        default=32,
                        help='Latent feature dimension for the MLP head network')
    parser.add_argument('--dist_norm_factor',
                        type=float,
                        default=5.0,
                        help='Normalization factor for a relative z distance')
    parser.add_argument('--use_sigmoid_dist',
                        default=False,
                        action='store_true',
                        help='Use a sigmoid distance to compute a relative z offset?')
    parser.add_argument('--use_align_corners', default=True, help='Use align_corners=True for grid sampling?')
    parser.add_argument('--use_normal_map', default=False, action='store_true', help='Use a normal map?')
    parser.add_argument('--use_pos_enc',
                        default=False,
                        action='store_true',
                        help='Use positional encoding for a relative z offset?')
    parser.add_argument('--use_sdf', default=False, action='store_true', help='Use sdf instead of occupancy maps?')
    parser.add_argument('--use_ray', default=False, action='store_true', help='Use ray-based representation?')
    parser.add_argument('--use_voxel', default=False, action='store_true', help='Use explicit voxel representation?')
    parser.add_argument('--grid_res', type=int, default=64, help='Resolution of a voxel grid')
    parser.add_argument('--use_ground_plane',
                        default=False,
                        action='store_true',
                        help='Use explicit ground plane representation?')
    parser.add_argument('--use_triplane',
                        default=False,
                        action='store_true',
                        help='Use explicit triplane representation?')
    parser.add_argument('--plane_res', type=int, default=256, help='Resolution of a ground plane')

    args, _ = parser.parse_known_args()
    args = vars(args)
    args_default = {k: parser.get_default(k) for k in args}
    if config_file_path is None:
        args_config = yaml.load(open(args['config']), Loader=yaml.FullLoader)
    else:
        args_config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    args_inline = {k: v for (k, v) in args.items() if v != args_default[k]}
    args = args_default.copy()
    args.update(args_config)
    args.update(args_inline)
    args = argparse.Namespace(**args)
    print(args)
    return args


INTRINSICS_K = {
    "mirage": [
        [572.41136339, 0.0, 325.2611084],
        [0.0, 573.57043286, 242.04899588],
        [0.0, 0.0, 1.0],
    ],
    "graspnet": [
        [927.1697387695312, 0.0, 651.3150634765625],
        [0.0, 927.3668823242188, 349.621337890625],
        [0.0, 0.0, 1.0],
    ],
    "ycb_video": [
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0],
    ],
    "hope": [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    "hb": [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
    "woven_easy": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_normal": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_hard": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
}

shuffle_buffer = 10  # usually, pick something bigger, like 1000
url = '/data/robotarm/dataset/zerograsp/train/shard-{000000..009999}.tar'
pil_dataset = wds.WebDataset(url).shuffle(shuffle_buffer).decode("pil")

def decode_depth(key, data):
    if not (key.endswith("depth.png") or key.endswith("depth_st.png")):
        return None
    return np.asarray(iio.imread(io.BytesIO(data)), dtype=np.float32)


def rle_to_binary_mask(rle, bbox_visib=None):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
    counts = rle.get("counts")

    start = 0

    if (
        bbox_visib is not None
        and len(counts) % 2 == 0
        and bbox_visib[0] == 0
        and bbox_visib[1] == 0
    ):
        counts.insert(0, 0)

    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get("size"), order="F")

    return binary_mask


def get_camera_rays(K, img_size, inv_scale=1):
    u, v = np.meshgrid(
        np.arange(0, img_size[1], inv_scale), np.arange(0, img_size[0], inv_scale)
    )

    # Convert to homogeneous coordinates
    u = u.reshape(-1)
    v = v.reshape(-1)
    ones = np.ones(u.shape[0])
    uv1 = np.stack((u, v, ones), axis=-1)  # shape (H*W, 3)

    K_inv = np.linalg.inv(K)
    pts = np.dot(uv1, K_inv.T)  # shape (H*W, 3)
    pts = pts.reshape((img_size[0] // inv_scale, img_size[1] // inv_scale, 3))

    return pts


def make_sample_wrapper(
    config,
    is_eval=False,
    K=[
        [572.41136339, 0.0, 325.2611084],
        [0.0, 573.57043286, 242.04899588],
        [0.0, 0.0, 1.0],
    ],
):
    img_size = (config.img_height, config.img_width)  # should use a config
    should_resize_square = config.backbone_model is not None and (
        "dinov2" in config.backbone_model or "clip" in config.backbone_model
    )
    resized_img_size = (224, 224) if should_resize_square else (480, 640)
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    K = np.asarray(K, dtype=np.float32)  # should use a config
    camera_rays = get_camera_rays(K, img_size)

    transform = transforms.Compose(
        [
            transforms.Resize(resized_img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    def make_sample(sample):
        rgb = sample["rgb.jpg"].crop(
            (0, 0, config.img_width, config.img_height)
        )  # Crop a stereo image
        if is_eval or config.use_gt_depth:
            depth = sample["camera.json"]["depth_scale"] * sample["depth.png"]
        else:
            depth = sample["depth_st.png"]
        depth = depth.astype(np.float32)
        if config.train_dataset_name == "mirage" and config.use_gt_depth:
            depth += (0.5 + np.maximum((depth - 500.0) / 1000.0, 0)) * np.random.normal(
                size=depth.shape
            )
        rgb = transform(rgb)
        obj_pose = sample["gt.json"]
        obj_info = sample["gt_info.json"]
        mask_rle = sample["mask_visib.json"]
        K = np.asarray(sample["camera.json"]["cam_K"]).astype(np.float32).reshape(3, 3)
        spc = th.from_numpy(sample["spc.npz"]["spc"].astype(np.float32))
        obj_ids = th.from_numpy(sample["spc.npz"]["obj_ids"].astype(np.int32))
        if not is_eval:
            grasp_poses = th.from_numpy(
                sample["grasp.npz"]["grasp_poses"].astype(np.float32)
            )

        masks = []
        dilated_masks = []
        visible_pts_3d = []
        visible_labels = []

        rays_pts_3d = th.from_numpy(
            (camera_rays * depth[:, :, None]).astype(np.float32)
        )

        filtered_idxs = []
        idxs = np.argsort(np.array([op["obj_id"] for op in obj_pose])).tolist()
        tmp = sorted(zip(obj_info, obj_pose), key=lambda x: x[1]["obj_id"])
        obj_info, obj_pose = map(list, zip(*tmp))
        for idx, oi, op in zip(idxs, obj_info, obj_pose):
            if is_eval:
                visib_fract_thresh = 0.0
            else:
                visib_fract_thresh = 0.2

            if (
                oi["visib_fract"] <= visib_fract_thresh
            ):  # get rid of heavily occluded objects
                continue

            imask_rle = mask_rle[str(idx)]
            imask = rle_to_binary_mask(imask_rle, oi["bbox_visib"])

            float_imask = imask.astype(np.float32)
            if not is_eval:
                flag = bool(random.getrandbits(1))
                if flag:
                    kernel_size = random.choice([1, 3, 5])
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    float_imask = cv2.dilate(float_imask, kernel, iterations=1)
                else:
                    shift_x = random.uniform(-3, 3)
                    shift_y = random.uniform(-3, 3)
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    float_imask = cv2.warpAffine(
                        float_imask, M, (img_size[1], img_size[0])
                    )

            imask = float_imask > 0.5
            imask = np.logical_and(imask, depth > 10.0)
            if imask.sum() < 10:
                # print('imask sum', imask.sum(), op['obj_id'])
                continue
            masks.append(imask)

            kernel = np.ones((5, 5), np.uint8)
            dilated_imask = cv2.dilate(float_imask, kernel, iterations=1) > 0.5
            dilated_masks.append(dilated_imask)
            filtered_idxs.append(op["obj_id"])

            masked_pts_3d = rays_pts_3d[imask > 0.0].reshape(-1, 3)
            # masked_pts_3d_normals = pts_3d_normals[imask > 0.0].reshape(-1, 3)
            if config.train_dataset_name == "graspnet":
                masked_pts_3d = masked_pts_3d[::2]
                # masked_pts_3d_normals = masked_pts_3d_normals[::2]
            # print('mask num', masked_pts_3d.shape[0], op['obj_id'])
            masked_labels = (
                th.ones((masked_pts_3d.shape[0], 1), dtype=th.long) * op["obj_id"]
            )
            visible_pts_3d.append(masked_pts_3d)
            visible_labels.append(masked_labels)
            # print('obj_id:', op['obj_id'], 'px_count_visib', oi['px_count_valid'], 'maskd_pts num', masked_pts_3d.shape[0], 'visib_fract:', oi['visib_fract'])

        visible_pts_3d = th.cat(visible_pts_3d, dim=0)
        visible_labels = th.cat(visible_labels, dim=0)
        z_min = (th.min(visible_pts_3d[:, 2]) // grid_size) * grid_size - 5 * grid_size

        if config.model_name == "octmae":
            masks = np.any(np.stack(masks, axis=-1), axis=-1)
        else:
            masks = np.stack(masks)
            dilated_masks = np.stack(dilated_masks)
            flat_masks = masks.reshape(masks.shape[0], -1)
            flat_dilated_masks = dilated_masks.reshape(dilated_masks.shape[0], -1)
            overlap_matrix = np.dot(flat_dilated_masks, flat_dilated_masks.T)
            np.fill_diagonal(overlap_matrix, False)
            neighbor_masks = overlap_matrix @ flat_masks
            neighbor_masks = neighbor_masks.reshape(masks.shape[0], *img_size)
            masks = [np.stack([masks, neighbor_masks], axis=-1)]

        vdb_grid_size = 2.5
        offset = vdb_grid_size * 0.5  # this offset is to fix the misalignment in VDB

        spc_mask = th.logical_and(spc[:, 3] < vdb_grid_size, spc[:, 3] > -vdb_grid_size)
        spc = spc[spc_mask]
        obj_ids = obj_ids[spc_mask]

        if is_eval:
            features = spc[:, 3:4]
        else:
            grasp_poses = grasp_poses[spc_mask]
            if grasp_poses.shape[1] == 9:
                grasp_poses = th.cat(
                    [grasp_poses, th.zeros(grasp_poses.shape[0], 1)], dim=-1
                )
            features = th.cat([spc[:, 3:4], grasp_poses], dim=-1)
            features = th.nan_to_num(features, posinf=0.0, neginf=0.0)

        # pts_3d_in = Points(
        #     points=normalize_pts(visible_pts_3d, z_min, grid_size, grid_res),
        #     labels=visible_labels,
        # )
        # pts_3d_gt = Points(
        #     points=normalize_pts(spc[:, :3] + offset, z_min, grid_size, grid_res),
        #     normals=spc[:, 4:7],
        #     features=features,
        #     labels=obj_ids,
        # )

        # if config.use_aug and (not is_eval):
        #     tangential = pts_3d_gt.features[:, 3:6]
        #     normal = pts_3d_gt.features[:, 6:9]
        #     axis_map = {"x": 0, "y": 1, "z": 2}
        #     for axis in "xy":
        #         flag = bool(random.getrandbits(1))
        #         if flag:
        #             pts_3d_in.flip(axis)
        #             pts_3d_gt.flip(axis)
        #             tangential[:, axis_map[axis]] = -tangential[:, axis_map[axis]]
        #             normal[:, axis_map[axis]] = -normal[:, axis_map[axis]]
        #     angle = th.tensor([0.0, 0.0, (random.random() * np.pi / 3) - np.pi / 6])
        #     pts_3d_in.rotate(angle)
        #     pts_3d_gt.rotate(angle)
        #     cos, sin = angle.cos(), angle.sin()
        #     rotz = th.Tensor([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])
        #     tangential = tangential @ rotz
        #     normal = normal @ rotz
        #     pts_3d_gt.features[:, 3:6] = tangential
        #     pts_3d_gt.features[:, 6:9] = normal

        # if pts_3d_in.points.shape[0] < 100 or pts_3d_gt.points.shape[0] < 1000:
        #     frame_idx = sample["__key__"]
        #     raise Exception("This item does not have enough points", frame_idx)

        # pts_3d_in.clip()
        # pts_3d_gt.clip()

        # unique_labels = th.unique(pts_3d_in.labels)
        # filtered_idxs = np.array(filtered_idxs)
        # if not np.array_equal(unique_labels.numpy(), filtered_idxs):
        #     print(unique_labels.numpy(), filtered_idxs)
        #     raise Exception('The number of input labels is different from the number of masks')

        return (rgb, masks, depth, K, z_min, sample["__key__"])

    return make_sample

config = parse_config('dataset/zerograsp.yaml')
batch_size = config.batch_size
dataset = (
    wds.WebDataset(
        url,
        nodesplitter=wds.split_by_node,
        handler=wds.warn_and_continue,
        shardshuffle=True,
    )
    .decode(decode_depth, "pil", handler=wds.warn_and_continue)
    .map(
        make_sample_wrapper(
            config, K=INTRINSICS_K[config.train_dataset_name]
        ),
        handler=wds.warn_and_continue,
    )
    .batched(batch_size)
)

for rgb, masks, depth, K, z_min, key in dataset:
    print(f"Batch key: {key}")
    print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}, K shape: {K.shape}, z_min: {z_min}")
    print(f"Number of masks: {len(masks)}")
    # Uncomment the following lines to visualize the first image in the batch
    # plt.imshow(rgb[0].permute(1, 2, 0).numpy())
    # plt.savefig("image.png")
    # break
    
# for image, json in pil_dataset:
#     plt.imshow(image)
#     plt.savefig("image.png")
#     break

print("Dataset created with the following parameters:")
print(f"root: {url}")
print(f"Shuffle buffer size: {shuffle_buffer}") 