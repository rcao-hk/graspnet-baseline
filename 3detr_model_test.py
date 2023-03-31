import argparse
import numpy as np
import torch
from models.model_3detr import build_3detr

class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # Semantic Segmentation Classes. Not used in 3DETR
        self.num_class_semseg = 20
        self.type2class_semseg = {
            "wall": 0,
            "floor": 1,
            "cabinet": 2,
            "bed": 3,
            "chair": 4,
            "sofa": 5,
            "table": 6,
            "door": 7,
            "window": 8,
            "bookshelf": 9,
            "picture": 10,
            "counter": 11,
            "desk": 12,
            "curtain": 13,
            "refrigerator": 14,
            "showercurtrain": 15,
            "toilet": 16,
            "sink": 17,
            "bathtub": 18,
            "garbagebin": 19,
        }
        self.class2type_semseg = {
            self.type2class_semseg[t]: t for t in self.type2class_semseg
        }
        self.nyu40ids_semseg = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    # def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
    #     box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
    #     boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
    #     return boxes
    #
    # def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
    #     box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
    #     boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
    #     return boxes

    # @staticmethod
    # def rotate_aligned_boxes(input_boxes, rot_mat):
    #     centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    #     new_centers = np.dot(centers, np.transpose(rot_mat))
    #
    #     dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    #     new_x = np.zeros((dx.shape[0], 4))
    #     new_y = np.zeros((dx.shape[0], 4))
    #
    #     for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
    #         crnrs = np.zeros((dx.shape[0], 3))
    #         crnrs[:, 0] = crnr[0] * dx
    #         crnrs[:, 1] = crnr[1] * dy
    #         crnrs = np.dot(crnrs, np.transpose(rot_mat))
    #         new_x[:, i] = crnrs[:, 0]
    #         new_y[:, i] = crnrs[:, 1]
    #
    #     new_dx = 2.0 * np.max(new_x, 1)
    #     new_dy = 2.0 * np.max(new_y, 1)
    #     new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)
    #
    #     return np.concatenate([new_centers, new_lengths], axis=1)


parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

##### Optimizer #####
parser.add_argument("--base_lr", default=5e-4, type=float)
parser.add_argument("--warm_lr", default=1e-6, type=float)
parser.add_argument("--warm_lr_epochs", default=9, type=int)
parser.add_argument("--final_lr", default=1e-6, type=float)
parser.add_argument("--lr_scheduler", default="cosine", type=str)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--filter_biases_wd", default=False, action="store_true")
parser.add_argument(
    "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
)

##### Model #####
parser.add_argument(
    "--model_name",
    default="3detr",
    type=str,
    help="Name of the model",
    choices=["3detr"],
)
### Encoder
parser.add_argument(
    "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
)
# Below options are only valid for vanilla encoder
parser.add_argument("--enc_nlayers", default=3, type=int)
parser.add_argument("--enc_dim", default=256, type=int)
parser.add_argument("--enc_ffn_dim", default=128, type=int)
parser.add_argument("--enc_dropout", default=0.1, type=float)
parser.add_argument("--enc_nhead", default=4, type=int)
parser.add_argument("--enc_pos_embed", default=None, type=str)
parser.add_argument("--enc_activation", default="relu", type=str)

### Decoder
parser.add_argument("--dec_nlayers", default=8, type=int)
parser.add_argument("--dec_dim", default=256, type=int)
parser.add_argument("--dec_ffn_dim", default=256, type=int)
parser.add_argument("--dec_dropout", default=0.1, type=float)
parser.add_argument("--dec_nhead", default=4, type=int)

### MLP heads for predicting bounding boxes
parser.add_argument("--mlp_dropout", default=0.3, type=float)
parser.add_argument(
    "--nsemcls",
    default=-1,
    type=int,
    help="Number of semantic object classes. Can be inferred from dataset",
)

### Other model params
parser.add_argument("--preenc_npoints", default=2048, type=int)
parser.add_argument(
    "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
)
parser.add_argument("--nqueries", default=256, type=int)
parser.add_argument("--use_color", default=False, action="store_true")

##### Set Loss #####
### Matcher
parser.add_argument("--matcher_giou_cost", default=2, type=float)
parser.add_argument("--matcher_cls_cost", default=1, type=float)
parser.add_argument("--matcher_center_cost", default=0, type=float)
parser.add_argument("--matcher_objectness_cost", default=0, type=float)

### Loss Weights
parser.add_argument("--loss_giou_weight", default=0, type=float)
parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
parser.add_argument(
    "--loss_no_object_weight", default=0.2, type=float
)  # "no object" or "background" class for detection
parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
parser.add_argument("--loss_center_weight", default=5.0, type=float)
parser.add_argument("--loss_size_weight", default=1.0, type=float)

##### Dataset #####
parser.add_argument(
    "--dataset_name", default="scannet", type=str, choices=["scannet", "sunrgbd"]
)
parser.add_argument(
    "--dataset_root_dir",
    type=str,
    default=None,
    help="Root directory containing the dataset files. \
          If None, default values from scannet.py/sunrgbd.py are used",
)
parser.add_argument(
    "--meta_data_dir",
    type=str,
    default=None,
    help="Root directory containing the metadata files. \
          If None, default values from scannet.py/sunrgbd.py are used",
)
parser.add_argument("--dataset_num_workers", default=4, type=int)
parser.add_argument("--batchsize_per_gpu", default=8, type=int)

##### Training #####
parser.add_argument("--start_epoch", default=-1, type=int)
parser.add_argument("--max_epoch", default=720, type=int)
parser.add_argument("--eval_every_epoch", default=10, type=int)
parser.add_argument("--seed", default=0, type=int)

##### Testing #####
parser.add_argument("--test_only", default=False, action="store_true")
parser.add_argument("--test_ckpt", default=None, type=str)

##### I/O #####
parser.add_argument("--checkpoint_dir", default=None, type=str)
parser.add_argument("--log_every", default=10, type=int)
parser.add_argument("--log_metrics_every", default=20, type=int)
parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

##### Distributed Training #####
parser.add_argument("--ngpus", default=1, type=int)
parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

args = parser.parse_args()
dataset_config = ScannetDatasetConfig()
model = build_3detr(args, dataset_config)
model.cuda()

input = torch.randn(1, 15000, 3).cuda()
# input = np.random.randn(1, 3, 15000).astype(np.float32)
point_cloud_dims_min = input[0].min(axis=1)[0]
point_cloud_dims_max = input[0].max(axis=1)[0]

inputs = {}
inputs["point_clouds"] = input
inputs["point_cloud_dims_min"] = point_cloud_dims_min
inputs["point_cloud_dims_max"] = point_cloud_dims_max

out = model(inputs)
