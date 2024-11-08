import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import MinkowskiEngine as ME
from .minkowski import MinkUNet14D

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from pytorch3d.ops.knn import knn_points
import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import RectangularQueryAndGroup, RectangularQuery, furthest_point_sample, gather_operation
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix, batch_get_key_points, transform_point_cloud, GRASPNESS_THRESHOLD, GRASP_MAX_WIDTH, NUM_ANGLE, NUM_VIEW, NUM_DEPTH, M_POINT
from models.coral_loss import corn_label_from_logits
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


base_depth = 0.04
angles = torch.tensor([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)])
views = generate_grasp_views(NUM_VIEW)  # num of views, (300,3), np.float32
# views_repeat = views.repeat_interleave(NUM_ANGLE, 0)  # (300*12,3)
# angles_repeat = angles.view(1, NUM_ANGLE).repeat_interleave(NUM_VIEW, 0).view(-1)  # (300*12,)
angles_repeat = angles.tile(NUM_VIEW)
views_repeat = views.repeat_interleave(NUM_ANGLE, dim=0)
grasp_rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat)  # (300, 12, 9)
depths = torch.linspace(0.01, 0.04, 4)
# score_bins = torch.tensor([0.2, 0.4, 0.6, 0.8])
# width_bins = torch.tensor([0.02, 0.04, 0.06, 0.08])

# v0.6.3.1
score_bins = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
width_bins = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])

width_intervals = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])


def value_unbucketize(bucketized_data, bins):
    """
    根据 bins 自动计算 bucket means，并将 bucketized_data 反映射回其原始数值范围。

    参数:
    - bucketized_data: 包含 bucket 索引的多维张量。
    - bins: 各个 bin 的边界值。

    返回:
    - 反映射后的原始数据的近似值。
    """
    # 计算每个 bin 的中间值
    bin_means = (bins[1:] + bins[:-1]) / 2.0
    # 添加 bins 第一个和最后一个区间的中点，这里假设它们与相邻的 bin 间隔相同
    first_mean = bins[0] - (bins[1] - bins[0]) / 2
    last_mean = bins[-1] + (bins[-1] - bins[-2]) / 2
    bin_means = torch.cat([first_mean[None], bin_means, last_mean[None]]).to(bucketized_data.device)

    # 如果bucketized_data是多维的，需要调整bin_means的形状以匹配所有除索引维之外的维度
    if bucketized_data.dim() > 1:
        # 维度-1保持维度不变，其他维度添加1以便进行广播
        view_shape = [-1] + [1] * (bucketized_data.dim() - 1)
        bin_means = bin_means.view(*view_shape)

    # 扩展 bin_means 以完全匹配 bucketized_data 的形状
    expanded_size = list(bucketized_data.shape)
    expanded_size[0] = -1
    bin_means = bin_means.expand(*expanded_size)

    # 使用 torch.gather 在第一个维度上收集数据
    return torch.gather(bin_means, 0, bucketized_data)



def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='trunc')
    return tuple(reversed(out))


def transform_views(views, trans):
    view_num = len(views)
    sampled_num = len(trans)
    views_ = views.unsqueeze(0).tile((sampled_num, 1, 1))
    trans = trans.unsqueeze(1).tile((1, view_num, 1, 1))

    views_trans = torch.matmul(trans, views_.unsqueeze(-1))
    views_trans = views_trans.squeeze(-1)
    return views_trans


def batch_normal_matrix(normal_vectors):
    normal_vectors = -normal_vectors
    normal_vectors /= torch.linalg.norm(normal_vectors, dim=1, keepdim=True)
    u = torch.tensor([0.0, 0.0, 1.0], device=normal_vectors.device).tile((len(normal_vectors), 1))  # Up direction
    s = torch.cross(normal_vectors, u)  # Side direction
    non_zero_mask = torch.count_nonzero(s, dim=1) == 0
    s[non_zero_mask] = torch.tensor([1.0, 0.0, 0.0], device=normal_vectors.device)
    s /= torch.linalg.norm(s, dim=1, keepdim=True)
    u = torch.cross(s, normal_vectors)  # Recompute up
    s = s.unsqueeze(-1)
    u = u.unsqueeze(-1)
    # normal_vectors = normal_vectors.unsqueeze(-1)
    # R = torch.cat([normal_vectors, s, u], dim=2)
    normal_vectors = -normal_vectors.unsqueeze(-1)
    R = torch.cat([s, u, normal_vectors], dim=2)
    return R


def normalize_tensor(tensor, eps=1e-8):
    # 计算最后一维上的最大值和最小值
    max_val = torch.amax(tensor, dim=-1, keepdim=True)
    min_val = torch.amin(tensor, dim=-1, keepdim=True)
    
    # 对最后一维进行归一化
    tensor = (tensor - min_val) / (max_val - min_val + eps)
    return tensor


def knn_key_points_matching_sym(p1_key_points, p2_key_points, p2_key_points_sym):
    dis, inds_, _ = knn_points(p1_key_points, p2_key_points, K=1)
    dis_sym, inds_sym_, _ = knn_points(p1_key_points, p2_key_points_sym, K=1)
    sym_mask = torch.lt(dis, dis_sym)
    inds = inds_ * sym_mask + inds_sym_ * (~sym_mask)
    return inds


def align_angle_index(grasp_rot, grasp_rot_trans, grasp_scores, grasp_widths):
    pred_grasp_rot_mat_ = grasp_rot.clone().to(grasp_rot_trans.device)
    pred_widths = 0.02 * torch.ones(len(pred_grasp_rot_mat_)).to(grasp_rot_trans.device)
    pred_depths = 0.02 * torch.ones(len(pred_grasp_rot_mat_)).to(grasp_rot_trans.device)
    orig_points = torch.zeros((len(pred_grasp_rot_mat_), 3)).to(grasp_rot_trans.device)
    pred_key_points, pred_key_points_sym = batch_get_key_points(orig_points, pred_grasp_rot_mat_, pred_widths, pred_depths)
    pred_key_points = pred_key_points.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))
    pred_key_points_sym = pred_key_points_sym.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))

    grasp_rot_trans_ = grasp_rot_trans.view((-1, 3, 3))
    temp_key_points, temp_key_points_sym = batch_get_key_points(orig_points, grasp_rot_trans_, pred_widths, pred_depths)
    temp_key_points = temp_key_points.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))
    temp_key_points_sym = temp_key_points_sym.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))

    view_angle_inds = knn_key_points_matching_sym(pred_key_points, temp_key_points, temp_key_points_sym)
    
    view_angle_inds = view_angle_inds.squeeze(-1)
    view_angle_rot_inds = view_angle_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
    view_angle_score_inds = view_angle_inds.unsqueeze(0).unsqueeze(-1).expand(len(grasp_scores), -1, -1, NUM_DEPTH)
    
    grasp_rot_trans = torch.gather(grasp_rot_trans, 1, view_angle_rot_inds)  # (NUM_VIEWS, NUM_ANGLE, 3, 3)
    grasp_scores = torch.gather(grasp_scores, 2, view_angle_score_inds)
    grasp_widths = torch.gather(grasp_widths, 2, view_angle_score_inds)
    return grasp_rot_trans, grasp_scores, grasp_widths


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max, _ = torch.max(view_score_, dim=2)
            view_score_min, _ = torch.min(view_score_, dim=2)
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = []
            for i in range(B):
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch)
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


# class CloudCrop(nn.Module):
#     def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
#         super().__init__()
#         self.nsample = nsample
#         self.in_dim = seed_feature_dim
#         self.cylinder_radius = cylinder_radius
#         mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

#         self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
#                                              use_xyz=True, normalize_xyz=True)
#         self.mlps = pt_utils.SharedMLP(mlps, bn=True)

#     def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
#         grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
#                                        seed_features_graspable)  # B*3 + feat_dim*M*K
#         new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
#         new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
#         new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
#         return new_features


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        # mlps = [self.in_dim, 256, 256]
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3
        self.grouper = RectangularQueryAndGroup(nsample=nsample, use_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, seed_features, rot, crop_size):
        grouped_feature = self.grouper(seed_xyz, seed_xyz, rot, crop_size, seed_features)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class WidthNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_depth, 1)
        self.act =  nn.ReLU(inplace=True)
        # classfication-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth * (len(width_bins)+1), 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        features = self.act(self.conv1(seed_features))
        width_pred = self.conv2(features)
        width_pred = width_pred.transpose(1, 2).contiguous() # (B, num_seed, num_depth)
        # classification-based
        # width_pred = width_pred.view(B, num_seed, self.num_view * self.num_angle * self.num_depth, len(width_bins)+1)
        end_points['grasp_width_pred'] = width_pred
        return end_points


class DepthNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        # regression-based
        self.conv2 = nn.Conv1d(self.in_dim, self.num_depth * 2, 1)
        # classfication-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_depth * 2 * (len(width_bins)+1), 1)
        # score only
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_depth, 1)
        self.act =  nn.ReLU(inplace=True)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        features = self.act(self.conv1(seed_features))
        features = self.conv2(features)
        # regression-based
        features = features.view(B, 2, self.num_depth, num_seed)
        features = features.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)
        
        # classification-based
        # features = features.view(B, 2, self.num_depth, len(width_bins)+1, num_seed)
        # features = features.permute(0, 1, 4, 2, 3)
        
        end_points['grasp_score_pred'] = features[:, 0]
        end_points['grasp_width_pred'] = features[:, 1]
        
        # regression-based
        # features = features.view(B, self.num_depth, num_seed)
        # features = features.permute(0, 2, 1) # (B, num_seed, num_depth)
        # end_points['grasp_score_pred'] = features
        
        return end_points


class DepthScoringNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_depth, 1)
        self.act =  nn.ReLU(inplace=True)
        # classfication-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth * (len(width_bins)+1), 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        features = self.act(self.conv1(seed_features))
        score_pred = self.conv2(features)
        score_pred = score_pred.transpose(1, 2).contiguous() # (B, num_seed, num_depth)
        # classification-based
        # width_pred = width_pred.view(B, num_seed, self.num_view * self.num_angle * self.num_depth, len(width_bins)+1)
        end_points['grasp_score_pred'] = score_pred
        return end_points
    

class RotationScoringNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        # # regression-based
        self.conv3 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle, 1)
        # classfication-based (len(score_bins)+1) CORN (len(score_bins))
        # self.conv3 = nn.Conv1d(self.in_dim * 2, self.num_view * self.num_angle * self.num_depth * len(score_bins), 1)
        self.act =  nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        # v0.3.6.8
        res_features = self.act(self.conv1(seed_features))
        res_features = self.act(self.conv2(res_features))
        features = self.conv3(res_features)
        rotation_scores = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)

        # classification-based
        # rotation_scores = rotation_scores.view(B, num_seed, self.num_view * self.num_angle, len(score_bins))
        end_points['grasp_rot_graspness_pred'] = rotation_scores

        if self.is_training:
            # normalize view graspness score to 0~1
            rot_score = rotation_scores.clone().detach()
            rot_score = normalize_tensor(rot_score)
            top_rot_inds = []
            for i in range(B):
                top_rot_inds_batch = torch.multinomial(rot_score[i], 1, replacement=False)
                top_rot_inds.append(top_rot_inds_batch)
            top_rot_inds = torch.stack(top_rot_inds, dim=0).squeeze(-1)  # B, num_seed
        else:
            _, top_rot_inds = torch.max(rotation_scores, dim=-1)  # (B, num_seed)
            temp_grasp_rots = grasp_rot.clone().to(seed_features.device)
            temp_grasp_rots = temp_grasp_rots.view(1, 1, self.num_view*self.num_angle, 3, 3)
            temp_grasp_rots = temp_grasp_rots.expand(B, num_seed, -1, -1, -1).contiguous()
            top_rot_inds_ = top_rot_inds.view(B, num_seed, 1, 1, 1).expand(-1, -1, -1, 3, 3)
            grasp_top_rot = torch.gather(temp_grasp_rots, 2, top_rot_inds_).squeeze(2)
            end_points['grasp_top_rot'] = grasp_top_rot

        end_points['grasp_top_rot_inds'] = top_rot_inds
        return end_points, res_features


class IGNet(nn.Module):
    def __init__(self,  num_view=300, num_angle=12, num_depth=4, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = num_depth
        self.num_angle = num_angle
        self.num_view = num_view

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.rot_head = RotationScoringNet(self.num_view, num_angle=self.num_angle,
                                                num_depth=self.num_depth,
                                                seed_feature_dim=self.seed_feature_dim, 
                                                is_training=self.is_training)
        self.crop = CloudCrop(nsample=32, seed_feature_dim=self.seed_feature_dim)
        # self.width_head = WidthNet(self.num_view, num_angle=self.num_angle, 
        #                            num_depth=self.num_depth,
        #                            seed_feature_dim=self.seed_feature_dim)
        # self.depth_head = DepthScoringNet(self.num_view, num_angle=self.num_angle, 
        #                            num_depth=self.num_depth,
        #                            seed_feature_dim=self.seed_feature_dim)
        self.depth_head = DepthNet(self.num_view, num_angle=self.num_angle, 
                                   num_depth=self.num_depth,
                                   seed_feature_dim=self.seed_feature_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
                
    def forward(self, end_points):
        # use all sampled point cloud, B*Ns*3
        seed_xyz = end_points['point_clouds']
        B, point_num, _ = seed_xyz.shape  # batch _size

        # point-wise features
        coordinates_batch, features_batch = ME.utils.sparse_collate(coords=[c for c in end_points['coors']], 
                                                                    feats=[f for f in end_points['feats']], 
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=seed_xyz.device)
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[quantize2original].view(B, point_num, -1).transpose(1, 2)
        
        # coordinates_batch = end_points['coors']
        # features_batch = end_points['feats']
        # mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        # seed_features = self.backbone(mink_input).F
        # seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        end_points['seed_features'] = seed_features  # (B, seed_feature_dim, num_seed)
        end_points, rot_features = self.rot_head(seed_features, end_points)
        seed_features = seed_features + rot_features

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_rots, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_rots = end_points['grasp_top_rot']

        crop_length = (0.04 + base_depth) * torch.ones((B, point_num, 1), device=seed_xyz.device)
        crop_width = GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=seed_xyz.device)
        crop_height = 0.02 * torch.ones_like(crop_length, device=seed_xyz.device)
        crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1).contiguous()

        group_features = self.crop(seed_xyz.contiguous(), seed_features.contiguous(),
                                   grasp_top_rots, crop_size)
        # end_points = self.width_head(group_features, end_points)
        end_points = self.depth_head(group_features, end_points)
        return end_points



def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['point_clouds']  # (B, M_point, 3)
    # seed_normals = end_points['cloud_normals'] # (B, M_point, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_rots = []
    # batch_grasp_rot_max = []
    # batch_grasp_depth_max = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_grasp_masks = []
    
    pred_grasp_rots = []
    pred_grasp_depths = []
    
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        object_pose = end_points['object_pose'][i]  # [(3, 4),]

        # get merged grasp points for label computation
        grasp_points = end_points['grasp_points'][i]  # (Np, 3)
        grasp_scores = end_points['grasp_labels'][i]  # (Np, V, A, D)
        grasp_widths = end_points['grasp_offsets'][i]  # (Np, V, A, D)
        _, V, A, D = grasp_scores.size()
        # num_grasp_points = grasp_points.size(0)
        
        # generate and transform template grasp views
        grasp_views = generate_grasp_views(V).to(object_pose.device)  # (V, 3)
        grasp_points_trans = transform_point_cloud(grasp_points, object_pose, '3x4')
        grasp_views_trans = transform_point_cloud(grasp_views, object_pose[:3, :3], '3x3')

        # generate and transform template grasp view rotation
        # angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
        # grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
        # grasp_views_rot_trans = torch.matmul(object_pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

        grasp_rot_trans = torch.matmul(object_pose[:3, :3], grasp_rot.to(object_pose.device))  # (V, 3, 3)
        grasp_rot_trans = grasp_rot_trans.view((NUM_VIEW, NUM_ANGLE, 3, 3))
        
        # assign views
        grasp_views_ = grasp_views.unsqueeze(0)
        grasp_views_trans_ = grasp_views_trans.unsqueeze(0)
        _, view_inds, _ = knn_points(grasp_views_, grasp_views_trans_, K=1)
        view_inds = view_inds.squeeze(-1).squeeze(0)
        
        grasp_rot_trans = torch.index_select(grasp_rot_trans, 0, view_inds)  # (V, A, 3, 3)
        # grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
        # grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)  # (Np, V, 3, 3)
        grasp_scores = torch.index_select(grasp_scores, 1, view_inds)  # (Np, V, A, D)
        grasp_widths = torch.index_select(grasp_widths, 1, view_inds)  # (Np, V, A, D)

        grasp_rot_trans, grasp_scores, grasp_widths = align_angle_index(grasp_rot, grasp_rot_trans, grasp_scores, grasp_widths)
        
        # compute nearest neighbors
        seed_xyz_ = seed_xyz.unsqueeze(0)  # (1, Ns, 3)
        grasp_points_trans_ = grasp_points_trans.unsqueeze(0)  # (1, Np', 3)
        _, nn_inds, _ = knn_points(seed_xyz_, grasp_points_trans_, K=1) # (Ns)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)

        # assign anchor points to real points
        grasp_points_trans = torch.index_select(grasp_points_trans, 0, nn_inds)  # (Ns, 3)
        # grasp_rot_trans = torch.index_select(grasp_rot_trans, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_scores = torch.index_select(grasp_scores, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths = torch.index_select(grasp_widths, 0, nn_inds)  # (Ns, V, A, D)

        # grasp_scores_mask = (grasp_scores > 0) & (grasp_widths <= GRASP_MAX_WIDTH)  # (Ns, V, A, D)
        # grasp_scores = 1.1 - grasp_scores
        # grasp_scores[~grasp_scores_mask] = 0

        # pred_rot_inds = end_points['grasp_top_rot_inds'][i]
        # pred_view_inds, pred_angle_inds = unravel_index(pred_rot_inds, (NUM_VIEW, NUM_ANGLE))
        # pred_rots = grasp_rot_trans[pred_view_inds, pred_angle_inds, :, :]
        # pred_rots_6d = matrix_to_rotation_6d(pred_rots) # (Ns, 6)
    
        # grasp_depth = depths.to(object_pose.device)
        # grasp_depth = grasp_depth.unsqueeze(0).tile((num_samples, 1))
        # pred_depth = torch.gather(grasp_depth, 1, pred_depth_inds.view(-1, 1)) # (Ns, 1)
        # match_grasp_width = torch.gather(grasp_widths.view(num_samples, -1), 1, pred_rot_depth_inds.view(-1, 1)) # (Ns, 1)
        # match_grasp_score_mask = torch.gather(grasp_scores_mask.view(num_samples, -1), 1, pred_rot_depth_inds.view(-1, 1)) # (Ns, 1)
                
        # v0.4
        # grasp_quality_scores = grasp_scores.clone()
        # po_mask = grasp_quality_scores > 0
        # grasp_quality_scores[po_mask] = 1.1 - grasp_quality_scores[po_mask]
        
        # _, grasp_score_inds = torch.max(grasp_quality_scores.view(num_samples, -1), dim=1)
        # view_inds, angle_inds, depth_inds = unravel_index(grasp_score_inds, (NUM_VIEW, NUM_ANGLE, NUM_DEPTH))
        # grasp_rot_max = grasp_rot_trans[view_inds, angle_inds, :, :]
        # depth_inds = depth_inds.unsqueeze(0)

        # grasp_scores_max = torch.gather(grasp_scores.view(num_samples, -1), 1, grasp_score_inds.view(-1, 1))
        # grasp_width_max = torch.gather(grasp_widths.view(num_samples, -1), 1, grasp_score_inds.view(-1, 1))
        
        grasp_rot_trans = grasp_rot_trans.unsqueeze(0).tile((num_samples, 1, 1, 1, 1))
        
        # add to batch
        batch_grasp_points.append(grasp_points_trans)
        batch_grasp_rots.append(grasp_rot_trans)
        # batch_grasp_rot_max.append(grasp_rot_max)
        # batch_grasp_depth_max.append(depth_inds)
        batch_grasp_scores.append(grasp_scores)
        batch_grasp_widths.append(grasp_widths)
        # pred_grasp_rots.append(pred_rots)
        # pred_grasp_depths.append(pred_depth)
        # batch_grasp_masks.append(match_grasp_score_mask)
        
    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_rots = torch.stack(batch_grasp_rots, 0)  # (B, Ns, V, 3, 3)
    # batch_grasp_rot_max = torch.stack(batch_grasp_rot_max, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)
    # batch_grasp_masks = torch.stack(batch_grasp_masks, 0)  # (B, Ns, V, A, D)
    
    # pred_grasp_rots = torch.stack(pred_grasp_rots, 0) # (B, Ns, 6)
    # pred_grasp_depths = torch.stack(pred_grasp_depths, 0) # (B, Ns, 1)
        
    batch_grasp_rot_graspness_mask = (batch_grasp_scores <= 0.6) & (batch_grasp_scores > 0) # (B, Ns, V, A, D)
    batch_grasp_rot_graspness = batch_grasp_rot_graspness_mask.float()
    batch_grasp_rot_graspness = torch.mean(batch_grasp_rot_graspness, dim=-1)  # (B, Ns, V, A)
    batch_grasp_rot_graspness = batch_grasp_rot_graspness.view((batch_size, num_samples, -1)) # (B, Ns, V*A)
    batch_grasp_rot_graspness = normalize_tensor(batch_grasp_rot_graspness) # (B, Ns, V*A)
    
    # compute view graspness
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores[~label_mask] = 0

    # process scores
    # batch_grasp_scores = batch_grasp_scores.view(batch_size, num_samples, -1)
    # batch_grasp_widths = batch_grasp_widths.view(batch_size, num_samples, -1)
    # batch_grasp_masks = batch_grasp_masks.view(batch_size, num_samples, -1)

    # batch_grasp_widths_ids = torch.bucketize(batch_grasp_widths, width_bins.to(batch_grasp_widths.device))
    # batch_grasp_scores_ids = torch.bucketize(batch_grasp_scores, score_bins.to(batch_grasp_scores.device))
    
    # end_points['pred_grasp_rots'] = pred_grasp_rots
    # end_points['pred_grasp_depths'] = pred_grasp_depths
    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rot'] = batch_grasp_rots
    
    # end_points['batch_grasp_rot_max'] = batch_grasp_rot_max
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_rot_graspness'] = batch_grasp_rot_graspness
    
    # end_points['batch_grasp_mask'] = batch_grasp_masks
    # end_points['batch_grasp_width_ids'] = batch_grasp_widths_ids
    # end_points['batch_grasp_score_ids'] = batch_grasp_scores_ids
    # end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness

    return end_points


def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    top_rot_inds = end_points['grasp_top_rot_inds']  # (B, Ns)
    template_views_rot = end_points['batch_grasp_rot']  # (B, Ns, V, A, 3, 3)
    grasp_scores = end_points['batch_grasp_score']  # (B, Ns, V, A, D)
    grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, D)

    B, Ns, V, A, D = grasp_scores.size()
    top_rot_inds_ = top_rot_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_rot_mat = torch.gather(template_views_rot.view(B, Ns, V*A, 3, 3), 2, top_rot_inds_).squeeze(2)
    
    top_rot_inds_ = top_rot_inds.view(B, Ns, 1, 1).expand(-1, -1, -1, D)
    top_rot_grasp_scores = torch.gather(grasp_scores.view(B, Ns, V*A, D), 2, top_rot_inds_).squeeze(2)
    top_rot_grasp_widths = torch.gather(grasp_widths.view(B, Ns, V*A, D), 2, top_rot_inds_).squeeze(2)

    # print(top_rot_grasp_scores.min(), top_rot_grasp_scores.max(), top_rot_grasp_scores.mean())
    # print(top_rot_grasp_widths.min(), top_rot_grasp_widths.max(), top_rot_grasp_widths.mean())
    
    u_max = top_rot_grasp_scores.max()
    po_mask = top_rot_grasp_scores > 0
    po_mask_num = torch.sum(po_mask)
    if po_mask_num > 0:
        # grasp_scores_record = grasp_scores[po_mask]
        # print('before min:{}, max:{}'.format(grasp_scores_record.min(), grasp_scores_record.max()))
        
        # grasp_scores[po_mask] = torch.log(u_max / grasp_scores[po_mask])
        
        # grasp_scores_record = grasp_scores[po_mask]
        # print('before min:{}, max:{}'.format(grasp_scores_record.min(), grasp_scores_record.max()))
        
        u_min = top_rot_grasp_scores[po_mask].min()
        top_rot_grasp_scores[po_mask] = torch.log(u_max / top_rot_grasp_scores[po_mask]) / \
            (torch.log(u_max / u_min) + 1e-8)

    batch_grasp_scores_ids = torch.bucketize(top_rot_grasp_scores, score_bins.to(grasp_widths.device))
    batch_grasp_widths_ids = torch.bucketize(top_rot_grasp_scores, width_bins.to(grasp_scores.device))

    end_points['batch_grasp_score'] = top_rot_grasp_scores  # (B, Ns, D)
    end_points['batch_grasp_width'] = top_rot_grasp_widths  # (B, Ns, D)
    
    end_points['batch_grasp_score_ids'] = batch_grasp_scores_ids  # (B, Ns, D, score_bin_num)
    end_points['batch_grasp_width_ids'] = batch_grasp_widths_ids  # (B, Ns, D, width_bin_num)
    return top_template_rot_mat, end_points


def compute_grasp_widths(grasp_points, grasp_poses, width_intervals, nsample):

    point_num, _ = grasp_points.size()
    crop_length = (0.04 + base_depth) * torch.ones((1, point_num, 1), device=grasp_points.device)
    crop_width = GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=grasp_points.device)
    crop_height = 0.02 * torch.ones_like(crop_length, device=grasp_points.device)
    crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1).contiguous()
    rectangular_idx = RectangularQuery.apply(crop_size, nsample, grasp_points.unsqueeze(0), grasp_points.unsqueeze(0), grasp_poses.unsqueeze(0)).long()

    query_points = torch.gather(grasp_points.unsqueeze(0).unsqueeze(2).expand(-1, -1, nsample, -1), 1,
                                rectangular_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    query_points = query_points.squeeze(0)
    grasp_poses = grasp_poses.view(point_num, 3, 3)
    
    finger_width = 0.01
    finger_height = 0.02
    finger_depth_base = 0.02

    grasp_depths = 0.04 * torch.ones((len(grasp_points),), device=grasp_points.device)
    grasp_widths = width_intervals.unsqueeze(0).tile((len(grasp_points), 1)).to(grasp_points.device)
    width_num = len(width_intervals)
    
    # 调整grasp_points形状以广播
    grasp_points = grasp_points.unsqueeze(1)  # 形状从(1024, 3)变为(1024, 1, 3)

    # 计算target
    target = query_points - grasp_points  # 利用广播减法
    target = torch.matmul(target, grasp_poses)  # 矩阵乘法

    # 扩展grasp_widths为 (1024, nsample, len(width_bins) )
    grasp_widths_expand = grasp_widths.unsqueeze(1).expand(-1, nsample, -1)

    mask1 = ((target[:, :, 2] > -finger_height / 2) & (target[:, :, 2] < finger_height / 2))
    mask2 = ((target[:, :, 0] > -finger_depth_base) & (target[:, :, 0] < grasp_depths.unsqueeze(1).expand(-1, nsample)))

    # mask3，mask4，mask5，已修正为考虑每个bin的宽度
    mask3 = target[:, :, 1].unsqueeze(-1) > -(grasp_widths_expand / 2 + finger_width)
    mask4 = target[:, :, 1].unsqueeze(-1) < -grasp_widths_expand / 2
    mask5 = target[:, :, 1].unsqueeze(-1) < (grasp_widths_expand / 2 + finger_width)
    mask6 = target[:, :, 1].unsqueeze(-1) > grasp_widths_expand / 2

    # mask7: 检查是否在指定深度的底部范围内
    mask7 = ((target[:, :, 0] > -(finger_depth_base + finger_width)) & (target[:, :, 0] < -finger_depth_base))

    mask1 = mask1.unsqueeze(-1).expand(-1, -1, width_num)
    mask2 = mask2.unsqueeze(-1).expand(-1, -1, width_num)
    mask7 = mask7.unsqueeze(-1).expand(-1, -1, width_num)

    # 计算每个bin的左、右和底部碰撞掩码
    left_mask = mask1 & mask2 & mask3 & mask4
    right_mask = mask1 & mask2 & mask5 & mask6
    bottom_mask = mask1 & mask3 & mask5 & mask7

    # 综合所有bin，检查是否有碰撞
    collision_mask = (left_mask | right_mask | bottom_mask).any(dim=1)

    first_no_collision = torch.argmax((~collision_mask).int(), dim=1)
    pred_grasp_widths = torch.gather(grasp_widths, 1, first_no_collision.view(-1, 1))
    return pred_grasp_widths


def pred_decode(end_points, normalize=False):
    grasp_center = end_points['point_clouds']
    batch_size, num_samples, _ = grasp_center.shape
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['point_clouds'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float() # (num_samples, D)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.  # 10 for multiply 10 in loss function

        if normalize:
            grasp_score = normalize_tensor(grasp_score)

        # grasp_score = end_points['grasp_score_pred'][i] # (num_samples, D, score_bin_num)
        # grasp_score = grasp_score.argmax(-1)
        # grasp_score = value_unbucketize(grasp_score, score_bins.clone().to(grasp_center.device))
        
        grasp_score, grasp_score_inds = torch.max(grasp_score, dim=-1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)

        grasp_depth = depths.to(grasp_center.device)
        grasp_depth = grasp_depth.unsqueeze(0).tile((num_samples, 1))
        grasp_depth = torch.gather(grasp_depth, 1, grasp_score_inds.view(-1, 1)) # (Ns, 1)

        # grasp_width = end_points['grasp_width_pred'][i] # (num_samples, D, width_bin_num)
        # grasp_width = grasp_width.argmax(-1)
        # grasp_width = value_unbucketize(grasp_width, width_bins.clone().to(grasp_center.device))
        
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)
        
        views_rot = grasp_rot.clone().to(grasp_center.device)
        views_rot = views_rot.unsqueeze(0).tile((num_samples, 1, 1, 1))
        top_rot_inds = end_points['grasp_top_rot_inds'][i]
        top_rot_inds = top_rot_inds.view((num_samples, 1, 1, 1)).expand(-1, -1, 3, 3)
        topk_grasp_rots = torch.gather(views_rot, 1, top_rot_inds).squeeze(1)
        topk_grasp_rots = topk_grasp_rots.view(-1, 9)

        # grasp_width  = compute_grasp_widths(grasp_center, topk_grasp_rots, width_intervals.clone(), nsample=512)
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height,
                                      grasp_depth, topk_grasp_rots, grasp_center, obj_ids],
                                     axis=-1).detach().cpu().numpy())

    return grasp_preds