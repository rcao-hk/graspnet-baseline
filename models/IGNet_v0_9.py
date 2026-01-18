import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
import MinkowskiEngine as ME
from models.minkowski import MinkUNet14D
from typing import Dict, Tuple, Optional

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from pytorch3d.ops.knn import knn_points
import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import RectangularQueryAndGroup, furthest_point_sample, gather_operation
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix, batch_get_key_points, transform_point_cloud, GRASPNESS_THRESHOLD, GRASP_MAX_WIDTH, NUM_ANGLE, NUM_VIEW, NUM_DEPTH, M_POINT
from models.coral_loss import corn_label_from_logits
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
# from rectangular_query_ext import rectangular_query

base_depth = 0.04
angles = torch.tensor([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)])
views = generate_grasp_views(NUM_VIEW)  # num of views, (300,3), np.float32
# views_repeat = views.repeat_interleave(NUM_ANGLE, 0)  # (300*12,3)
# angles_repeat = angles.view(1, NUM_ANGLE).repeat_interleave(NUM_VIEW, 0).view(-1)  # (300*12,)
angles_repeat = angles.tile(NUM_VIEW)
views_repeat = views.repeat_interleave(NUM_ANGLE, dim=0)
grasp_rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat)  # (300*12, 3, 3)
depths = torch.linspace(0.01, 0.04, 4)
# score_bins = torch.tensor([0.2, 0.4, 0.6, 0.8])
# width_bins = torch.tensor([0.02, 0.04, 0.06, 0.08])

# v0.6.3.1
score_bins = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
width_bins = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])


def _as_tensor(x, device):
    if torch.is_tensor(x):
        return x
    return torch.tensor(x, dtype=torch.float32, device=device)


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


def normalize_tensor(x: torch.Tensor, eps: float = 1e-6):
    # x: (..., K)
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


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


from models.pspnet import PSPNet
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50')
}


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, out_dim):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.out_dim = out_dim
        mlps = [3 + self.in_dim, 256, self.out_dim]   # use xyz, so plus 3
        self.grouper = RectangularQueryAndGroup(nsample=nsample, use_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        # self.attnpool = AttentionPool1d(spacial_dim=nsample, embed_dim=256, num_heads=4)

    def forward(self, seed_xyz, seed_features, rot, crop_size):
        grouped_feature = self.grouper(seed_xyz, seed_xyz, rot, crop_size, seed_features)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, nsample) (32, 256, 1024, 32)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
    
        # Reshape to (B*M, C, K) to apply AttentionPool1d across K=32
        # B, C, M, K = new_features.shape
        # new_features = new_features.view(B * M, C, K)  # (32*1024, 256, 32)
        # new_features = self.attnpool(new_features)  # (32*1024, 256, 1)
        # new_features = new_features.view(B, C, M, 1)  # (32, 256, 1024, 1)
        
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class AttentionPool1d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x shape: (B*M, C, K)
        B_M, C, K = x.shape
        x = x.permute(2, 0, 1)  # (K, B*M, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (1 + K, B*M, C)
        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (1 + K, B*M, C)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0).view(B_M, -1, 1)  # (B*M, output_dim, 1)
    
# class OperationNet(nn.Module):
#     """ Grasp configure estimation.

#         Input:
#             num_depth: [int]
#                 number of gripper depth classes
#     """

#     def __init__(self, num_view, num_angle, num_depth, seed_feature_dim):
#         super().__init__()
#         self.in_dim = seed_feature_dim
#         self.num_view = num_view
#         self.num_angle = num_angle
#         self.num_depth = num_depth

#         self.conv1 = nn.Conv1d(self.in_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, self.num_depth * 2, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)

#     def forward(self, seed_features, end_points):
#         B, _, num_seed = seed_features.size()
#         features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
#         features = F.relu(self.bn2(self.conv2(features)), inplace=True)
#         features = self.conv3(features)
#         features = features.view(B, 2, self.num_depth, num_seed)
#         features = features.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)
#         end_points['grasp_score_pred'] = features[:, 0]
#         end_points['grasp_width_pred'] = features[:, 1]
#         return end_points
        
        

class DepthNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
        )
        self.depthnet = nn.Sequential(nn.Dropout(0.15),
                                      nn.Conv1d(self.in_dim, self.num_depth * 2, 1))


    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        features = self.mlp(seed_features)
        predicts = self.depthnet(features)
        
        # regression-based
        predicts = predicts.view(B, 2, self.num_depth, num_seed)
        predicts = predicts.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)

        end_points['grasp_score_pred'] = predicts[:, 0]
        end_points['grasp_width_pred'] = predicts[:, 1]
    
        return end_points


class RotationScoringNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        # self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        # self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
        )
        # self.mlp = nn.Sequential(
        #     nn.Conv1d(self.in_dim, self.in_dim, 1),
        #     nn.BatchNorm1d(self.in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(self.in_dim, self.in_dim, 1),
        #     nn.BatchNorm1d(self.in_dim),
        #     nn.ReLU(inplace=True),
        # )
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     # nn.ReLU(inplace=True),
        # )
        # self.mlp = ResMLP(self.in_dim, self.in_dim)
                
        # # # regression-based
        # # self.outconv = nn.Conv1d(self.in_dim, self.num_view * self.num_angle, 1)
        # self.outconv = nn.Sequential(
        #     nn.Conv1d(self.in_dim, self.in_dim * 2, 1),
        #     nn.BatchNorm1d(self.in_dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(self.in_dim * 2, self.in_dim * 4, 1),
        #     nn.BatchNorm1d(self.in_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(self.in_dim * 4, self.num_view * self.num_angle, 1))
        self.outconv = nn.Sequential(nn.Dropout(0.15),
                                     nn.Conv1d(self.in_dim, self.num_view * self.num_angle, 1))
        
        # self.outconv = nn.Sequential(
        #     nn.Linear(self.in_dim, self.in_dim * 2),
        #     nn.LayerNorm(self.in_dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.in_dim * 2, self.in_dim * 4),
        #     nn.LayerNorm(self.in_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.15),
        #     nn.Linear(self.in_dim * 4, self.num_view * self.num_angle))
        # self.outconv = nn.Sequential(LinearLNReLU(self.in_dim, self.in_dim),
        #                              nn.Dropout(0.2),
        #                              nn.Linear(self.in_dim, self.num_view * self.num_angle))
        
        # classfication-based (len(score_bins)+1) CORN (len(score_bins))
        # self.conv3 = nn.Conv1d(self.in_dim * 2, self.num_view * self.num_angle * self.num_depth * len(score_bins), 1)
        # self.act =  nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        # v0.3.6.8
        # res_features = self.act(self.conv1(seed_features))
        # res_features = self.act(self.conv2(res_features))
        # rotation_scores = self.outconv(res_features)
        # rotation_scores = rotation_scores.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)
        
        res_features = self.mlp(seed_features)
        seed_features = seed_features + res_features
        rotation_scores = self.outconv(seed_features)
        rotation_scores = rotation_scores.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)
        
        # seed_features = seed_features.transpose(1, 2)
        # res_features = self.mlp(seed_features)
        # seed_features = F.relu(seed_features + res_features, inplace=True)
        # rotation_scores = self.outconv(seed_features)
        # # rotation_scores = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)
        # seed_features = seed_features.transpose(1, 2)
                
        # seed_features = seed_features.transpose(1, 2).contiguous()
        # res_features = self.mlp(seed_features, seed_features)
        # rotation_scores = self.outconv(res_features) # (B, num_seed, num_view * num_angle)
        # res_features = res_features.transpose(1, 2).contiguous()
        
        # classification-based
        # rotation_scores = rotation_scores.view(B, num_seed, self.num_view * self.num_angle, len(score_bins))
        end_points['grasp_rot_graspness_pred'] = rotation_scores

        if self.is_training:
            # normalize view graspness score to 0~1
            rot_score = rotation_scores.clone().detach()
            rot_score = normalize_tensor(rot_score)
            top_rot_inds = []
            for i in range(B):
                try:
                    top_rot_inds_batch = torch.multinomial(rot_score[i], 1, replacement=False)
                except:
                    print('outliers in rotation_scores')
                    _, top_rot_inds_batch = torch.max(rot_score[i], dim=-1)  # (B, num_seed)
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
        return end_points, seed_features


from transformers.activations import ACT2FN
class GateFFN(nn.Module):
    def __init__(self, model_dim, dropout_rate, hidden_unit=2048):
        super(GateFFN, self).__init__()
        """
        对应论文中的W，V，以及W2，激活函数对应GELU_new
        """
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_gelu = self.gelu_act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.W2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    

# class DepthNetGate(nn.Module):
#     def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, dropout_rate):
#         super().__init__()
#         self.in_dim = seed_feature_dim
#         self.num_view = num_view
#         self.num_angle = num_angle
#         self.num_depth = num_depth
#         self.dropout_rate = dropout_rate
#         self.gate_ffn = GateFFN(self.in_dim, self.dropout_rate * 2, self.in_dim * 2)
#         self.depthnet = nn.Sequential(nn.Dropout(self.dropout_rate),
#                                       nn.Conv1d(self.in_dim, self.num_depth * 2, 1))

#     def forward(self, seed_features, end_points):
#         B, _, num_seed = seed_features.size()

#         features = self.gate_ffn(seed_features)
#         predicts = self.depthnet(features)

#         predicts = predicts.view(B, 2, self.num_depth, num_seed)
#         predicts = predicts.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)

#         end_points['grasp_score_pred'] = predicts[:, 0]
#         end_points['grasp_width_pred'] = predicts[:, 1]

#         return end_points


class RotationScoringNetGate(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, dropout_rate, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.dropout_rate = dropout_rate       
        self.gate_ffn = GateFFN(self.in_dim, self.dropout_rate * 2, self.in_dim * 2)
        self.conv_out = nn.Sequential(nn.Dropout(self.dropout_rate),
                                      nn.Conv1d(self.in_dim, self.num_view * self.num_angle, 1))

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        res_features = self.gate_ffn(seed_features)
        seed_features = seed_features + res_features
        features = self.conv_out(res_features)
        rotation_scores = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)

        end_points['grasp_rot_graspness_pred'] = rotation_scores

        if self.is_training:
            # normalize view graspness score to 0~1
            rot_score = rotation_scores.clone().detach()
            rot_score = normalize_tensor(rot_score)
            top_rot_inds = []
            for i in range(B):
                try:
                    top_rot_inds_batch = torch.multinomial(rot_score[i], 1, replacement=False)
                except:
                    print('outliers in rotation_scores')
                    _, top_rot_inds_batch = torch.max(rot_score[i], dim=-1)  # (B, num_seed)
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
        return end_points, seed_features
    

# add two features
class AddFusion(nn.Module):
    def __init__(self, point_dim, img_dim):
        super(AddFusion, self).__init__()
        self.point_dim = point_dim
        self.img_dim = img_dim
        self.img_mlp = nn.Sequential(
            nn.Conv1d(img_dim, 128, 1),
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, point_dim, 1)
        )
        
    def forward(self, point_feat, img_feat):
        img_feat = self.img_mlp(img_feat)
        fused_feat = img_feat + point_feat
        return fused_feat
    

# Proposed by Multi-Source Fusion for Voxel-Based 7-DoF Grasping Pose Estimation
class GatedFusion(nn.Module):
    def __init__(self, point_dim, img_dim):
        super(GatedFusion, self).__init__()
        self.point_dim = point_dim
        self.img_dim = img_dim
        self.expand_img = nn.Sequential(
            nn.Conv1d(img_dim, 128, 1),
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, point_dim, 1)
        )
        
        self.img_mlp = nn.Conv1d(self.point_dim, self.point_dim, 1)
        self.point_mlp = nn.Conv1d(self.point_dim, self.point_dim, 1)
        
        # Decoders
        self.gate_mlp = nn.Sequential(
            nn.Conv1d(self.point_dim, self.point_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, point_feat, img_feat):
        img_feat = self.expand_img(img_feat)
        img_feat = self.img_mlp(img_feat)
        point_fuse_feat = self.point_mlp(point_feat)
        gate_feat = self.gate_mlp(F.relu(img_feat+point_fuse_feat))
        img_feat = gate_feat * img_feat
        fused_feat = img_feat + point_feat
        return fused_feat


# class CrossAttentionConcat(nn.Module):
#     def __init__(self, point_dim, img_dim, dropout, num_heads, normalize=False):
#         super(CrossAttentionConcat, self).__init__()
#         self.point_dim = point_dim
#         self.img_dim = img_dim
#         self.num_heads = num_heads
#         self.normalize = normalize
#         self.dropout = dropout
        
#         if self.normalize:
#             self.point_norm = nn.LayerNorm(self.point_dim)
#             self.img_norm = nn.LayerNorm(self.img_dim)

#         # 多头注意力层
#         self.p2m_attn = nn.MultiheadAttention(embed_dim=self.point_dim, num_heads=self.num_heads, 
#                                               dropout=self.dropout, kdim=self.img_dim, vdim=self.img_dim)
#         self.m2p_attn = nn.MultiheadAttention(embed_dim=self.img_dim, num_heads=self.num_heads, 
#                                               dropout=self.dropout, kdim=self.point_dim, vdim=self.point_dim)
        
#     def forward(self, point_feat, img_feat):
#         if self.normalize:
#             point_feat = self.point_norm(point_feat)
#             img_feat = self.img_norm(img_feat)

#         # 调整维度符合多头注意力输入要求 (Seq_len, Batch, Embedding_dim)
#         point_feat = point_feat.transpose(0, 1)  # (num_pc, B, point_dim)
#         img_feat = img_feat.transpose(0, 1)  # (num_pc, B, img_dim)

#         fused_point_feat, _ = self.p2m_attn(point_feat, img_feat, img_feat)
#         fused_img_feat, _  = self.m2p_attn(img_feat, point_feat, point_feat)
#         fused_feat = torch.concat([fused_point_feat, fused_img_feat], dim=2)

#         fused_feat = fused_feat.permute((1, 2, 0))  # (B, output_dim, num_pc)
#         return fused_feat


# from flash_attn.modules.mha import FlashCrossAttention, CrossAttention
# from einops import rearrange

# class CrossModalAttention(nn.Module):
#     def __init__(self, point_dim, img_dim, dropout, normalize=False, in_proj_bias=True):
#         super(CrossModalAttention, self).__init__()
#         self.feat_dim = self.point_dim = point_dim
#         self.img_dim = img_dim
#         self.normalize = normalize
#         self.dropout = dropout
        
#         if self.normalize:
#             self.point_norm = nn.LayerNorm(self.point_dim)
#             self.img_norm = nn.LayerNorm(self.img_dim)

#         self.img_mlp = nn.Sequential(
#             nn.Conv1d(img_dim, 128, 1),
#             nn.BatchNorm1d(128), 
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, self.feat_dim, 1)
#         )
        
#         # self.point_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
#         # self.img_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
#         self.point_kv_proj = nn.Conv1d(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias, kernel_size=1)
#         self.img_q_proj = nn.Conv1d(self.feat_dim, self.feat_dim, bias=in_proj_bias, kernel_size=1)
        
#         # self.img_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
#         # self.point_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        
#         # self.point_cross_attn = FlashCrossAttention(attention_dropout=dropout)
#         # self.image_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        
#         self.point_cross_attn = CrossAttention(attention_dropout=dropout)
#         # self.image_cross_attn = CrossAttention(attention_dropout=dropout)
                                                             
#     def forward(self, point_feat, img_feat):
#         if self.normalize:
#             point_feat = self.point_norm(point_feat)
#             img_feat = self.img_norm(img_feat)

#         Bs, N, _ = point_feat.shape
        
#         point_feat = point_feat.transpose(1, 2)  # (B, point_dim, num_pc)
#         img_feat = img_feat.transpose(1, 2)  # (B, img_dim, num_pc)
        
#         img_feat = self.img_mlp(img_feat)
#         img_q = self.img_q_proj(img_feat)
#         # img_kv = self.img_kv_proj(img_feat)
#         # point_q = self.point_q_proj(point_feat)
#         point_kv = self.point_kv_proj(point_feat)
        
#         # img_q = self.img_q_proj(img_feat).half()
#         # img_kv = self.img_kv_proj(img_feat).half()
#         # point_q = self.point_q_proj(point_feat).half()
#         # point_kv = self.point_kv_proj(point_feat).half()
        
#         img_q = rearrange(img_q.transpose(1, 2), "... (h d) -> ... h d", d=self.feat_dim)
#         point_kv = rearrange(point_kv.transpose(1, 2), "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
#         point_fuse = self.point_cross_attn(img_q, point_kv)
#         # point_fuse = point_feat + point_fuse.view(Bs, N, -1)
#         # point_fuse = torch.concat([point_feat, point_fuse.view(Bs, N, -1)], dim=-1)

#         fused_feat = torch.concat([point_feat.transpose(1, 2), point_fuse.view(Bs, N, -1)], dim=-1)
        
#         # point_q = rearrange(point_q, "... (h d) -> ... h d", d=self.feat_dim)
#         # img_kv = rearrange(img_kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
#         # image_fuse = self.image_cross_attn(point_q, img_kv)
#         # image_fuse = img_feat + image_fuse.view(Bs, N, -1)
#         # image_fuse = torch.concat([img_feat, image_fuse.view(Bs, N, -1)], dim=-1)
        
#         # fused_feat = torch.concat([point_fuse, image_fuse], dim=-1)
        
#         fused_feat = fused_feat.permute((0, 2, 1))  # (B, output_dim, num_pc)
#         return fused_feat

# from flash_attn.modules.mha import CrossAttention
# from einops import rearrange
# class LearnableAlign(nn.Module):
#     def __init__(self, point_dim, img_dim, dropout, normalize=False, in_proj_bias=True):
#         super(LearnableAlign, self).__init__()
#         self.feat_dim = self.point_dim = point_dim
#         self.img_dim = img_dim
#         self.normalize = normalize
#         self.dropout = dropout
        
#         if self.normalize:
#             self.point_norm = nn.LayerNorm(self.point_dim)
#             self.img_norm = nn.LayerNorm(self.img_dim)

#         self.img_mlp = nn.Sequential(
#             nn.Linear(img_dim, 128),
#             nn.LayerNorm(128), 
#             nn.ReLU(inplace=True),
#             nn.Linear(128, self.feat_dim),
#             nn.LayerNorm(self.feat_dim), 
#             nn.ReLU(inplace=True),
#         )
#         # self.point_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
#         # self.img_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        
#         self.img_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
#         self.point_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        
#         # self.point_cross_attn = FlashCrossAttention(attention_dropout=dropout)
#         # self.image_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        
#         # self.point_cross_attn = CrossAttention(attention_dropout=dropout)
#         self.image_cross_attn = CrossAttention(attention_dropout=dropout)
#         self.img_feat_out = nn.Linear(self.feat_dim, self.feat_dim)
                                                        
#     def forward(self, point_feat, img_feat):

#         Bs, N, _ = point_feat.shape
        
#         point_feat = point_feat.transpose(1, 2)  # (B, point_dim, num_pc)
#         img_feat = img_feat.transpose(1, 2)  # (B, img_dim, num_pc)
        
#         if self.normalize:
#             point_feat = self.point_norm(point_feat)
#             img_feat = self.img_norm(img_feat)

#         img_feat = self.img_mlp(img_feat)
#         # img_q = self.img_q_proj(img_feat)
#         img_kv = self.img_kv_proj(img_feat)
#         point_q = self.point_q_proj(point_feat)
#         # point_kv = self.point_kv_proj(point_feat)
        
#         # img_q = self.img_q_proj(img_feat).half()
#         # img_kv = self.img_kv_proj(img_feat).half()
#         # point_q = self.point_q_proj(point_feat).half()
#         # point_kv = self.point_kv_proj(point_feat).half()
        
#         # img_q = rearrange(img_q.transpose(1, 2), "... (h d) -> ... h d", d=self.feat_dim)
#         # point_kv = rearrange(point_kv.transpose(1, 2), "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
#         # point_fuse = self.point_cross_attn(img_q, point_kv)
#         # # point_fuse = point_feat + point_fuse.view(Bs, N, -1)
#         # # point_fuse = torch.concat([point_feat, point_fuse.view(Bs, N, -1)], dim=-1)

#         # fused_feat = torch.concat([point_feat.transpose(1, 2), point_fuse.view(Bs, N, -1)], dim=-1)
        
#         point_q = rearrange(point_q, "... (h d) -> ... h d", d=self.feat_dim)
#         img_kv = rearrange(img_kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
#         image_fuse = self.image_cross_attn(point_q, img_kv)
#         img_feat = self.img_feat_out(image_fuse)
#         # image_fuse = img_feat + image_fuse.view(Bs, N, -1)
#         # image_fuse = torch.concat([img_feat, image_fuse.view(Bs, N, -1)], dim=-1)
#         fused_feat = torch.concat([point_feat, image_fuse.view(Bs, N, -1)], dim=-1)
#         fused_feat = fused_feat.transpose(1, 2)  # (B, output_dim, num_pc)
#         return fused_feat


from models.pspnet import PSPUpsample
class dino_extractor(nn.Module):
    def __init__(self, feat_ext, deep_features_size=64):
        super(dino_extractor, self).__init__()

        if feat_ext == "dino":
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        else:
            raise NotImplementedError
        
        self.up_1 = PSPUpsample(384, 256)
        self.up_2 = PSPUpsample(256, 128)
        self.up_3 = PSPUpsample(128, deep_features_size)
        
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.15)
        
    def forward(self, img):
        B, _, H, W = img.size()
        features_dict = self.dino.forward_features(img)
        dino_feats = features_dict['x_norm_patchtokens'].view(B, H//14, W//14, -1)
        dino_feats = dino_feats.permute(0, 3, 1, 2)
        feat = self.drop_1(dino_feats)
        feat = self.up_1(feat)
        feat = self.drop_2(feat)
        feat = self.up_2(feat)
        feat = self.drop_2(feat)
        feat = self.up_3(feat)
        feat = F.interpolate(feat, (H, W), mode='bilinear')
        return feat

class LinearLNReLU(nn.Sequential):

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.append(nn.Linear(in_dim, out_dim))
        self.append(nn.LayerNorm(out_dim))
        self.append(nn.ReLU(inplace=True))


class ResMLP(nn.Module):

    def __init__(self, in_dim, out_dim, expand=0.25) -> None:
        super().__init__()
        neck_dim = int(expand * out_dim)
        self.net = nn.Sequential(LinearLNReLU(in_dim, neck_dim),
                                 nn.Linear(neck_dim, out_dim),
                                 nn.LayerNorm(out_dim))

    def forward(self, x1, x2):
        return F.relu(x1 + self.net(x2), True)


class ObjectnessNet(nn.Module):
    def __init__(self, seed_feature_dim: int):
        super().__init__()
        self.obj_head = nn.Sequential(
            nn.Conv1d(seed_feature_dim, seed_feature_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(seed_feature_dim, 2, 1),
        )

    def forward(self, seed_features, end_points):
        # seed_features: (B, C, N)
        obj_logits = self.obj_head(seed_features)          # (B, 2, N)
        end_points["objectness_score"] = obj_logits
        return end_points


class MinkUNet14D_InterFuse(MinkUNet14D):
    def __init__(self, in_channels_3d, out_channels, img_dim, D=3):
        super().__init__(in_channels=in_channels_3d, out_channels=out_channels, D=D)

        # 逐层 concat 后用 1x1 sparse conv 压回原通道
        self.fuse_p1  = ME.MinkowskiConvolution(self.INIT_DIM + img_dim, self.INIT_DIM, kernel_size=1, dimension=D)
        self.fuse_p2  = ME.MinkowskiConvolution(self.INIT_DIM + img_dim, self.INIT_DIM, kernel_size=1, dimension=D)
        self.fuse_p4  = ME.MinkowskiConvolution(self.INIT_DIM + img_dim, self.INIT_DIM, kernel_size=1, dimension=D)
        self.fuse_p8  = ME.MinkowskiConvolution(self.PLANES[1] + img_dim, self.PLANES[1], kernel_size=1, dimension=D)  # 64
        self.fuse_p16 = ME.MinkowskiConvolution(self.PLANES[2] + img_dim, self.PLANES[2], kernel_size=1, dimension=D)  # 128

    @staticmethod
    def _scatter_mean(feats: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
        # feats: (K,C), idx: (K,) -> (M,C)
        C = feats.shape[1]
        out = feats.new_zeros((M, C))
        cnt = feats.new_zeros((M, 1))
        out.index_add_(0, idx, feats)
        cnt.index_add_(0, idx, torch.ones((feats.shape[0], 1), device=feats.device, dtype=feats.dtype))
        return out / cnt.clamp_min_(1.0)

    @staticmethod
    def _make_bn_coords_from_coors(coors_list, stride: int, device) -> torch.Tensor:
        """
        coors_list: list(B) of (N,3) base-grid voxel coords (no batch col)
        return: (B*N,4) coords in base-grid, but snapped to stride grid
        """
        coords4 = []
        for b, c in enumerate(coors_list):
            if not torch.is_tensor(c):
                c = torch.as_tensor(c, device=device)

            # ensure integer coords in base-grid
            if c.dtype.is_floating_point:
                c = torch.floor(c).long()
            else:
                c = c.long()

            if stride > 1:
                # SNAP to stride grid (still in base coordinate system)
                # floor_div * stride works for negative coords too
                c = torch.div(c, stride, rounding_mode='floor') * stride

            bcol = torch.full((c.shape[0], 1), b, device=device, dtype=torch.long)
            coords4.append(torch.cat([bcol, c], dim=1))

        return torch.cat(coords4, dim=0)


    @staticmethod
    def _pack_keys(coords4: torch.Tensor, shift_xyz: torch.Tensor, mx: int, my: int, mz: int) -> torch.Tensor:
        b = coords4[:, 0]
        xyz = coords4[:, 1:] + shift_xyz.view(1, 3)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return (((b * mx + x) * my + y) * mz + z)

    def _build_img_sparse_like(self, target: ME.SparseTensor, pfeat_BNC: torch.Tensor, coors_list, stride: int) -> ME.SparseTensor:
        """
        target: 当前 3D feature（决定坐标图）
        pfeat_BNC: (B,N,C) 对应尺度的 per-point 2D 语义
        stride: 1/2/4/8/16
        输出：与 target 完全相同 coord map key 的 img_sparse
        """
        device = target.F.device
        B, N, C = pfeat_BNC.shape
        BN = B * N
        feats_bnC = pfeat_BNC.reshape(BN, C)

        tgt_coords = target.C.to(device).long()      # (M,4) 可能原本在 CPU
        M = tgt_coords.shape[0]

        # 目标层 coords 是否落在 stride 网格上？
        if stride > 1:
            mod = torch.remainder(tgt_coords[:, 1:], stride)
            # 正常情况下应该全 0（或至少绝大多数为 0）
            if (mod != 0).any():
                print(f"[InterFuse][WARN] target coords not aligned to stride={stride} grid. ratio={(mod!=0).float().mean().item():.4f}")

        pts_coords = self._make_bn_coords_from_coors(coors_list, stride=stride, device=device)  # (BN,4)

        # shift to non-negative for packing
        xyz_all = torch.cat([pts_coords[:, 1:], tgt_coords[:, 1:]], dim=0)
        min_xyz = xyz_all.min(dim=0).values
        shift = (-min_xyz).clamp_min(0).long()

        pts_xyz = pts_coords[:, 1:] + shift.view(1, 3)
        tgt_xyz = tgt_coords[:, 1:] + shift.view(1, 3)
        max_xyz = torch.max(pts_xyz.max(dim=0).values, tgt_xyz.max(dim=0).values).long()

        mx = int(max_xyz[0].item()) + 1
        my = int(max_xyz[1].item()) + 1
        mz = int(max_xyz[2].item()) + 1
        if mx * my * mz > 2**62:
            raise RuntimeError(f"[InterFuse] key packing overflow risk: mx*my*mz={mx*my*mz}")

        pts_key = self._pack_keys(pts_coords, shift, mx, my, mz)  # (BN,)
        tgt_key = self._pack_keys(tgt_coords, shift, mx, my, mz)  # (M,)

        # map each point -> target index
        tgt_key_sorted, order = torch.sort(tgt_key)
        pos = torch.searchsorted(tgt_key_sorted, pts_key)
        valid = (pos < M) & (tgt_key_sorted[pos] == pts_key)
        if not torch.all(valid):
            bad = int((~valid).sum().item())
            raise RuntimeError(f"[InterFuse] {bad} points cannot map to target coords at stride={stride}. Check coors quantization/stride.")

        tgt_idx = order[pos]  # (BN,)
        img_feat_u = self._scatter_mean(feats_bnC, tgt_idx, M)  # (M,C)

        return ME.SparseTensor(
            features=img_feat_u,
            coordinate_map_key=target.coordinate_map_key,
            coordinate_manager=target.coordinate_manager
        )

    def forward(self, x_sparse: ME.SparseTensor, pfeat: dict, coors_list):
        # stride=1
        out = self.conv0p1s1(x_sparse)
        out = self.bn0(out)
        out = self.relu(out)

        img1 = self._build_img_sparse_like(out, pfeat['p1'], coors_list, stride=1)
        out = self.fuse_p1(ME.cat(out, img1))
        out_p1 = out

        # stride=2
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)

        img2 = self._build_img_sparse_like(out, pfeat['p2'], coors_list, stride=2)
        out = self.fuse_p2(ME.cat(out, img2))
        out_b1p2 = self.block1(out)

        # stride=4
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)

        img4 = self._build_img_sparse_like(out, pfeat['p4'], coors_list, stride=4)
        out = self.fuse_p4(ME.cat(out, img4))
        out_b2p4 = self.block2(out)

        # stride=8
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)

        img8 = self._build_img_sparse_like(out, pfeat['p8'], coors_list, stride=8)
        out = self.fuse_p8(ME.cat(out, img8))
        out_b3p8 = self.block3(out)

        # stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)

        img16 = self._build_img_sparse_like(out, pfeat['p16'], coors_list, stride=16)
        out = self.fuse_p16(ME.cat(out, img16))
        out = self.block4(out)

        # decoder unchanged
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)
    
    
class IGNet(nn.Module):
    def __init__(self,  m_point=1024, num_view=300, num_angle=12, num_depth=4, seed_feat_dim=256, img_feat_dim=64, 
                 is_training=True, multi_scale_grouping=False, fuse_type='early'):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim

        self.num_depth = num_depth
        self.num_angle = num_angle
        self.num_view = num_view
        self.multi_scale_grouping = multi_scale_grouping
        self.M_points = m_point
        assert self.num_view == NUM_VIEW and self.num_angle == NUM_ANGLE and self.num_depth == NUM_DEPTH
        self.fuse_type = fuse_type
        # self.img_backbone = psp_models['resnet34'.lower()]()

        # self.img_backbone = dino_extractor(feat_ext='dino')
        # self.img_backbone = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=3, classes=64)
        # for param in self.img_backbone.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.img_backbone.decoder.parameters():
        #     param.requires_grad = False
        # early fusion
        if self.fuse_type == 'none':
            self.img_feature_dim = 0
            self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            print('no fusion')
        elif self.fuse_type == 'early':
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                    deep_features_size=img_feat_dim, backend='resnet34')
            self.img_feature_dim = 0
            self.point_backbone = MinkUNet14D(in_channels=img_feat_dim, out_channels=self.seed_feature_dim, D=3)
            print('sparse convolution, early fusion')
        elif self.fuse_type == 'concat':
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                    deep_features_size=img_feat_dim, backend='resnet34')
            self.img_feature_dim = img_feat_dim
            self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            print('sparse convolution, late fusion (concatentation)')
        elif self.fuse_type == 'gate':
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                    deep_features_size=img_feat_dim, backend='resnet34')
            self.img_feature_dim = 0
            self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            self.fusion_module = GatedFusion(point_dim=self.seed_feature_dim, img_dim=img_feat_dim)
            print('sparse convolution, late fusion (Gated fusion)')
        elif self.fuse_type == 'add':
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                    deep_features_size=img_feat_dim, backend='resnet34')
            self.img_feature_dim = 0
            self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            self.fusion_module = AddFusion(point_dim=self.seed_feature_dim, img_dim=img_feat_dim)
            print('sparse convolution, late fusion (Add fusion)')
        elif self.fuse_type == 'direct':
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                    deep_features_size=img_feat_dim, backend='resnet34')
            self.img_feature_dim = 0
            self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            print('sparse convolution, direct fusion (RGB as sparse feats)')
        elif self.fuse_type == 'intermediate':
            self.img_feature_dim = 0
            self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                backend='resnet34',
                deep_features_size=img_feat_dim,      # 你现在 p1/p2 是 64
                out_dim=img_feat_dim,       # 统一维度（=img_dim）
                return_pyramid=True,
                pretrained=True
            )

            self.point_backbone = MinkUNet14D_InterFuse(
                in_channels_3d=3,           # 你 3D feats 的通道
                out_channels=self.seed_feature_dim,
                img_dim=img_feat_dim
            )
            print('sparse convolution, intermediate fusion (DeepViewAgg style)')
        elif self.fuse_type == 'learnable_align':
            raise NotImplementedError

            # self.img_feature_dim = img_feat_dim
            # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
            # self.fusion_module = LearnableAlign(self.seed_feature_dim, img_feat_dim, dropout=0.15, normalize=False)
            # print('late fusion (LearnableAlign)')
        # late fusion (Cross attention concatentation)
        # self.img_feature_dim = self.seed_feature_dim
        # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.fusion_module = LearnableAlign(self.seed_feature_dim, img_feat_dim, dropout=0.15, normalize=False)
        # print('late fusion (LearnableAlign)')
    
        self.objectness = ObjectnessNet(seed_feature_dim=self.seed_feature_dim + self.img_feature_dim)
        self.rot_head = RotationScoringNet(self.num_view, num_angle=self.num_angle,
                                                num_depth=self.num_depth,
                                                seed_feature_dim=self.seed_feature_dim + self.img_feature_dim, 
                                                is_training=self.is_training)
        self.depth_head = DepthNet(self.num_view, num_angle=self.num_angle, 
                                   num_depth=self.num_depth,
                                   seed_feature_dim=self.seed_feature_dim)

        # self.rot_head = RotationScoringNetGate(self.num_view, num_angle=self.num_angle,
        #                                         num_depth=self.num_depth,
        #                                         seed_feature_dim=self.seed_feature_dim + self.img_feature_dim, 
        #                                         dropout_rate=0.15,
        #                                         is_training=self.is_training)
        # self.depth_head = DepthNetGate(self.num_view, num_angle=self.num_angle, 
        #                            num_depth=self.num_depth,
        #                            seed_feature_dim=self.seed_feature_dim,
        #                            dropout_rate=0.15)
        
        if self.multi_scale_grouping:
            feat_dim = self.seed_feature_dim + self.img_feature_dim
            self.crop_scales = [0.25, 0.5, 0.75, 1.0]
            self.multi_scale_fuse = nn.Conv1d(feat_dim * 4, feat_dim, 1)
            self.multi_scale_gate = nn.Sequential(
                nn.Conv1d(feat_dim, feat_dim, 1),
                nn.Sigmoid()
            )
            self.crop1 = CloudCrop(nsample=16, seed_feature_dim=feat_dim, out_dim=self.seed_feature_dim)
            self.crop2 = CloudCrop(nsample=16, seed_feature_dim=feat_dim, out_dim=self.seed_feature_dim)
            self.crop3 = CloudCrop(nsample=16, seed_feature_dim=feat_dim, out_dim=self.seed_feature_dim)
            self.crop4 = CloudCrop(nsample=16, seed_feature_dim=feat_dim, out_dim=self.seed_feature_dim)
            self.crop_op_list = [self.crop1, self.crop2, self.crop3, self.crop4]
        else:
            self.crop = CloudCrop(nsample=32, seed_feature_dim=self.seed_feature_dim + self.img_feature_dim, out_dim=self.seed_feature_dim)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            # 跳过 img_backbone 的所有子模块
            # if name.startswith('img_backbone.encoder') or name.startswith('img_backbone.decoder'):
            #     continue
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, np.math.sqrt(2. / n))
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def _select_M_points(self, objectness_score, xyz_full):
        """
        objectness_score: (B, 2, N) logits
        xyz_full:         (B, N, 3)
        return:
            inds: (B, M) long indices into N
        """
        device = xyz_full.device
        B, N, _ = xyz_full.shape
        M = self.M_points

        # argmax mask: object class == 1
        objectness_pred = torch.argmax(objectness_score, dim=1)  # (B,N)
        objectness_mask = (objectness_pred == 1)                 # (B,N)

        inds_batch = []
        for b in range(B):
            idx_obj = torch.nonzero(objectness_mask[b], as_tuple=False).squeeze(1)  # (K,)

            # fallback：一个 object 点都没有 -> 用 “class1 logit 最大的 topk” 来凑
            if idx_obj.numel() == 0:
                cls1_logit = objectness_score[b, 1]  # (N,)
                k = min(M, N)
                idx_obj = torch.topk(cls1_logit, k=k, largest=True).indices  # (k,)

            K = int(idx_obj.numel())
            xyz_obj = xyz_full[b].index_select(0, idx_obj).contiguous()  # (K,3)

            # FPS only on object points
            if K >= M:
                fps_local = furthest_point_sample(xyz_obj.unsqueeze(0), M).squeeze(0)  # (M,)
                fps_local = fps_local.to(dtype=torch.long, device=device).contiguous()
                sel = idx_obj.index_select(0, fps_local)
            else:
                # K < M：先尽量取全，再 pad
                if K > 1:
                    fps_local = furthest_point_sample(xyz_obj.unsqueeze(0), K).squeeze(0)  # (K,)
                    fps_local = fps_local.to(dtype=torch.long, device=device).contiguous()
                    sel = idx_obj.index_select(0, fps_local)
                else:
                    sel = idx_obj  # (1,)

                pad = sel[torch.randint(0, sel.numel(), (M - sel.numel(),), device=device)]
                sel = torch.cat([sel, pad], dim=0)

            inds_batch.append(sel)

        return torch.stack(inds_batch, dim=0).contiguous()  # (B,M)

    def _gather_2d_to_points(self, feat2d: torch.Tensor, img_idxs: torch.Tensor, base_hw=(448, 448)):
        """
        feat2d: (B,C,Hf,Wf)
        img_idxs: (B,N)  flatten idx in base_hw (Hb*Wb), base_hw 默认为 (448,448)
        return: (B,N,C)
        """
        Hb, Wb = base_hw
        B, C, Hf, Wf = feat2d.shape

        ys = torch.div(img_idxs, Wb, rounding_mode='floor')   # (B,N)
        xs = img_idxs - ys * Wb                               # (B,N)

        # map base pixel (ys,xs) -> feature pixel (yf,xf)
        # 用 float 缩放再 floor，和你 dataset 的 resize 思路一致
        yf = torch.clamp((ys.float() * (Hf / Hb)).long(), 0, Hf - 1)
        xf = torch.clamp((xs.float() * (Wf / Wb)).long(), 0, Wf - 1)

        flat_f = (yf * Wf + xf)                               # (B,N)

        feat_flat = feat2d.view(B, C, -1)                     # (B,C,Hf*Wf)
        gather_idx = flat_f.unsqueeze(1).expand(-1, C, -1)     # (B,C,N)
        out = torch.gather(feat_flat, 2, gather_idx)          # (B,C,N)
        return out.transpose(1, 2).contiguous()               # (B,N,C)
     
    def forward(self, end_points):
        # use all sampled point cloud, B*Ns*3
        xyz_full = end_points['point_clouds']
        B, N, _ = xyz_full.shape
        device = xyz_full.device

        if self.fuse_type == 'intermediate':
            img = end_points['img']              # (B,3,448,448)
            img_idxs = end_points['img_idxs']    # (B,N) flat idx on 448*448
            H0, W0 = img.shape[-2], img.shape[-1]

            # 2D pyramid from PSPNet (you need to implement return_pyramid=True)
            pyr = self.img_backbone(img, return_pyramid=True)  # dict: p1/p2/p4/p8/p16

            # per-point 2D feats at each scale: (B,N,C)
            pfeat = {k: self._gather_2d_to_points(pyr[k], img_idxs, base_hw=(H0,W0))
                    for k in ['p1', 'p2', 'p4', 'p8', 'p16']}  # each -> (B,N,Cimg)

            # 3D backbone 的输入特征（保持你原逻辑：ones 或 direct 都行，这里用 feats）
            input_feats = end_points['feats']

        elif self.fuse_type in ['early', 'concat', 'gate', 'add']:
            img = end_points['img']
            img_idxs = end_points['img_idxs']
            img_feat = self.img_backbone(img)
            _, Cimg, _, _ = img_feat.shape

            img_feat = img_feat.view(B, Cimg, -1)
            img_idxs = img_idxs.unsqueeze(1).repeat(1, Cimg, 1)
            image_features = torch.gather(img_feat, 2, img_idxs).transpose(1, 2).contiguous()
            if self.fuse_type == 'early':
                input_feats = image_features
            else:
                input_feats = end_points['feats']
        else:
            image_features = None
            if self.fuse_type == 'direct':
                input_feats = [c * 2.0 - 1.0 for c in end_points['cloud_colors']]
            else:
                input_feats = end_points['feats']

        coordinates_batch, features_batch = ME.utils.sparse_collate(coords=[c for c in end_points['coors']], 
                                                                    feats=[f for f in input_feats], 
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, unique_map, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

        # # collate (建议先放 CPU，最稳)
        # coords_list = [c.detach().cpu().int() for c in end_points['coors']]   # list of (N,3)
        # feats_list  = [f.detach().cpu().float() for f in input_feats]        # list of (N,C)

        # coords_c, feats_c = ME.utils.sparse_collate(coords=coords_list, feats=feats_list, dtype=torch.float32)
        # # orig_num = coords_c.shape[0]   # 这里才是真正的原始点数，应该是 B*N (=512)

        # # quantize on CPU (不传 device)
        # coords_q, feats_q, unique_map, quantize2original = ME.utils.sparse_quantize(
        #     coords_c, feats_c, return_index=True, return_inverse=True
        # )

        # # # ---- sanity ----
        # # assert quantize2original is not None and quantize2original.numel() == orig_num, \
        # #     f"inverse_map numel {quantize2original.numel()} != orig_num {orig_num}"

        # # move BOTH to GPU before SparseTensor
        # coordinates_batch = coords_q.to(device)
        # features_batch  = feats_q.to(device)

        mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)

        if self.fuse_type == 'intermediate':
            # 后面构建 mink_input 不变
            out_sparse = self.point_backbone(mink_input, pfeat, end_points['coors'])
        else:
            out_sparse = self.point_backbone(mink_input)

        point_features = out_sparse.F
        point_features = point_features[quantize2original].view(B, N, -1).transpose(1, 2).contiguous()

        if self.fuse_type in ['concat']:
            feat_full = torch.concat([point_features, image_features.transpose(1, 2)], dim=1)
        elif self.fuse_type in ['gate', 'add']:
            feat_full = self.fusion_module(point_features, image_features.transpose(1, 2))
        else:
            feat_full = point_features

        # ----- graspable head on full scene -----
        end_points = self.objectness(feat_full, end_points)
        
        # ----- 方案A：选 M graspable points -----
        with torch.no_grad():
            inds = self._select_M_points(end_points["objectness_score"], xyz_full)
            
        xyz_sel = torch.gather(xyz_full, 1, inds.unsqueeze(-1).expand(-1, -1, 3)).contiguous()  # (B,M,3)
        feat_sel = torch.gather(feat_full, 2, inds.unsqueeze(1).expand(-1, feat_full.size(1), -1)).contiguous()  # (B,C,M)

        end_points["graspable_inds"] = inds
        end_points["xyz_graspable"] = xyz_sel
        end_points["seed_features"] = feat_sel
        end_points["point_clouds"] = xyz_sel
        end_points["D: Graspable Points"] = _as_tensor(float(self.M_points), device)  # 避免 float.item 崩

        # end_points['seed_features'] = seed_features  # (B, seed_feature_dim, num_seed)
        # end_points, rot_features = self.rot_head(seed_features, end_points)
        # seed_features = seed_features + rot_features
        end_points, feat_sel = self.rot_head(feat_sel, end_points)

        if self.is_training:
            top_inds = end_points["grasp_top_rot_inds"].long()  # (B,M)

            end_points = process_grasp_labels_scene(
                end_points, GRASP_MAX_WIDTH=GRASP_MAX_WIDTH,
                top_rot_inds=top_inds  # 用 pred top 来 slice GT score/width
            )
            grasp_rot_flat = grasp_rot.to(device)               # (V*A,3,3) 例如 (3600,3,3)
            grasp_top_rots = grasp_rot_flat[top_inds]           # (B,M,3,3)

        else:
            top_inds = end_points["grasp_top_rot_inds"].long()
            grasp_top_rots = grasp_rot.to(device)[top_inds]     # (B,M,3,3)
            
        B, M, _ = xyz_sel.shape
        if self.multi_scale_grouping:
            group_features = []
            for crop_scale, crop_op in zip(self.crop_scales, self.crop_op_list):
                crop_length = (0.04 + base_depth) * torch.ones((B, M, 1), device=device)
                crop_width = crop_scale * GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=device)
                crop_height = 0.02 * torch.ones_like(crop_length, device=device)
                crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1).contiguous()
                group_features.append(crop_op(xyz_sel, feat_sel, grasp_top_rots, crop_size))
            group_features = torch.cat(group_features, dim=1) #            
            group_features = self.multi_scale_fuse(group_features)
            seed_features_gate = self.multi_scale_gate(feat_sel) * feat_sel
            group_features = group_features + seed_features_gate
        else:
            crop_length = (0.04 + base_depth) * torch.ones((B, M, 1), device=device)
            crop_width = GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=device)
            crop_height = 0.02 * torch.ones_like(crop_length, device=device)
            crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1).contiguous()
            group_features = self.crop(xyz_sel, feat_sel, grasp_top_rots, crop_size)
        end_points = self.depth_head(group_features, end_points)
        return end_points


@torch.no_grad()
def process_grasp_labels_scene(end_points, GRASP_MAX_WIDTH=0.10, top_rot_inds=None):
    """
    生成 scene-level label，并把 match_grasp_view_and_label 的“top rot 切片 + log 归一化”合并进来。
    输出：
      - batch_grasp_score_full: (B,M,V,A,D)
      - batch_grasp_width_full: (B,M,V,A,D)
      - batch_grasp_rot_graspness: (B,M,V*A)
      - gt_top_rot_inds: (B,M)
      - batch_grasp_score: (B,M,D)   (top rot slice + log normalize)
      - batch_grasp_width: (B,M,D)
      - grasp_top_rot_mat_pred: (B,M,3,3)  # grasp_rot[pred_top]
      - grasp_top_rot_mat_gt:   (B,M,3,3)  # grasp_rot[gt_top]
    依赖 end_points：
      object_poses_list / grasp_points_list / grasp_offsets_list / grasp_labels_list
      point_clouds: (B,M,3)
    """
    device = end_points["point_clouds"].device
    seed_xyzs = end_points["point_clouds"]  # (B,M,3)
    B, M, _ = seed_xyzs.shape

    # 全局常量（你文件开头定义的）
    # grasp_rot: (V*A,3,3)  !!! 必须是 flatten 后的
    # NUM_VIEW, NUM_ANGLE, NUM_DEPTH
    assert grasp_rot.dim() == 3 and grasp_rot.shape[-2:] == (3, 3), "grasp_rot must be (V*A,3,3)"
    VA = NUM_VIEW * NUM_ANGLE
    assert grasp_rot.shape[0] == VA, f"grasp_rot first dim must be V*A={VA}"

    grasp_views = generate_grasp_views(NUM_VIEW).to(device)   # (V,3)
    grasp_views_ = grasp_views.unsqueeze(0)                   # (1,V,3)

    batch_scores_full = []
    batch_widths_full = []
    batch_rot_graspness = []
    batch_gt_top = []

    for i in range(B):
        seed_xyz = seed_xyzs[i]  # (M,3)

        poses_list = end_points["object_poses_list"][i]
        pts_list   = end_points["grasp_points_list"][i]
        off_list   = end_points["grasp_offsets_list"][i]   # widths: (P,V,A,D)
        lab_list   = end_points["grasp_labels_list"][i]    # scores: (P,V,A,D)

        if len(poses_list) == 0:
            scores_i = torch.zeros((M, NUM_VIEW, NUM_ANGLE, NUM_DEPTH), device=device)
            widths_i = torch.zeros((M, NUM_VIEW, NUM_ANGLE, NUM_DEPTH), device=device)
            rot_g_i  = torch.zeros((M, VA), device=device)
            batch_scores_full.append(scores_i)
            batch_widths_full.append(widths_i)
            batch_rot_graspness.append(rot_g_i)
            batch_gt_top.append(torch.zeros((M,), dtype=torch.long, device=device))
            continue

        all_points_trans, all_scores_aligned, all_widths_aligned = [], [], []

        for pose, gpts, gwidths, gscores in zip(poses_list, pts_list, off_list, lab_list):
            pose = torch.as_tensor(pose, dtype=torch.float32, device=device)
            if pose.shape == (4, 4):
                R = pose[:3, :3]
                pose3x4 = pose[:3, :]
            else:
                R = pose[:3, :3]
                pose3x4 = pose

            gpts    = torch.as_tensor(gpts, dtype=torch.float32, device=device)     # (P,3)
            gwidths = torch.as_tensor(gwidths, dtype=torch.float32, device=device)  # (P,V,A,D)
            gscores = torch.as_tensor(gscores, dtype=torch.float32, device=device)  # (P,V,A,D)

            # (1) grasp points -> scene/cam
            gpts_trans = transform_point_cloud(gpts, pose3x4, "3x4")  # (P,3)

            # (2) view re-index：把 “旋转后的 view” 映射回模板 view index
            grasp_views_trans = transform_point_cloud(grasp_views, R, "3x3")  # (V,3)
            _, view_inds, _ = knn_points(grasp_views_, grasp_views_trans.unsqueeze(0), K=1)
            view_inds = view_inds.squeeze(-1).squeeze(0)  # (V,)

            # (3) (V,A,3,3) rot set in cam, reorder by view_inds
            rot_cam = torch.matmul(R.unsqueeze(0), grasp_rot.to(device))      # (VA,3,3)
            rot_cam = rot_cam.view(NUM_VIEW, NUM_ANGLE, 3, 3)
            rot_cam = torch.index_select(rot_cam, 0, view_inds)               # (V,A,3,3)

            gscores = torch.index_select(gscores, 1, view_inds)               # (P,V,A,D)
            gwidths = torch.index_select(gwidths, 1, view_inds)               # (P,V,A,D)

            # (4) angle 对齐（你给的 align_angle_index）
            # 注意：这里必须传 flatten 的 grasp_rot (VA,3,3)
            _, gscores, gwidths = align_angle_index(grasp_rot.to(device), rot_cam, gscores, gwidths)

            all_points_trans.append(gpts_trans)
            all_scores_aligned.append(gscores)
            all_widths_aligned.append(gwidths)

        all_points_trans   = torch.cat(all_points_trans, dim=0)     # (Ptot,3)
        all_scores_aligned = torch.cat(all_scores_aligned, dim=0)   # (Ptot,V,A,D)
        all_widths_aligned = torch.cat(all_widths_aligned, dim=0)   # (Ptot,V,A,D)

        # (5) seed point -> nearest anchor
        _, nn_inds, _ = knn_points(seed_xyz.unsqueeze(0), all_points_trans.unsqueeze(0), K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)                    # (M,)

        scores_i = all_scores_aligned.index_select(0, nn_inds)      # (M,V,A,D)
        widths_i = all_widths_aligned.index_select(0, nn_inds)      # (M,V,A,D)

        # valid mask
        valid = (scores_i > 0) & (widths_i <= GRASP_MAX_WIDTH)
        scores_i = scores_i.clone()
        widths_i = widths_i.clone()
        scores_i[~valid] = 0

        # rot graspness label (M, V*A)
        rot_mask = (scores_i <= 0.6) & (scores_i > 0)               # (M,V,A,D)
        rot_g = rot_mask.float().mean(dim=-1).view(M, -1)           # (M,VA)
        rot_g = normalize_tensor(rot_g)

        gt_top = rot_g.argmax(dim=-1)                               # (M,)
        batch_gt_top.append(gt_top)

        batch_scores_full.append(scores_i)
        batch_widths_full.append(widths_i)
        batch_rot_graspness.append(rot_g)

    # stack full labels
    end_points["batch_grasp_score_full"] = torch.stack(batch_scores_full, dim=0)     # (B,M,V,A,D)
    end_points["batch_grasp_width_full"] = torch.stack(batch_widths_full, dim=0)     # (B,M,V,A,D)
    end_points["batch_grasp_rot_graspness"] = torch.stack(batch_rot_graspness, dim=0)  # (B,M,VA)
    end_points["gt_top_rot_inds"] = torch.stack(batch_gt_top, dim=0)                 # (B,M)

    # ---------- integrate match logic: slice top rot -> (B,M,D) ----------
    if top_rot_inds is None:
        # 默认用 pred 的话你传进来；否则就用 GT top
        top_rot_inds = end_points["gt_top_rot_inds"]
    end_points["top_rot_inds_for_label"] = top_rot_inds

    scores_full = end_points["batch_grasp_score_full"]   # (B,M,V,A,D)
    widths_full = end_points["batch_grasp_width_full"]   # (B,M,V,A,D)
    B, M, V, A, D = scores_full.shape
    scores_flat = scores_full.view(B, M, V * A, D)
    widths_flat = widths_full.view(B, M, V * A, D)

    idx = top_rot_inds.view(B, M, 1, 1).expand(-1, -1, 1, D)  # gather along rot dim
    top_scores = torch.gather(scores_flat, 2, idx).squeeze(2)  # (B,M,D)
    top_widths = torch.gather(widths_flat, 2, idx).squeeze(2)  # (B,M,D)

    # log normalize (按你原 match 的写法)
    u_max = top_scores.max()
    po_mask = top_scores > 0
    if po_mask.any() and float(u_max.item()) > 0:
        u_min = top_scores[po_mask].min()
        denom = torch.log(u_max / u_min + 1e-8) + 1e-8
        top_scores = top_scores.clone()
        top_scores[po_mask] = torch.log(u_max / top_scores[po_mask]) / denom

    end_points["batch_grasp_score"] = top_scores  # (B,M,D)
    end_points["batch_grasp_width"] = top_widths  # (B,M,D)

    # ---------- provide rot mats for crop without batch_grasp_rot ----------
    end_points["grasp_top_rot_mat_pred"] = grasp_rot.to(device)[end_points["grasp_top_rot_inds"]] \
        if "grasp_top_rot_inds" in end_points else None
    end_points["grasp_top_rot_mat_gt"] = grasp_rot.to(device)[end_points["gt_top_rot_inds"]]  # (B,M,3,3)

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

    # batch_grasp_scores_ids = torch.bucketize(top_rot_grasp_scores, score_bins.to(grasp_widths.device))
    # batch_grasp_widths_ids = torch.bucketize(top_rot_grasp_scores, width_bins.to(grasp_scores.device))

    end_points['batch_grasp_score'] = top_rot_grasp_scores  # (B, Ns, D)
    end_points['batch_grasp_width'] = top_rot_grasp_widths  # (B, Ns, D)
    
    # end_points['batch_grasp_score_ids'] = batch_grasp_scores_ids  # (B, Ns, D, score_bin_num)
    # end_points['batch_grasp_width_ids'] = batch_grasp_widths_ids  # (B, Ns, D, width_bin_num)
    return top_template_rot_mat, end_points


def pred_decode(end_points, normalize=False):
    batch_size, num_samples, _ = end_points['xyz_graspable'].shape
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()
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

        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height,
                                      grasp_depth, topk_grasp_rots, grasp_center, obj_ids],
                                     axis=-1).detach().cpu().numpy())

    return grasp_preds