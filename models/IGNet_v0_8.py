import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
import MinkowskiEngine as ME
from models.minkowski import MinkUNet14D

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from pytorch3d.ops.knn import knn_points
import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import RectangularQueryAndGroup
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
grasp_rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat)  # (300, 12, 9)
depths = torch.linspace(0.01, 0.04, 4)
# score_bins = torch.tensor([0.2, 0.4, 0.6, 0.8])
# width_bins = torch.tensor([0.02, 0.04, 0.06, 0.08])

# v0.6.3.1
score_bins = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
width_bins = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])


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
        # self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     # nn.ReLU(inplace=True),
        # )
        
        # # # regression-based
        # self.outconv = nn.Conv1d(self.in_dim, self.num_depth * 2, 1)
        # self.outconv = nn.Linear(self.in_dim, self.num_depth * 2)
        # self.widthnet = nn.Sequential(nn.Dropout(0.15),
        #                               nn.Linear(self.in_dim, self.num_depth))
        # self.scorenet = nn.Sequential(nn.Dropout(0.15),
        #                               nn.Linear(self.in_dim, self.num_depth))
        # self.widthnet = nn.Sequential(nn.Dropout(0.15),
        #                               nn.Conv1d(self.in_dim, self.num_depth, 1))
        # self.scorenet = nn.Sequential(nn.Dropout(0.15),
        #                               nn.Conv1d(self.in_dim, self.num_depth, 1))
        # self.widthnet = nn.Sequential(nn.Dropout(0.2),
        #                               nn.Conv1d(self.in_dim, self.num_depth, 1))
        # self.scorenet = nn.Sequential(nn.Dropout(0.2),
        #                               nn.Conv1d(self.in_dim, self.num_depth, 1))
        self.depthnet = nn.Sequential(nn.Dropout(0.15),
                                      nn.Conv1d(self.in_dim, self.num_depth * 2, 1))
        # self.widthnet = nn.Sequential(
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.15),
        #     nn.Linear(self.in_dim, self.num_depth))
        # self.scorenet = nn.Sequential(
        #     nn.Linear(self.in_dim, self.in_dim),
        #     nn.LayerNorm(self.in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.15),
        #     nn.Linear(self.in_dim, self.num_depth))
        # self.depthnet = nn.Sequential(LinearLNReLU(self.in_dim, self.in_dim),
        #                                nn.Dropout(0.2),
        #                                nn.Linear(self.in_dim, self.num_depth * 2))
        
        # classfication-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_depth * 2 * (len(width_bins)+1), 1)
        # regression-based score only
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_depth, 1)
        # self.act =  nn.ReLU(inplace=True)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        # features = self.act(self.conv1(seed_features))
        # features = self.mlp(seed_features)
        # predicts = self.depthnet(features)

        # seed_features = seed_features.transpose(1, 2)
        # res_features = self.mlp(seed_features)
        # features = F.relu(seed_features + res_features, inplace=True)
        # pred_width = self.widthnet(features)
        # pred_score = self.scorenet(features)

        # end_points['grasp_width_pred'] = pred_width
        # end_points['grasp_score_pred'] = pred_score

        # features = self.mlp(seed_features)
        # pred_width = self.widthnet(features).transpose(1, 2)
        # pred_score = self.scorenet(features).transpose(1, 2)
        
        # end_points['grasp_width_pred'] = pred_width
        # end_points['grasp_score_pred'] = pred_score

        features = self.mlp(seed_features)
        predicts = self.depthnet(features)
        
        # predicts = self.depthnet(seed_features.transpose(1, 2).contiguous())
        # predicts = predicts.transpose(1, 2).contiguous()
        
        # regression-based
        predicts = predicts.view(B, 2, self.num_depth, num_seed)
        predicts = predicts.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)
        
        # # classification-based
        # predicts = predicts.view(B, 2, self.num_depth, len(width_bins)+1, num_seed)
        # predicts = predicts.permute(0, 1, 4, 2, 3)
        
        end_points['grasp_score_pred'] = predicts[:, 0]
        end_points['grasp_width_pred'] = predicts[:, 1]
        
        # regression-based score only
        # predicts = predicts.view(B, self.num_depth, num_seed)
        # predicts = predicts.permute(0, 2, 1) # (B, num_seed, num_depth)
        
        # end_points['grasp_score_pred'] = predicts
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
    

class DepthNetGate(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, dropout_rate):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.dropout_rate = dropout_rate
        self.gate_ffn = GateFFN(self.in_dim, self.dropout_rate * 2, self.in_dim * 2)
        self.depthnet = nn.Sequential(nn.Dropout(self.dropout_rate),
                                      nn.Conv1d(self.in_dim, self.num_depth * 2, 1))

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()

        features = self.gate_ffn(seed_features)
        predicts = self.depthnet(features)

        predicts = predicts.view(B, 2, self.num_depth, num_seed)
        predicts = predicts.permute(0, 1, 3, 2) # (B, 2, num_seed, num_depth)

        end_points['grasp_score_pred'] = predicts[:, 0]
        end_points['grasp_width_pred'] = predicts[:, 1]

        return end_points


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
    
    
# class RotationGraspableNet(nn.Module):
#     def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
#         super().__init__()
#         self.num_view = num_view
#         self.num_angle = num_angle
#         self.num_depth = num_depth
#         self.in_dim = seed_feature_dim
#         self.is_training = is_training
#         self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
#         self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
#         self.conv3 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle, 1)
#         self.bn1 = nn.BatchNorm1d(self.in_dim)
#         self.bn2 = nn.BatchNorm1d(self.in_dim)
        
#     def forward(self, seed_features, end_points):
#         B, _, num_seed = seed_features.size()
        
#         res_features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
#         res_features = F.relu(self.bn2(self.conv2(res_features)), inplace=True)
#         rotation_scores = self.conv3(res_features)
#         rotation_scores = rotation_scores.transpose(1, 2).contiguous()  # (B, num_seed, num_view * num_angle)
        
#         end_points['grasp_rot_graspness_pred'] = rotation_scores

#         if self.is_training:
#             # normalize view graspness score to 0~1
#             rot_score = rotation_scores.clone().detach()
#             rot_score = normalize_tensor(rot_score)
#             top_rot_inds = []
#             for i in range(B):
#                 top_rot_inds_batch = torch.multinomial(rot_score[i], 1, replacement=False)
#                 top_rot_inds.append(top_rot_inds_batch)
#             top_rot_inds = torch.stack(top_rot_inds, dim=0).squeeze(-1)  # B, num_seed
#         else:
#             _, top_rot_inds = torch.max(rotation_scores, dim=-1)  # (B, num_seed)
#             temp_grasp_rots = grasp_rot.clone().to(seed_features.device)
#             temp_grasp_rots = temp_grasp_rots.view(1, 1, self.num_view*self.num_angle, 3, 3)
#             temp_grasp_rots = temp_grasp_rots.expand(B, num_seed, -1, -1, -1).contiguous()
#             top_rot_inds_ = top_rot_inds.view(B, num_seed, 1, 1, 1).expand(-1, -1, -1, 3, 3)
#             grasp_top_rot = torch.gather(temp_grasp_rots, 2, top_rot_inds_).squeeze(2)
#             end_points['grasp_top_rot'] = grasp_top_rot

#         end_points['grasp_top_rot_inds'] = top_rot_inds
#         return end_points, res_features
    
    
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
        point_feat = point_feat.transpose(1, 2)
        img_feat = img_feat.transpose(1, 2)
        
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
        point_feat = point_feat.transpose(1, 2)
        img_feat = img_feat.transpose(1, 2)
        
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

from flash_attn.modules.mha import CrossAttention
from einops import rearrange
class LearnableAlign(nn.Module):
    def __init__(self, point_dim, img_dim, dropout, normalize=False, in_proj_bias=True):
        super(LearnableAlign, self).__init__()
        self.feat_dim = self.point_dim = point_dim
        self.img_dim = img_dim
        self.normalize = normalize
        self.dropout = dropout
        
        if self.normalize:
            self.point_norm = nn.LayerNorm(self.point_dim)
            self.img_norm = nn.LayerNorm(self.img_dim)

        self.img_mlp = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LayerNorm(128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, self.feat_dim),
            nn.LayerNorm(self.feat_dim), 
            nn.ReLU(inplace=True),
        )
        # self.point_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
        # self.img_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        
        self.img_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
        self.point_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        
        # self.point_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        # self.image_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        
        # self.point_cross_attn = CrossAttention(attention_dropout=dropout)
        self.image_cross_attn = CrossAttention(attention_dropout=dropout)
        self.img_feat_out = nn.Linear(self.feat_dim, self.feat_dim)
                                                        
    def forward(self, point_feat, img_feat):
        if self.normalize:
            point_feat = self.point_norm(point_feat)
            img_feat = self.img_norm(img_feat)

        Bs, N, _ = point_feat.shape
        
        # point_feat = point_feat.transpose(1, 2)  # (B, point_dim, num_pc)
        # img_feat = img_feat.transpose(1, 2)  # (B, img_dim, num_pc)
        
        img_feat = self.img_mlp(img_feat)
        # img_q = self.img_q_proj(img_feat)
        img_kv = self.img_kv_proj(img_feat)
        point_q = self.point_q_proj(point_feat)
        # point_kv = self.point_kv_proj(point_feat)
        
        # img_q = self.img_q_proj(img_feat).half()
        # img_kv = self.img_kv_proj(img_feat).half()
        # point_q = self.point_q_proj(point_feat).half()
        # point_kv = self.point_kv_proj(point_feat).half()
        
        # img_q = rearrange(img_q.transpose(1, 2), "... (h d) -> ... h d", d=self.feat_dim)
        # point_kv = rearrange(point_kv.transpose(1, 2), "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
        # point_fuse = self.point_cross_attn(img_q, point_kv)
        # # point_fuse = point_feat + point_fuse.view(Bs, N, -1)
        # # point_fuse = torch.concat([point_feat, point_fuse.view(Bs, N, -1)], dim=-1)

        # fused_feat = torch.concat([point_feat.transpose(1, 2), point_fuse.view(Bs, N, -1)], dim=-1)
        
        point_q = rearrange(point_q, "... (h d) -> ... h d", d=self.feat_dim)
        img_kv = rearrange(img_kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
        image_fuse = self.image_cross_attn(point_q, img_kv)
        img_feat = self.img_feat_out(image_fuse)
        # image_fuse = img_feat + image_fuse.view(Bs, N, -1)
        # image_fuse = torch.concat([img_feat, image_fuse.view(Bs, N, -1)], dim=-1)
        fused_feat = torch.concat([point_feat, image_fuse.view(Bs, N, -1)], dim=-1)
        fused_feat = fused_feat.transpose(1, 2)  # (B, output_dim, num_pc)
        return fused_feat


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

# import segmentation_models_pytorch as smp

class IGNet(nn.Module):
    def __init__(self,  num_view=300, num_angle=12, num_depth=4, seed_feat_dim=256, img_feat_dim=64, 
                 is_training=True, multi_scale_grouping=False):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim

        self.num_depth = num_depth
        self.num_angle = num_angle
        self.num_view = num_view
        self.multi_scale_grouping = multi_scale_grouping

        # self.img_backbone = psp_models['resnet34'.lower()]()
        self.img_backbone = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, 
                                   deep_features_size=img_feat_dim, backend='resnet34')
        # self.img_backbone = dino_extractor(feat_ext='dino')
        # self.img_backbone = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=3, classes=64)
        # for param in self.img_backbone.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.img_backbone.decoder.parameters():
        #     param.requires_grad = False
                
        # early fusion
        self.img_feature_dim = 0
        self.point_backbone = MinkUNet14D(in_channels=img_feat_dim, out_channels=self.seed_feature_dim, D=3)
        print('early fusion')
        
        # # late fusion (concatentation)
        # self.img_feature_dim = img_feat_dim
        # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # print('late fusion (concatentation)')
        
        # late fusion (Cross attention concatentation)
        # self.img_feature_dim = self.seed_feature_dim
        # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.fusion_module = LearnableAlign(self.seed_feature_dim, img_feat_dim, dropout=0.15, normalize=False)
        # print('late fusion (LearnableAlign)')
        
        # late fusion (Gated fusion)
        # self.img_feature_dim = 0
        # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.fusion_module = GatedFusion(point_dim=self.seed_feature_dim, img_dim=img_feat_dim)
        # print('late fusion (Gated fusion)')
        
        # late fusion (Add fusion)
        # self.img_feature_dim = 0
        # self.point_backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.fusion_module = AddFusion(point_dim=self.seed_feature_dim, img_dim=img_feat_dim)
        # print('late fusion (Add fusion)')
                
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
        # self.apply(self._init_weights)
        self._init_weights()

    def _init_weights(self):
        # # if isinstance(m, nn.BatchNorm1d):
        # #     nn.init.constant_(m.bias, 0)
        # #     nn.init.constant_(m.weight, 1.0)
        # # elif isinstance(m, nn.Conv1d):
        # #     trunc_normal_(m.weight, std=.02)
        # #     if m.bias is not None:
        # #         nn.init.constant_(m.bias, 0)
        # if isinstance(self.img_backbone, PSPNet):
        #     for name, m in self.img_backbone.named_modules():
        #         if name.startswith('feats'):
        #             continue
        #         else:
        #             if isinstance(m, nn.Conv2d):
        #                 nn.init.xavier_normal_(m.weight)
        #                 if m.bias is not None:
        #                     # nn.init.normal_(m.bias)
        #                     nn.init.constant_(m.bias, 0)
        #             if isinstance(m, nn.BatchNorm2d):
        #                 nn.init.constant_(m.bias, 0)
        #                 nn.init.constant_(m.weight, 1.0)

        # for head in [self.rot_head, self.depth_head]:
        #     for m in head.modules():
        #         if isinstance(m, nn.BatchNorm1d):
        #             nn.init.constant_(m.bias, 0)
        #             nn.init.constant_(m.weight, 1.0)
        #         elif isinstance(m, nn.Conv1d):
        #             nn.init.kaiming_normal_(m.weight, std=0.02)
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, np.math.sqrt(2. / n))
        #     elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, (nn.Linear, nn.Conv1d)):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
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
                              
    def forward(self, end_points):
        # use all sampled point cloud, B*Ns*3
        seed_xyz = end_points['point_clouds']
        B, point_num, _ = seed_xyz.shape  # batch _size
        
        img = end_points['img']
        img_idxs = end_points['img_idxs']
        
        img_feat = self.img_backbone(img)
        _, img_dim, _ , _ = img_feat.size()
        
        img_feat = img_feat.view(B, img_dim, -1)
        img_idxs = img_idxs.unsqueeze(1).repeat(1, img_dim, 1)
        image_features = torch.gather(img_feat, 2, img_idxs).contiguous()
        
        # early fusion
        image_features = image_features.transpose(1, 2)
        coordinates_batch, features_batch = ME.utils.sparse_collate(coords=[c for c in end_points['coors']], 
                                                                    feats=[f for f in image_features], 
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=seed_xyz.device)
        mink_input = ME.SparseTensor(coordinates=coordinates_batch, features=features_batch)
        point_features = self.point_backbone(mink_input).F
        seed_features = point_features[quantize2original].view(B, point_num, -1).transpose(1, 2)

        # late fusion (concatentation)
        # coordinates_batch, features_batch = ME.utils.sparse_collate(coords=[c for c in end_points['coors']], 
        #                                                             feats=[f for f in end_points['feats']], 
        #                                                             dtype=torch.float32)
        # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        #     coordinates_batch, features_batch, return_index=True, return_inverse=True, device=seed_xyz.device)
        # mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        # point_features = self.point_backbone(mink_input).F
        # point_features = point_features[quantize2original].view(B, point_num, -1).transpose(1, 2)
        # seed_features = torch.concat([point_features, image_features], dim=1)
    
        # late fusion (cross attention concatentation, gated fusion, add fusion)
        # coordinates_batch, features_batch = ME.utils.sparse_collate(coords=[c for c in end_points['coors']], 
        #                                                             feats=[f for f in end_points['feats']], 
        #                                                             dtype=torch.float32)
        # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        #     coordinates_batch, features_batch, return_index=True, return_inverse=True, device=seed_xyz.device)
        # mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        # point_features = self.point_backbone(mink_input).F
        # point_features = point_features[quantize2original].view(B, point_num, -1)
        # image_features = image_features.transpose(1, 2)
        # seed_features = self.fusion_module(point_features, image_features)
        
        end_points['seed_features'] = seed_features  # (B, seed_feature_dim, num_seed)
        # end_points, rot_features = self.rot_head(seed_features, end_points)
        # seed_features = seed_features + rot_features
        end_points, seed_features = self.rot_head(seed_features, end_points)

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_rots, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_rots = end_points['grasp_top_rot']
        
        if self.multi_scale_grouping:
            group_features = []
            for crop_scale, crop_op in zip(self.crop_scales, self.crop_op_list):
                crop_length = (0.04 + base_depth) * torch.ones((B, point_num, 1), device=seed_xyz.device)
                crop_width = crop_scale * GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=seed_xyz.device)
                crop_height = 0.02 * torch.ones_like(crop_length, device=seed_xyz.device)
                crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1)
                group_features.append(crop_op(seed_xyz.contiguous(), seed_features.contiguous(), 
                                                grasp_top_rots, crop_size.contiguous()))
            group_features = torch.cat(group_features, dim=1) #            
            group_features = self.multi_scale_fuse(group_features)
            seed_features_gate = self.multi_scale_gate(seed_features) * seed_features
            group_features = group_features + seed_features_gate
        else:
            crop_length = (0.04 + base_depth) * torch.ones((B, point_num, 1), device=seed_xyz.device)
            crop_width = GRASP_MAX_WIDTH * torch.ones_like(crop_length, device=seed_xyz.device)
            crop_height = 0.02 * torch.ones_like(crop_length, device=seed_xyz.device)
            crop_size = torch.concat([crop_length, crop_width, crop_height], dim=-1).contiguous()
            group_features = self.crop(seed_xyz.contiguous(), seed_features.contiguous(),
                                       grasp_top_rots, crop_size)
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

    # batch_grasp_scores_ids = torch.bucketize(top_rot_grasp_scores, score_bins.to(grasp_widths.device))
    # batch_grasp_widths_ids = torch.bucketize(top_rot_grasp_scores, width_bins.to(grasp_scores.device))

    end_points['batch_grasp_score'] = top_rot_grasp_scores  # (B, Ns, D)
    end_points['batch_grasp_width'] = top_rot_grasp_widths  # (B, Ns, D)
    
    # end_points['batch_grasp_score_ids'] = batch_grasp_scores_ids  # (B, Ns, D, score_bin_num)
    # end_points['batch_grasp_width_ids'] = batch_grasp_widths_ids  # (B, Ns, D, width_bin_num)
    return top_template_rot_mat, end_points


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

        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height,
                                      grasp_depth, topk_grasp_rots, grasp_center, obj_ids],
                                     axis=-1).detach().cpu().numpy())

    return grasp_preds