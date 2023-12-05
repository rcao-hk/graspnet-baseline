import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from minkowski import MinkUNet14D

# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)

from pytorch3d.ops.knn import knn_points
import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup, furthest_point_sample, gather_operation
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix, batch_key_points, transform_point_cloud, \
                       GRASPNESS_THRESHOLD, GRASP_MAX_WIDTH, NUM_ANGLE, NUM_VIEW, NUM_DEPTH, M_POINT


angles = torch.tensor([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)])
views = generate_grasp_views(NUM_VIEW)  # num of views, (300,3), np.float32
# views_repeat = views.repeat_interleave(NUM_ANGLE, 0)  # (300*12,3)
# angles_repeat = angles.view(1, NUM_ANGLE).repeat_interleave(NUM_VIEW, 0).view(-1)  # (300*12,)
angles_repeat = angles.tile(NUM_VIEW)
views_repeat = views.repeat_interleave(NUM_ANGLE, dim=0)
grasp_rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat)  # (300, 12, 9)
depths = torch.linspace(0.01, 0.04, 4)
width_bins = torch.tensor([0.02, 0.04, 0.06, 0.08])
score_bins = torch.tensor([0.2, 0.4, 0.6, 0.8])


def generate_half_grasp_views(views, azimuth_range=(0, 2 * np.pi), elev_range=(0, 0.5 * np.pi)):
    views_constrained = []
    for pt in views:
        azimuth = np.math.atan2(pt[1], pt[0])

        if azimuth < 0:
            azimuth += 2.0 * np.pi

        # Elevation from (-0.5 * pi, 0.5 * pi)
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = np.math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        # if hemisphere and (pt[2] < 0 or pt[0] < 0 or pt[1] < 0):
        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                elev_range[0] <= elev <= elev_range[1]):
            continue
        views_constrained.append(pt)
    return views_constrained

views_constrained = np.asarray(generate_half_grasp_views(views.detach().cpu().numpy()))
views_constrained = torch.from_numpy(views_constrained)


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


eps = 1e-12
def normalize_tensor(tensor):
    max = tensor.max()
    min = tensor.min()
    tensor = (tensor - min) / (max - min + eps)
    return tensor


def knn_key_points_matching_sym(p1_key_points, p2_key_points, p2_key_points_sym):
    dis, inds_, _ = knn_points(p1_key_points, p2_key_points, K=1)
    dis_sym, inds_sym_, _ = knn_points(p1_key_points, p2_key_points_sym, K=1)
    sym_mask = torch.lt(dis, dis_sym)
    inds = inds_ * sym_mask + inds_sym_ * (~sym_mask)
    return inds


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


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points
    

# from pytorch3d.ops import estimate_pointcloud_normals
class NormalPolicy(nn.Module):
    def __init__(self, num_view, search_radius=30):
        super().__init__()
        self.num_view = num_view
        self.search_radius = search_radius
        self.cos = nn.CosineSimilarity(dim=2)
        
    def forward(self, end_points):
        # seed_xyz = end_points['point_clouds']
        xyz_graspable_idxs = end_points['xyz_graspable_idxs']
        B, num_seed = xyz_graspable_idxs.shape  # batch * NS

        # seed_normals = estimate_pointcloud_normals(seed_xyz, self.search_radius, 
        #                                            disambiguate_directions=False, 
        #                                            use_symeig_workaround=False)
        
        seed_normals = end_points['cloud_normals']
        seed_normals_flipped = seed_normals.transpose(1, 2).contiguous()
        vp_xyz = gather_operation(seed_normals_flipped, xyz_graspable_idxs).transpose(1, 2).squeeze(0).contiguous()  # Ns*3
        
        # camera_ray = torch.tensor([0, 0, 1]).to(vp_xyz.device)
        # camera_ray = camera_ray.unsqueeze(0).unsqueeze(0).tile((B, num_seed, 1))
        # vec_dis = self.cos(camera_ray, vp_xyz)
        # vec_mask = vec_dis > 0
        # vec_mask = vec_mask.unsqueeze(-1).tile((1, 1, 3))
        # vp_xyz[vec_mask] = -1 * vp_xyz[vec_mask]
        
        # for i in range(B):
        #     cur_xyz_idxs = xyz_graspable_idxs[i]
        #     cur_xyz_idxs = cur_xyz_idxs.unsqueeze(0)  # 1*Ns*3
        #
        #     cur_seed_normals = seed_normals[i]
        #     seed_normals_flipped = cur_seed_normals.unsqueeze(0).transpose(1, 2).contiguous()  # 1*3*Ns
        #     xyz_normals = gather_operation(seed_normals_flipped, cur_xyz_idxs).transpose(1, 2).squeeze(0).contiguous()  # Ns*3

        grasp_views = generate_grasp_views(self.num_view).to(vp_xyz.device)  # (num_view, 3)
        grasp_views = grasp_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
        
        if B == 1:
            vp_xyz = vp_xyz.unsqueeze(0)
        top_view_inds = []
        for i in range(B):

            cur_vp_xyz = vp_xyz[i]
            cur_grasp_view = grasp_views[i]

            cur_vp_xyz = cur_vp_xyz.unsqueeze(-1).transpose(1, 2) # (1, Ns, D)
            _, top_view_ind, _ = knn_points(cur_vp_xyz, cur_grasp_view, K=1)
            top_view_inds.append(top_view_ind)

        top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        end_points['grasp_top_view_inds'] = top_view_inds

        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot

        return end_points


# class GraspNet(nn.Module):
#     def __init__(self,  num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
#         super().__init__()
#         self.is_training = is_training
#         self.seed_feature_dim = seed_feat_dim
#         self.num_depth = num_depth
#         self.num_angle = num_angle
#         self.M_points = 1024
#         self.num_view = num_view

#         self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
#         self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
#         self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
#         # self.rotation = NormalPolicy(self.num_view)
#         self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
#         self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

#     def forward(self, end_points):
#         # use all sampled point cloud, B*Ns*3
#         seed_xyz = end_points['point_clouds']
#         B, point_num, _ = seed_xyz.shape  # batch _size
        
#         # point-wise features
#         coordinates_batch = end_points['coors']
#         features_batch = end_points['feats']
#         mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
#         seed_features = self.backbone(mink_input).F
#         seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

#         end_points = self.graspable(seed_features, end_points)
#         seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
#         objectness_score = end_points['objectness_score']
#         graspness_score = end_points['graspness_score'].squeeze(1)
#         objectness_pred = torch.argmax(objectness_score, 1)
#         objectness_mask = (objectness_pred == 1)
#         graspness_mask = graspness_score > GRASPNESS_THRESHOLD
#         graspable_mask = objectness_mask & graspness_mask

#         seed_features_graspable = []
#         seed_xyz_graspable = []
#         seed_xyz_graspable_idxs = []
#         graspable_num_batch = 0.
#         for i in range(B):
#             cur_mask = graspable_mask[i]
#             graspable_num_batch += cur_mask.sum()
#             cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
#             cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3

#             cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
#             fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
#             cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
#             cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
#             cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
#             cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()  # feat_dim*Ns

#             seed_features_graspable.append(cur_feat)
#             seed_xyz_graspable.append(cur_seed_xyz)
#             seed_xyz_graspable_idxs.append(fps_idxs.squeeze(0))

#         seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
#         seed_features_graspable = torch.stack(seed_features_graspable, 0)  # B*feat_dim*Ns
#         seed_xyz_graspable_idxs = torch.stack(seed_xyz_graspable_idxs, 0)  # B*1*Ns
#         end_points['xyz_graspable'] = seed_xyz_graspable
#         end_points['xyz_graspable_idxs'] = seed_xyz_graspable_idxs
#         end_points['graspable_count_stage1'] = graspable_num_batch / B
        
#         end_points, res_feat = self.rotation(seed_features_graspable, end_points)
#         seed_features_graspable = seed_features_graspable + res_feat

#         # end_points = self.rotation(end_points)

#         if self.is_training:
#             end_points = process_grasp_labels(end_points)
#             grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
#         else:
#             grasp_top_views_rot = end_points['grasp_top_view_rot']

#         group_features = self.crop(seed_xyz_graspable.contiguous(), 
#                                    seed_features_graspable.contiguous(), grasp_top_views_rot)
#         end_points = self.swad(group_features, end_points)

#         return end_points


# v0.4
class ScoringNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = F.relu(self.conv1(seed_features), inplace=True)
        scores = self.conv2(features)
        scores = scores.transpose(1, 2).contiguous() # (B, num_seed, 1)
        end_points['grasp_score_pred'] = scores
        return end_points


class RotationNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 6, 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(features)
        rotation_pred = features.transpose(1, 2).contiguous() # (B, num_seed, 6)
        end_points['grasp_rot_pred'] = rotation_pred
        return end_points


class WidthNet(nn.Module):
    def __init__(self, num_view, num_angle, num_depth, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        # regression-based
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth, 1)
        # classfication-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth * (len(width_bins)+1), 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(features)
        width_pred = features.transpose(1, 2).contiguous() # (B, num_seed, num_view*num_angle)
        
        # classification-based
        # width_pred = width_pred.view(B, num_seed, self.num_view * self.num_angle * self.num_depth, len(width_bins)+1)
        end_points['grasp_width_pred'] = width_pred
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
        # regression-based
        # self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth, 1)
        # classfication-based
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view * self.num_angle * self.num_depth * (len(score_bins)+1), 1)

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(features)
        rotation_scores = features.transpose(1, 2).contiguous() # (B, num_seed, num_view*num_angle)

        # classification-based
        rotation_scores = rotation_scores.view(B, num_seed, self.num_view * self.num_angle * self.num_depth, len(score_bins)+1)
        end_points['grasp_score_pred'] = rotation_scores
        return end_points


class IGNet(nn.Module):
    def __init__(self,  num_view=300, num_angle=12, num_depth=4, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = num_depth
        self.num_angle = num_angle
        # self.M_points = 1024
        self.num_view = num_view

        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        # self.scoring = nn.Conv1d(self.seed_feature_dim, 1, 1)
        self.rotation_head = RotationScoringNet(self.num_view, num_angle=self.num_angle,
                                                num_depth=self.num_depth,
                                                seed_feature_dim=self.seed_feature_dim, 
                                                is_training=self.is_training)
        # v0.4
        # self.scoring_head = ScoringNet(seed_feature_dim=self.seed_feature_dim)
        # self.rotation_head = RotationNet(seed_feature_dim=self.seed_feature_dim)
        self.width_head = WidthNet(self.num_view, num_angle=self.num_angle, 
                                   num_depth=self.num_depth,
                                   seed_feature_dim=self.seed_feature_dim)
        # self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)
    
    def forward(self, end_points):
        # use all sampled point cloud, B*Ns*3
        seed_xyz = end_points['point_clouds']
        B, point_num, _ = seed_xyz.shape  # batch _size
        
        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)

        # end_points['grasp_score'] = self.scoring(seed_features)
        # v0.4
        # end_points = self.scoring_head(seed_features, end_points)
        end_points = self.rotation_head(seed_features, end_points)
        end_points = self.width_head(seed_features, end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
            end_points = match_grasp_view_and_label(end_points)
        # else:
        #     grasp_top_views_rot = end_points['grasp_top_view_rot']

        # group_features = self.crop(seed_xyz_graspable.contiguous(), 
        #                            seed_features_graspable.contiguous(), grasp_top_views_rot)
        # end_points = self.swad(group_features, end_points)

        return end_points


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['point_clouds']  # (B, M_point, 3)
    # seed_normals = end_points['cloud_normals'] # (B, M_point, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_views_rot = []
    # batch_grasp_rot_max = []
    # batch_grasp_depth_max = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        object_pose = end_points['object_pose'][i]  # [(3, 4),]

        # get merged grasp points for label computation
        # grasp_points_merged = []
        # grasp_views_rot_merged = []
        # grasp_scores_merged = []
        # grasp_widths_merged = []
        # for obj_idx, pose in enumerate(poses):
        grasp_points = end_points['grasp_points'][i]  # (Np, 3)
        grasp_scores = end_points['grasp_labels'][i]  # (Np, V, A, D)
        grasp_widths = end_points['grasp_offsets'][i]  # (Np, V, A, D)
        _, V, A, D = grasp_scores.size()
        num_grasp_points = grasp_points.size(0)
        
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

        pred_grasp_rot_mat_ = grasp_rot.clone().to(object_pose.device)
        pred_width = 0.02 * torch.ones(len(pred_grasp_rot_mat_)).to(object_pose.device)
        orig_points = torch.zeros((len(pred_grasp_rot_mat_), 3)).to(object_pose.device)
        pred_key_points, pred_key_points_sym = batch_key_points(orig_points, pred_grasp_rot_mat_, pred_width)
        pred_key_points = pred_key_points.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))
        pred_key_points_sym = pred_key_points_sym.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))

        grasp_rot_trans_ = grasp_rot_trans.view((-1, 3, 3))
        temp_key_points, temp_key_points_sym = batch_key_points(orig_points, grasp_rot_trans_, pred_width)
        temp_key_points = temp_key_points.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))
        temp_key_points_sym = temp_key_points_sym.contiguous().view((NUM_VIEW, NUM_ANGLE, -1))

        view_angle_inds = knn_key_points_matching_sym(pred_key_points, temp_key_points, temp_key_points_sym)
        
        view_angle_inds = view_angle_inds.squeeze(-1)
        view_angle_rot_inds = view_angle_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        view_angle_score_inds = view_angle_inds.unsqueeze(0).unsqueeze(-1).expand(num_grasp_points, -1, -1, 4)
        
        grasp_rot_trans = torch.gather(grasp_rot_trans, 1, view_angle_rot_inds)  # (NUM_VIEWS, NUM_ANGLE, 3, 3)
        grasp_scores = torch.gather(grasp_scores, 2, view_angle_score_inds)
        grasp_widths = torch.gather(grasp_widths, 2, view_angle_score_inds)
        
        # add to list
        # grasp_points_merged.append(grasp_points_trans)
        # grasp_views_rot_merged.append(grasp_views_rot_trans)
        # grasp_scores_merged.append(grasp_scores)
        # grasp_widths_merged.append(grasp_widths)
        
        # grasp_points_merged = grasp_points_trans
        # grasp_views_rot_merged = grasp_views_rot_trans
        # grasp_scores_merged = grasp_scores
        # grasp_widths_merged = grasp_widths
        
        # grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        # grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        # grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # (Np', V, A, D)
        # grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', V, A, D)

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.unsqueeze(0)  # (1, Ns, 3)
        grasp_points_trans_ = grasp_points_trans.unsqueeze(0)  # (1, Np', 3)
        _, nn_inds, _ = knn_points(seed_xyz_, grasp_points_trans_, K=1) # (Ns)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)

        # assign anchor points to real points
        grasp_points_trans = torch.index_select(grasp_points_trans, 0, nn_inds)  # (Ns, 3)
        # grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_scores = torch.index_select(grasp_scores, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths = torch.index_select(grasp_widths, 0, nn_inds)  # (Ns, V, A, D)

        # v0.3.2
        # seed_normal = seed_normals[i]  # (Ns, 3)
        # seed_normal_mat = batch_normal_matrix(seed_normal)
        
        # views_cons = views_constrained.clone().to(seed_normal_mat.device)
        # views_cons_trans = transform_views(views_cons, seed_normal_mat)

        # grasp_views_trans_ = grasp_views.unsqueeze(0).tile((num_samples, 1, 1))
        # _, view_cons_inds, _ = knn_points(views_cons_trans, grasp_views_trans_, K=1)
        # view_cons_inds = view_cons_inds.squeeze(-1)
        
        # view_cons_score_inds = view_cons_inds.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, NUM_ANGLE, NUM_DEPTH))
        # grasp_scores = torch.gather(grasp_scores, 1, view_cons_score_inds)
        # grasp_widths = torch.gather(grasp_widths, 1, view_cons_score_inds)
                
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
        
        # add to batch
        batch_grasp_points.append(grasp_points_trans)
        # batch_grasp_views_rot.append(grasp_views_rot_trans)
        # batch_grasp_rot_max.append(grasp_rot_max)
        # batch_grasp_depth_max.append(depth_inds)
        batch_grasp_scores.append(grasp_scores)
        batch_grasp_widths.append(grasp_widths)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    # batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    # batch_grasp_rot_max = torch.stack(batch_grasp_rot_max, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)
        
    # compute view graspness
    # view_u_threshold = 0.6
    # view_grasp_num = 48
    # batch_grasp_view_valid_mask = (batch_grasp_scores <= view_u_threshold) & (batch_grasp_scores > 0) # (B, Ns, V, A, D)
    # batch_grasp_view_valid = batch_grasp_view_valid_mask.float()
    # batch_grasp_view_graspness = torch.sum(torch.sum(batch_grasp_view_valid, dim=-1), dim=-1) / view_grasp_num  # (B, Ns, V)
    # view_graspness_min, _ = torch.min(batch_grasp_view_graspness, dim=-1)  # (B, Ns)
    # view_graspness_max, _ = torch.max(batch_grasp_view_graspness, dim=-1)
    # view_graspness_max = view_graspness_max.unsqueeze(-1).expand(-1, -1, 300)  # (B, Ns, V)
    # view_graspness_min = view_graspness_min.unsqueeze(-1).expand(-1, -1, 300)  # same shape as batch_grasp_view_graspness
    # batch_grasp_view_graspness = (batch_grasp_view_graspness - view_graspness_min) / (view_graspness_max - view_graspness_min + 1e-5)

    # batch_grasp_scores = batch_grasp_scores[:, :, :, :, 0].view(batch_size, num_samples, -1)
    # batch_grasp_widths = batch_grasp_widths[:, :, :, :, 0].view(batch_size, num_samples, -1)
    batch_grasp_scores = batch_grasp_scores[:, :, :, :, :].view(batch_size, num_samples, -1)
    batch_grasp_widths = batch_grasp_widths[:, :, :, :, :].view(batch_size, num_samples, -1)

    # score_mask = batch_grasp_scores > 0.0
    # batch_grasp_scores[score_mask] = torch.log(1.0 / batch_grasp_scores[score_mask])

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores = 1.1 - batch_grasp_scores
    batch_grasp_scores[~label_mask] = 0
    
    batch_grasp_widths_ids = torch.bucketize(batch_grasp_widths, width_bins.to(batch_grasp_widths.device))
    batch_grasp_scores_ids = torch.bucketize(batch_grasp_scores, score_bins.to(batch_grasp_scores.device))
    
    end_points['batch_grasp_point'] = batch_grasp_points
    # end_points['batch_grasp_view_rot'] = batch_grasp_views_rot
    # end_points['batch_grasp_rot_max'] = batch_grasp_rot_max
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_width_ids'] = batch_grasp_widths_ids
    end_points['batch_grasp_score_ids'] = batch_grasp_scores_ids
    # end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness

    return end_points


def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    # top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    # template_views_rot = end_points['batch_grasp_view_rot']  # (B, Ns, V, 3, 3)
    grasp_scores = end_points['batch_grasp_score']  # (B, Ns, V, A)
    grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, 3)

    # B, Ns, V, A, D = grasp_scores.size()
    # top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    # top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    # top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    # top_view_grasp_scores = torch.gather(grasp_scores, 2, top_view_inds_).squeeze(2)
    # top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)

    # u_max = grasp_scores.max()
    # po_mask = grasp_scores > 0
    # po_mask_num = torch.sum(po_mask)
    # if po_mask_num > 0:
        # grasp_scores_record = grasp_scores[po_mask]
        # print('before min:{}, max:{}'.format(grasp_scores_record.min(), grasp_scores_record.max()))
        
        # grasp_scores[po_mask] = torch.log(u_max / grasp_scores[po_mask])
        
        # grasp_scores_record = grasp_scores[po_mask]
        # print('before min:{}, max:{}'.format(grasp_scores_record.min(), grasp_scores_record.max()))
        
    # u_min = top_view_grasp_scores[po_mask].min()
    # top_view_grasp_scores[po_mask] = torch.log(u_max / top_view_grasp_scores[po_mask]) / torch.log(u_max / u_min)
    
    end_points['batch_grasp_score'] = grasp_scores  # (B, Ns, A, D)
    end_points['batch_grasp_width'] = grasp_widths  # (B, Ns, A, D)

    return end_points


def pred_decode(end_points, normalize=False):
    grasp_center = end_points['point_clouds']
    batch_size, num_samples, _ = grasp_center.shape
    
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['point_clouds'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.

        if normalize:
            grasp_score = normalize_tensor(grasp_score)
            
        # v0.4
        # grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_score, grasp_score_inds = torch.max(grasp_score, dim=1)  # [M_POINT]
        topk_grasp_score, topk_grasp_inds = torch.topk(grasp_score, k=300)

        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        topk_grasp_width = grasp_width[topk_grasp_inds]
        topk_grasp_center = grasp_center[topk_grasp_inds]
        
        # grasp_rots_temp = grasp_rot.clone().to(grasp_center.device)
        # topk_grasp_rots = grasp_rots_temp[grasp_score_inds[topk_grasp_inds]].view(-1, 9)
        # topk_grasp_rots = topk_grasp_rots.to(grasp_center.device)
        
        # v0.3.3
        views_rot = grasp_rot.clone().to(grasp_center.device)
        views_num = len(views_rot)
        views_rot = views_rot.unsqueeze(0).tile((num_samples, 1, 1, 1))
        views_rot = views_rot.view((num_samples, NUM_VIEW, NUM_ANGLE, 3, 3))
        
        # v0.3.2
        # grasp_normal = end_points['cloud_normals'][i].float()
        # grasp_normal_mat = batch_normal_matrix(grasp_normal)
        
        # views_cons = views_constrained.clone().to(grasp_normal_mat.device)
        # views_num = len(views_cons)
        # views_cons_trans = transform_views(views_cons, grasp_normal_mat)
        # views_cons_repeat = views_cons_trans.repeat_interleave(NUM_ANGLE, dim=1)
        # views_cons_repeat = views_cons_repeat.view((-1, 3))
        # angles_cons_repeat = angles.unsqueeze(0).tile((num_samples, views_num)).to(grasp_normal_mat.device)
        # angles_cons_repeat = angles_cons_repeat.view(-1)
        # views_cons_rot = batch_viewpoint_params_to_matrix(-views_cons_repeat, angles_cons_repeat)  # (150, 12, 9)
        
        # views_cons_rot = views_cons_rot.view((num_samples, views_cons_num*NUM_ANGLE, 3, 3))
        # topk_grasp_rots = views_cons_rot[topk_grasp_inds, grasp_score_inds[topk_grasp_inds]].view(-1, 9)
        # grasp_depth = 0.01 * torch.ones_like(topk_grasp_score)
        # views_cons_rot = views_cons_rot.view((num_samples, views_num, NUM_ANGLE, 3, 3))
                
        # v0.3.2.2
        view_inds, angle_inds, depth_inds = unravel_index(grasp_score_inds[topk_grasp_inds], 
                                                          (views_num, NUM_ANGLE, NUM_DEPTH))
        topk_grasp_rots = views_rot[topk_grasp_inds, view_inds, angle_inds].view(-1, 9)
        
        grasp_depth = depths.clone().to(grasp_center.device)
        grasp_depth = grasp_depth.unsqueeze(0).tile((num_samples, 1))
        grasp_depth = torch.gather(grasp_depth, 1, depth_inds.view(-1, 1))
        
        topk_grasp_score = topk_grasp_score.view(-1, 1)
        grasp_height = 0.02 * torch.ones_like(topk_grasp_score)
        obj_ids = -1 * torch.ones_like(topk_grasp_score)
        grasp_preds.append(torch.cat([topk_grasp_score, topk_grasp_width, grasp_height,
                                      grasp_depth, topk_grasp_rots, topk_grasp_center, obj_ids], axis=-1).detach().cpu().numpy())
        
    return grasp_preds