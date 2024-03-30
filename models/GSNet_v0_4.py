import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from .minkowski import MinkUNet14D

from pytorch3d.ops.knn import knn_points
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix, batch_get_key_points, transform_point_cloud, GRASPNESS_THRESHOLD, GRASP_MAX_WIDTH, NUM_ANGLE, NUM_VIEW, NUM_DEPTH, M_POINT
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
# from models.coral_loss import corn_label_from_logits


angles = torch.tensor([np.pi / NUM_ANGLE * i for i in range(NUM_ANGLE)])
views = generate_grasp_views(NUM_VIEW)  # num of views, (300,3), np.float32
# views_repeat = views.repeat_interleave(NUM_ANGLE, 0)  # (300*12,3)
# angles_repeat = angles.view(1, NUM_ANGLE).repeat_interleave(NUM_VIEW, 0).view(-1)  # (300*12,)
angles_repeat = angles.tile(NUM_VIEW)
views_repeat = views.repeat_interleave(NUM_ANGLE, dim=0)
grasp_rot = batch_viewpoint_params_to_matrix(-views_repeat, angles_repeat)  # (300, 12, 9)
depths = torch.linspace(0.01, 0.04, 4)
width_bins = torch.tensor([0.02, 0.04, 0.06, 0.08])
# score_bins = torch.tensor([0.2, 0.4, 0.6, 0.8])
score_bins = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def score_unbucketize(bucketized_data):

    # bucket_means = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).to(bucketized_data.device)
    bucket_means = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]).to(bucketized_data.device)
    
    # 为了使用 torch.gather，我们需要确保 bucket_means 在非索引维度上与 bucketized_data 的形状相匹配
    # 首先将 bucket_means 调整为适当的形状
    bucket_means = bucket_means.view(-1, *([1] * (bucketized_data.dim() - 1)))

    # 使用 expand 方法使 bucket_means 在非索引维度上与 bucketized_data 的形状相匹配
    expanded_size = [-1] + list(bucketized_data.shape[1:])
    bucket_means = bucket_means.expand(expanded_size)

    # 使用 torch.gather 在第一个维度上收集数据
    return torch.gather(bucket_means, 0, bucketized_data)


# def generate_half_grasp_views(views, azimuth_range=(0, 2 * np.pi), elev_range=(0, 0.5 * np.pi)):
#     views_constrained = []
#     for pt in views:
#         azimuth = np.math.atan2(pt[1], pt[0])

#         if azimuth < 0:
#             azimuth += 2.0 * np.pi

#         # Elevation from (-0.5 * pi, 0.5 * pi)
#         a = np.linalg.norm(pt)
#         b = np.linalg.norm([pt[0], pt[1], 0])
#         elev = np.math.acos(b / a)
#         if pt[2] < 0:
#             elev = -elev

#         # if hemisphere and (pt[2] < 0 or pt[0] < 0 or pt[1] < 0):
#         if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
#                 elev_range[0] <= elev <= elev_range[1]):
#             continue
#         views_constrained.append(pt)
#     return views_constrained

# views_constrained = np.asarray(generate_half_grasp_views(views.detach().cpu().numpy()))
# views_constrained = torch.from_numpy(views_constrained)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='trunc')
    return tuple(reversed(out))


def ravel_index(index, shape):
    linear_indices = index[:, 0] * shape[1] + index[:, 1]
    return linear_indices


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


def batch_get_depth_inds(pred_depths, gt_depth):
    '''
    **Input:**

    - pred_depths: torch.Tensor of shape (N, 1) for the predicted depths.
    - gt_depth: torch.Tensor of shape (N_d) for depth search space.
    **Output:**

    - gt_indices: torch.Tensor of shape (N,) containing the indices of the nearest ground truth depths.
    '''
    diffs = torch.abs(pred_depths - gt_depth.unsqueeze(0))
    _, gt_indices = torch.min(diffs, dim=1)
    return gt_indices


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
    view_angle_score_inds = view_angle_inds.unsqueeze(0).unsqueeze(-1).expand(len(grasp_scores), -1, -1, 4)
    
    grasp_rot_trans = torch.gather(grasp_rot_trans, 1, view_angle_rot_inds)  # (NUM_VIEWS, NUM_ANGLE, 3, 3)
    grasp_scores = torch.gather(grasp_scores, 2, view_angle_score_inds)
    grasp_widths = torch.gather(grasp_widths, 2, view_angle_score_inds)
    return grasp_rot_trans, grasp_scores, grasp_widths


# v0.4
class ScoringNet(nn.Module):
    def __init__(self, seed_feature_dim, out_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim * 2, 1)
        self.conv3 = nn.Conv1d(self.in_dim * 2, self.out_dim, 1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)
        # self.out_act = nn.Sigmoid()
        self.out_act = nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = self.act(self.conv1(seed_features))
        features = self.act(self.conv2(features))
        score_pred = self.out_act(self.conv3(features))
        # score_pred = self.conv3(features)
        score_pred = score_pred.transpose(1, 2).contiguous() # (B, num_seed, 1)
        end_points['grasp_score_pred'] = score_pred
        return end_points
    


class RotationNet(nn.Module):
    def __init__(self, seed_feature_dim, out_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim * 2, 1)
        self.conv3 = nn.Conv1d(self.in_dim * 2, self.out_dim, 1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = self.act(self.conv1(seed_features))
        features = self.act(self.conv2(features))
        rotation_pred = self.conv3(features)
        rotation_pred = rotation_pred.transpose(1, 2).contiguous() # (B, num_seed, 6)
        end_points['grasp_rot_pred'] = rotation_pred
        return end_points
    
    
class WidthNet(nn.Module):
    def __init__(self, seed_feature_dim, out_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim * 2, 1)
        self.conv3 = nn.Conv1d(self.in_dim * 2, self.out_dim, 1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)
        self.out_act = nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = self.act(self.conv1(seed_features))
        features = self.act(self.conv2(features))
        width_pred = self.out_act(self.conv3(features))
        width_pred = width_pred.transpose(1, 2).contiguous() # (B, num_seed, 1)
        end_points['grasp_width_pred'] = width_pred
        return end_points
    

class DepthNet(nn.Module):
    def __init__(self, seed_feature_dim, out_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim * 2, 1)
        self.conv3 = nn.Conv1d(self.in_dim * 2, self.out_dim, 1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)
        self.out_act = nn.ReLU(inplace=True)
        
    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        features = self.act(self.conv1(seed_features))
        features = self.act(self.conv2(features))
        depth_pred = self.out_act(self.conv3(features))
        depth_pred = depth_pred.transpose(1, 2).contiguous() # (B, num_seed, 1)
        end_points['grasp_depth_pred'] = depth_pred
        return end_points


class IGNet(nn.Module):
    def __init__(self,  num_view=300, num_angle=12, num_depth=4, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = num_depth
        self.num_angle = num_angle
        self.num_view = num_view
        
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        
        # v0.4
        self.scoring_head = ScoringNet(seed_feature_dim=self.seed_feature_dim, out_dim=1)
        self.rotation_head = RotationNet(seed_feature_dim=self.seed_feature_dim, out_dim=6)
        self.width_head = WidthNet(seed_feature_dim=self.seed_feature_dim, out_dim=1)
        self.depth_head = DepthNet(seed_feature_dim=self.seed_feature_dim, out_dim=1)
        
        
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

        end_points['seed_features'] = seed_features
        
        # v0.4
        end_points = self.scoring_head(seed_features, end_points)
        end_points = self.rotation_head(seed_features, end_points)
        end_points = self.width_head(seed_features, end_points)
        end_points = self.depth_head(seed_features, end_points)
        
        if self.is_training:
            end_points = match_grasp_labels(end_points)
            # end_points = match_grasp_view_and_label(end_points)

        return end_points


def match_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['point_clouds']  # (B, M_point, 3)
    # seed_normals = end_points['cloud_normals'] # (B, M_point, 3)
    batch_size, num_pt, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_rots = []
    batch_grasp_depths = []
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
        
        # generate and transform template grasp views
        grasp_views = generate_grasp_views(V).to(seed_xyz.device)  # (V, 3)
        grasp_points_trans = transform_point_cloud(grasp_points, object_pose, '3x4')
        grasp_views_trans = transform_point_cloud(grasp_views, object_pose[:3, :3], '3x3')

        # generate and transform template grasp view rotation
        # angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
        # grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
        # grasp_views_rot_trans = torch.matmul(object_pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

        grasp_rot_trans = torch.matmul(object_pose[:3, :3], grasp_rot.to(seed_xyz.device))  # (V, 3, 3)
        grasp_rot_trans = grasp_rot_trans.view((NUM_VIEW, NUM_ANGLE, 3, 3))
        
        # assign views
        grasp_views_ = grasp_views.unsqueeze(0)
        grasp_views_trans_ = grasp_views_trans.unsqueeze(0)
        _, view_inds, _ = knn_points(grasp_views_, grasp_views_trans_, K=1)
        view_inds = view_inds.squeeze(-1).squeeze(0)
        
        grasp_rot_trans = torch.index_select(grasp_rot_trans, 0, view_inds)  # (V, A, 3, 3)
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
        grasp_rot_trans = grasp_rot_trans.unsqueeze(0).tile((num_pt, 1, 1, 1, 1))
        grasp_scores = torch.index_select(grasp_scores, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths = torch.index_select(grasp_widths, 0, nn_inds)  # (Ns, V, A, D)

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

        pred_grasp_rots = end_points['grasp_rot_pred'][i]
        pred_grasp_depths = end_points['grasp_depth_pred'][i]
        pred_grasp_rots_mat = rotation_6d_to_matrix(pred_grasp_rots)
        
        pred_temp_widths = 0.02 * torch.ones(len(seed_xyz)).to(seed_xyz.device)
        pred_temp_depths = 0.02 * torch.ones(len(seed_xyz)).to(seed_xyz.device)

        pred_key_points, _ = batch_get_key_points(seed_xyz, pred_grasp_rots_mat, pred_temp_widths, pred_temp_depths)
        pred_key_points = pred_key_points.contiguous().view((num_pt, 1, -1))
    
        seed_xyz_ = seed_xyz.unsqueeze(1).tile((1, NUM_VIEW * NUM_ANGLE, 1)).view(-1, 3)
        grasp_rot_trans_ = grasp_rot_trans.view((-1, 3, 3))
        temp_widths = 0.02 * torch.ones(len(grasp_rot_trans_)).to(seed_xyz.device)
        temp_depths = 0.02 * torch.ones(len(grasp_rot_trans_)).to(seed_xyz.device)
        temp_key_points, temp_key_points_sym = batch_get_key_points(seed_xyz_, grasp_rot_trans_, temp_widths,
                                                                    temp_depths)

        temp_key_points = temp_key_points.contiguous().view((num_pt, NUM_VIEW * NUM_ANGLE, -1))
        temp_key_points_sym = temp_key_points_sym.contiguous().view((num_pt, NUM_VIEW * NUM_ANGLE, -1))
        pred_rot_inds = knn_key_points_matching_sym(pred_key_points, temp_key_points, temp_key_points_sym)

        pred_depth_inds = batch_get_depth_inds(pred_grasp_depths.view(-1, 1), depths.to(seed_xyz.device))

        pred_rot_inds = pred_rot_inds.squeeze(-1)
        pred_depth_inds = pred_depth_inds.unsqueeze(-1)
        pred_grasp_inds = ravel_index(torch.concat([pred_rot_inds, pred_depth_inds], dim=1), (NUM_VIEW*NUM_ANGLE, NUM_DEPTH))
        pred_grasp_inds = pred_grasp_inds.unsqueeze(-1)

        gt_widths = torch.gather(grasp_widths.view(num_pt, -1), 1, pred_grasp_inds)
        gt_scores = torch.gather(grasp_scores.view(num_pt, -1), 1, pred_grasp_inds)
        gt_depths = torch.gather(depths.tile((num_pt, 1)).to(seed_xyz.device), 1, pred_depth_inds)
        gt_rot_mats = torch.gather(grasp_rot_trans.view((num_pt, -1, 3, 3)), 1,
                                   pred_rot_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        gt_rot_mats = gt_rot_mats.view((num_pt, 3, 3))
        gt_rots = matrix_to_rotation_6d(gt_rot_mats)
        
        # add to batch
        batch_grasp_points.append(grasp_points_trans)
        # batch_grasp_views_rot.append(grasp_views_rot_trans)
        batch_grasp_rots.append(gt_rots)
        batch_grasp_depths.append(gt_depths)
        batch_grasp_scores.append(gt_scores)
        batch_grasp_widths.append(gt_widths)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    # batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    # batch_grasp_rot_max = torch.stack(batch_grasp_rot_max, 0)
    batch_grasp_rots = torch.stack(batch_grasp_rots, 0)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)
    batch_grasp_depths = torch.stack(batch_grasp_depths, 0)  # (B, Ns, V, A, D)
    
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
    # batch_grasp_scores = batch_grasp_scores[:, :, :, :, :].view(batch_size, num_samples, -1)
    # batch_grasp_widths = batch_grasp_widths[:, :, :, :, :].view(batch_size, num_samples, -1)

    # score_mask = batch_grasp_scores > 0.0
    # batch_grasp_scores[score_mask] = torch.log(1.0 / batch_grasp_scores[score_mask])

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores = 1.1 - batch_grasp_scores
    batch_grasp_scores[~label_mask] = 0
    
    batch_grasp_widths_ids = torch.bucketize(batch_grasp_widths, width_bins.to(batch_grasp_widths.device))
    batch_grasp_scores_ids = torch.bucketize(batch_grasp_scores, score_bins.to(batch_grasp_scores.device))
    
    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rot'] = batch_grasp_rots
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_depth'] = batch_grasp_depths
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
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_depth = end_points['grasp_depth_pred'][i]
        grasp_rot = end_points['grasp_rot_pred'][i]
        grasp_rot_mat = rotation_6d_to_matrix(grasp_rot).view(-1, 9)
        
        if normalize:
            grasp_score = normalize_tensor(grasp_score)
        
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height,
                                      grasp_depth, grasp_rot_mat, grasp_center, obj_ids], axis=-1).detach().cpu().numpy())
        
    return grasp_preds