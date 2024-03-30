import torch.nn as nn
import torch
# from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import rotation_6d_to_matrix
# from kornia.losses.focal import FocalLoss
from utils.loss_utils import batch_get_key_points
from models.ordinal_entropy_loss import ordinalentropy
from models.coral_loss import corn_loss
TOP_K = 300


def sym_loss(p1_key_points, p2_key_points, p2_key_points_sym):
    criterion = nn.MSELoss(reduction='none')
    dis = criterion(p1_key_points, p2_key_points)
    dis_sym = criterion(p1_key_points, p2_key_points_sym)
    sym_mask = torch.lt(dis, dis_sym)
    dis = dis * sym_mask + dis_sym * (~sym_mask)
    return dis


def get_loss(end_points, device):
    # score_loss, score_oe_loss, end_points = compute_score_loss(end_points, device)
    score_loss, end_points = compute_score_loss(end_points, device)
    width_loss, end_points = compute_width_loss(end_points, device)
    # rotation_loss, end_points = compute_rotation_loss(end_points)
    # loss = 10 * score_loss + width_loss
    # loss = 0.1 * score_loss + width_loss  # BCE  bs = 16
    # loss = 10 * score_loss + 0.1*score_oe_loss + width_loss # focal loss  bs = 10
    loss = 10 * score_loss + 0.1 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


# regression-based loss
# def compute_score_loss(end_points, device):
#     criterion = nn.SmoothL1Loss(reduction='mean').to(device)
#     # criterion = nn.MSELoss(reduction='mean').to(device)
#     grasp_score_pred = end_points['grasp_score_pred']
#     grasp_score_label = end_points['batch_grasp_score']
#     loss = criterion(grasp_score_pred, grasp_score_label)
#     end_points['loss/score_loss'] = loss
#     return loss, end_points


# regression-based with oridinal loss
# def compute_score_loss(end_points, device):
#     # criterion = nn.SmoothL1Loss(reduction='mean').to(device)
#     criterion = nn.MSELoss(reduction='mean').to(device)
#     grasp_score_pred = end_points['grasp_score_pred']
#     grasp_score_label = end_points['batch_grasp_score']
#     loss = criterion(grasp_score_pred, grasp_score_label)
#     oe_loss = ordinalentropy(end_points['seed_features'], grasp_score_label)
#     end_points['loss/score_loss'] = loss
#     end_points['loss/score_oe_loss'] = oe_loss
#     return loss ,oe_loss, end_points


# classifcation-based
# def compute_score_loss(end_points, device):
#     # criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1).to(device)
#     # criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean').to(device)
#     grasp_score_pred = end_points['grasp_score_pred'].permute((0, 3, 1, 2))
#     grasp_score_label = end_points['batch_grasp_score_ids']
#     loss = corn_loss(grasp_score_pred, grasp_score_label, 10)
#     # loss = criterion(grasp_score_pred, grasp_score_label)
#     # grasp_score_label = end_points['batch_grasp_score']
#     # loss_mask = grasp_score_label > 0
#     # loss = loss[loss_mask].mean()
#     end_points['loss/score_loss'] = loss
#     return loss, end_points


# regression-based
# def compute_width_loss(end_points, device):
#     criterion = nn.SmoothL1Loss(reduction='none').to(device)
#     # criterion = nn.MSELoss(reduction='none').to(device)
#     grasp_width_pred = end_points['grasp_width_pred']
#     grasp_width_label = end_points['batch_grasp_width'] * 10
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     grasp_score_label = end_points['batch_grasp_score']
#     loss_mask = grasp_score_label > 0
#     loss = loss[loss_mask].mean()
#     end_points['loss/width_loss'] = loss
#     return loss, end_points


# v0.5
# def compute_width_loss(end_points, device):
#     criterion = nn.SmoothL1Loss().to(device)
#     # criterion = nn.MSELoss(reduction='none').to(device)
#     grasp_width_pred = end_points['grasp_width_pred']
#     grasp_width_label = end_points['batch_grasp_width'] * 10
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     end_points['loss/width_loss'] = loss
#     return loss, end_points


# classifcation-based
# def compute_width_loss(end_points):
#     criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
#     grasp_width_pred = end_points['grasp_width_pred'].permute((0, 3, 1, 2))
#     grasp_width_label = end_points['batch_grasp_width_ids']
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     grasp_score_label = end_points['batch_grasp_score']
#     loss_mask = grasp_score_label > 0
#     loss = loss[loss_mask].mean()
#     end_points['loss/width_loss'] = loss
#     return loss, end_points


# v0.3.7
def compute_score_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    # criterion = nn.MSELoss(reduction='mean').to(device)
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)
    end_points['loss/score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points, device):
    criterion = nn.SmoothL1Loss().to(device)
    # criterion = nn.MSELoss(reduction='none').to(device)
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    end_points['loss/width_loss'] = loss
    return loss, end_points

# v0.4
# def compute_score_loss(end_points, device):
#     criterion = nn.MSELoss(reduction='mean').to(device)
#     grasp_score_pred = end_points['grasp_score_pred']
#     grasp_score_label = end_points['batch_grasp_score']
#     loss = criterion(grasp_score_pred, grasp_score_label)
#     end_points['loss/score_loss'] = loss
#     return loss, end_points


# def compute_width_loss(end_points, device):
#     criterion = nn.MSELoss(reduction='none').to(device)
#     grasp_width_pred = end_points['grasp_width_pred']
#     grasp_width_label = end_points['batch_grasp_width'] * 10
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     grasp_score_label = end_points['batch_grasp_score']
#     loss_mask = grasp_score_label > 0
#     loss = loss[loss_mask].mean()
#     end_points['loss/width_loss'] = loss
#     return loss, end_points


# def compute_rotation_loss(end_points, device):
#     criterion = nn.MSELoss(reduction='none').to(device)
#     grasp_rot_pred = end_points['grasp_rot_pred']
#     grasp_rot_label = end_points['batch_grasp_rot']
#     loss = criterion(grasp_rot_pred, grasp_rot_label)
#     grasp_score_label = end_points['batch_grasp_score']
#     loss_mask = grasp_score_label > 0
#     loss = loss[loss_mask.expand(-1, -1, 6)].mean()
#     end_points['loss/rot_loss'] = loss
#     return loss, end_points


# 0.4.2
# def get_loss(end_points, device):
#     score_loss, end_points = compute_score_loss(end_points, device)
#     rotation_loss, end_points = compute_rotation_loss(end_points, device)
#     width_loss, end_points = compute_width_loss(end_points, device)
#     loss = score_loss + rotation_loss + width_loss
#     end_points['loss/overall_loss'] = loss
#     return loss, end_points


# 0.4.3
# def get_loss(end_points, device):
#     score_loss, end_points = compute_score_loss(end_points, device)
#     geo_loss, end_points = compute_geo_loss(end_points)
#     # rotation_loss, end_points = compute_rotation_loss(end_points, device)
#     # width_loss, end_points = compute_width_loss(end_points, device)
#     loss = score_loss + 1000 * geo_loss
#     end_points['loss/overall_loss'] = loss
#     return loss, end_points


# def compute_geo_loss(end_points):
#     grasp_rot_pred = end_points['grasp_rot_pred']
#     grasp_rot_label = end_points['batch_grasp_rot']
    
#     grasp_rot_pred_mat = rotation_6d_to_matrix(grasp_rot_pred)
#     grasp_rot_label_mat = rotation_6d_to_matrix(grasp_rot_label)
    
#     grasp_points = end_points['point_clouds']
#     grasp_width_pred = end_points['grasp_width_pred']
#     grasp_width_label = end_points['batch_grasp_width']
    
#     grasp_depth_pred = end_points['grasp_depth_pred']
#     grasp_depth_label = end_points['batch_grasp_depth']
    
#     # temp_widths = 0.02 * torch.ones(len(grasp_points)).to(grasp_points.device)
#     # temp_depths = 0.02 * torch.ones(len(grasp_points)).to(grasp_points.device)

#     pred_key_points, _ = batch_get_key_points(grasp_points, grasp_rot_pred_mat, grasp_width_pred, grasp_depth_pred)
#     gt_key_points, gt_key_points_sym = batch_get_key_points(grasp_points, grasp_rot_label_mat, grasp_width_label, grasp_depth_label)
    
#     loss = sym_loss(pred_key_points, gt_key_points, gt_key_points_sym)
#     loss = loss.sum(dim=-1)
#     # loss = end_points['grasp_score_pred'] * loss.mean(dim=-1, keepdim=True)
#     loss = end_points['batch_grasp_score'] * loss.mean(dim=-1, keepdim=True)
#     loss = loss.mean()
#     # grasp_score_label = end_points['batch_grasp_score']
#     # loss_mask = grasp_score_label > 0
#     # loss_mask = loss_mask.unsqueeze(-1).expand(-1, -1, 4, 3)
#     # loss = loss[loss_mask].mean()
#     end_points['loss/geo_loss'] = loss
#     return loss, end_points