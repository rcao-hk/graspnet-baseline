import torch.nn as nn
import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import rotation_6d_to_matrix
from kornia.losses.focal import FocalLoss
from loss_utils import batch_key_points
from models.ordinal_entropy_loss import ordinalentropy
from models.coral_loss import corn_loss


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
    loss = 10 * score_loss + width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


# regression-based loss
def compute_score_loss(end_points, device):
    # criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    criterion = nn.MSELoss(reduction='mean').to(device)
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)
    end_points['loss/score_loss'] = loss
    return loss, end_points


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
def compute_width_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='none').to(device)
    # criterion = nn.MSELoss(reduction='none').to(device)
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/width_loss'] = loss
    return loss, end_points


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


# v0.4
def compute_rotation_loss(end_points):
    grasp_rot_pred = end_points['grasp_rot_pred']
    grasp_rot_mat_pred = rotation_6d_to_matrix(grasp_rot_pred)
    grasp_rot_max = end_points['batch_grasp_rot_max']
    grasp_points = end_points['point_clouds']
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width']
    
    pred_key_points, _ = batch_key_points(grasp_points, grasp_rot_mat_pred, grasp_width_pred)
    gt_key_points, gt_key_points_sym = batch_key_points(grasp_points, grasp_rot_max, grasp_width_label)
    loss = sym_loss(pred_key_points, gt_key_points, gt_key_points_sym)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss_mask = loss_mask.unsqueeze(-1).expand(-1, -1, 5, 3)
    loss = loss[loss_mask].mean()
    end_points['loss/rotation_loss'] = loss
    return loss, end_points