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
    graspness_loss, end_points = compute_graspness_loss(end_points, device)
    score_loss, end_points = compute_score_loss(end_points, device)
    width_loss, end_points = compute_width_loss(end_points, device)
    # rotation_loss, end_points = compute_rotation_loss(end_points)
    # loss = 10 * score_loss + width_loss
    # loss = 0.1 * score_loss + width_loss  # BCE  bs = 16
    # loss = 10 * score_loss + 0.1*score_oe_loss + width_loss # focal loss  bs = 10
    # loss = 10 * score_loss + 0.1 * width_loss
    loss = 5 * graspness_loss + 5 * score_loss + width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_graspness_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    grasp_rot_graspness_pred = end_points['grasp_rot_graspness_pred']
    grasp_graspness_label = end_points['batch_grasp_rot_graspness']
    loss = criterion(grasp_rot_graspness_pred, grasp_graspness_label)
    end_points['loss/rot_graspness_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)
    end_points['loss/score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='none').to(device)
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/width_loss'] = loss
    return loss, end_points


# v0.5
# def compute_width_loss(end_points, device):
#     criterion = nn.SmoothL1Loss(reduction='none').to(device)
#     # criterion = nn.MSELoss(reduction='none').to(device)
#     grasp_width_pred = end_points['grasp_width_pred']
#     grasp_width_label = end_points['batch_grasp_width'] * 10
#     grasp_score_mask = end_points['batch_grasp_mask']
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     loss = loss[grasp_score_mask].mean()
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