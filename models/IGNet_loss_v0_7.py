import torch.nn as nn
import torch


def get_loss(end_points, device):
    view_loss, end_points = compute_view_graspness_loss(end_points, device)
    score_loss, end_points = compute_score_loss(end_points, device)
    width_loss, end_points = compute_width_loss(end_points, device)
    # loss = 10 * view_loss + 5 * score_loss + width_loss
    loss = 10 * view_loss + 0.5 * score_loss + 0.1 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    view_score = end_points['grasp_view_graspness_pred']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/view_graspness_loss'] = loss
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