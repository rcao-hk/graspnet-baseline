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


eps = 1e-8
def get_loss(end_points, device):
    graspness_loss, end_points = compute_graspness_loss(end_points, device)
    # graspness_loss, end_points = compute_graspness_loss_view_topk(end_points, device, top_k=60)
    score_loss, end_points = compute_score_loss(end_points, device)
    width_loss, end_points = compute_width_loss(end_points, device)
    # rotation_loss, end_points = compute_rotation_loss(end_points)
    # loss = 10 * score_loss + width_loss
    # loss = 0.1 * score_loss + width_loss  # BCE  bs = 16
    # loss = 10 * score_loss + 0.1*score_oe_loss + width_loss # focal loss  bs = 10
    # loss = 10 * score_loss + 0.1 * width_loss
    # classifcation-based
    # loss = 10 * graspness_loss + 0.5 * score_loss + 0.1 * width_loss
    # regression-based
    loss = 5 * graspness_loss + 5 * score_loss + width_loss  # vanilla
    # loss = 3 * graspness_loss + 5 * score_loss + width_loss
    # loss = 10 * graspness_loss + 10 * score_loss + width_loss
    # score only
    # loss = 5 * graspness_loss + score_loss
    # loss = graspness_loss / (eps + graspness_loss.detach()) + \
    #        score_loss / (eps + score_loss.detach()) + \
    #        width_loss / (eps + width_loss.detach())
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_graspness_loss(end_points, device):
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    grasp_rot_graspness_pred = end_points['grasp_rot_graspness_pred']
    grasp_graspness_label = end_points['batch_grasp_rot_graspness']
    loss = criterion(grasp_rot_graspness_pred, grasp_graspness_label)
    end_points['loss/rot_graspness_loss'] = loss
    return loss, end_points


# def compute_graspness_loss(end_points, device, top_k=720):
#     criterion = nn.SmoothL1Loss(reduction='mean').to(device)

#     pred  = end_points['grasp_rot_graspness_pred']      # (B, Ns, VA)
#     label = end_points['batch_grasp_rot_graspness']     # (B, Ns, VA)
#     assert pred.shape == label.shape, "pred/label shape mismatch"

#     B, Ns, VA = label.shape
#     K = min(top_k, VA)

#     # (B*Ns, VA)
#     pred_f  = pred.reshape(-1, VA)
#     label_f = label.reshape(-1, VA)

#     # 可选：屏蔽无效值，避免 NaN/Inf 影响 topk
#     # valid = torch.isfinite(label_f)
#     # label_for_topk = torch.where(valid, label_f, torch.full_like(label_f, -1e6))
#     label_for_topk = label_f

#     # 取每行 GT 的 Top-K 索引（值越大越好）
#     _, idx = torch.topk(label_for_topk, K, dim=-1, largest=True, sorted=False)  # (B*Ns, K)

#     # 采样对应位置的 pred / label
#     top_pred  = torch.gather(pred_f,  dim=-1, index=idx)     # (B*Ns, K)
#     top_label = torch.gather(label_f, dim=-1, index=idx)     # (B*Ns, K)

#     loss = criterion(top_pred, top_label)

#     end_points['loss/rot_graspness_loss'] = loss
#     return loss, end_points


def compute_graspness_loss_view_topk(end_points, device, top_k=60):
    """
    只对 GT 的 view 方向 graspness Top-K（每个点/每个样本独立）对应的位置回传 loss。
    - pred: (B, Ns, V*A)
    - label: (B, Ns, V*A) —— 由你给的流程得到、已归一化
    - reduce: 'max' 或 'mean'，用于把 A 维聚合为每个 view 的分数
    """
    criterion = nn.SmoothL1Loss(reduction='none').to(device)

    pred  = end_points['grasp_rot_graspness_pred']      # (B, Ns, V*A)
    label = end_points['batch_grasp_rot_graspness']     # (B, Ns, V*A)
    assert pred.shape == label.shape, "pred/label shape mismatch"

    B, Ns, VA = label.shape
    # 需要你在外部知道 V 和 A，或把它们传进来；若不能，且你固定 V=300、A=12：
    V, A = 300, 12
    assert VA == V * A, f"Expected V*A={V*A}, got {VA}"

    # 还原到 (B, Ns, V, A)
    pred_4d  = pred.view(B, Ns, V, A)
    label_4d = label.view(B, Ns, V, A)

    # 计算每个 view 的 GT 分数（沿 A 维聚合）
    view_score = label_4d.mean(dim=-1)        # (B, Ns, V)

    # 若有无效值（例如=0），可避免其进入 topk（可选）
    # 把无效 view 置为极小值，确保不会被选中
    view_score_masked = torch.where(
        torch.isfinite(view_score),
        view_score,
        torch.full_like(view_score, -1e6)
    )

    K = min(top_k, V)
    # 每个 (B, Ns) 选出 view Top-K 索引
    topk_idx = torch.topk(view_score_masked, K, dim=-1, largest=True, sorted=False).indices  # (B, Ns, K)

    # 构造掩码：被选中的 view 的所有 A 桶都参与 loss
    mask_4d = torch.zeros((B, Ns, V, A), dtype=torch.bool, device=pred.device)
    # 将 (B, Ns, K) 扩展到 A 维
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(B, Ns, K, A)  # (B, Ns, K, A)
    # 在 dim=2（V 维）用 scatter_ 置 True
    mask_4d.scatter_(dim=2, index=topk_idx_expanded, value=True)    # (B, Ns, V, A)

    # 计算逐元素 SmoothL1Loss
    elem_loss = criterion(pred_4d, label_4d)  # (B, Ns, V, A)

    # 只保留 Top-K view 的位置
    masked_loss = elem_loss[mask_4d]          # 1D 向量，长度 = B*Ns*K*A

    # 防止空掩码（极端情况下）
    if masked_loss.numel() == 0:
        loss = elem_loss.mean()
    else:
        loss = masked_loss.mean()

    end_points['loss/rot_graspness_loss'] = loss
    return loss, end_points


# def compute_graspness_loss(end_points, device, top_k=60*12):
#     criterion = nn.SmoothL1Loss(reduction='none').to(device)
    
#     # 预测和标签的误差
#     grasp_rot_graspness_pred = end_points['grasp_rot_graspness_pred']
#     grasp_graspness_label = end_points['batch_grasp_rot_graspness']
#     graspness_error = torch.abs(grasp_rot_graspness_pred - grasp_graspness_label)
    
#     # graspness_error 的形状是 (B, Ns, V*A)
#     B, Ns, V_A = graspness_error.size()
    
#     # 计算每个点（每个实例的每个点）在最后一维（V*A）上的 top-k 错误
#     graspness_error_flat = graspness_error.view(B, Ns, -1)  # 将 V*A 拉平为一维
    
#     # 对每个点（每个实例的每个点）选择 top_k 错误
#     top_k_errors, top_k_indices = torch.topk(graspness_error_flat, top_k, dim=-1, largest=False)  # 选取 top-k 错误
    
#     # 创建掩码，标记 top-k 错误的位置
#     loss_mask = torch.zeros_like(graspness_error_flat)  # 初始化一个全零的掩码，形状为 (B, Ns, V*A)
    
#     # 使用 scatter_ 来标记 top-k 错误位置为 1
#     loss_mask = loss_mask.scatter_(2, top_k_indices, 1.0)  # 只标记 top-k 错误的位置
    
#     # 计算掩码后的损失
#     loss = criterion(graspness_error, grasp_graspness_label)
    
#     # 只计算 top-k 错误位置的损失
#     loss = loss[loss_mask.bool()].mean()  
    
#     end_points['loss/rot_graspness_loss'] = loss
#     return loss, end_points


# regression-based
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


# classifcation-based
# def compute_score_loss(end_points, device):
#     criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
#     grasp_score_pred = end_points['grasp_score_pred'].permute(0, 3, 1, 2)
#     grasp_score_label = end_points['batch_grasp_score_ids']
#     loss = criterion(grasp_score_pred, grasp_score_label)
#     end_points['loss/score_loss'] = loss
#     return loss, end_points


# def compute_width_loss(end_points, device):
#     criterion = nn.CrossEntropyLoss(reduction='none').to(device)
#     grasp_width_pred = end_points['grasp_width_pred'].permute(0, 3, 1, 2)
#     grasp_width_label = end_points['batch_grasp_width_ids']
#     loss = criterion(grasp_width_pred, grasp_width_label)
#     grasp_score_label = end_points['batch_grasp_score']
#     loss_mask = grasp_score_label > 0
#     loss = loss[loss_mask].mean()
#     end_points['loss/width_loss'] = loss
#     return loss, end_points
 
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