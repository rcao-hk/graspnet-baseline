import torch.nn as nn
import torch

eps = 1e-6
def get_loss(end_points, device):
    objectness_loss, end_points = compute_objectness_loss(end_points, device)
    graspness_loss, end_points = compute_graspness_loss(end_points, device)
    # graspness_loss, end_points = compute_graspness_loss_view_topk(end_points, device, top_k=60)
    score_loss, end_points = compute_score_loss(end_points, device)
    width_loss, end_points = compute_width_loss(end_points, device)
    # rotation_loss, end_points = compute_rotation_loss(end_points)

    grasp_loss = 5 * graspness_loss + 5 * score_loss + width_loss  # vanilla
    loss = objectness_loss + grasp_loss  # vanilla
    # score only
    # loss = 5 * graspness_loss + score_loss
    # loss = graspness_loss / (eps + graspness_loss.detach()) + \
    #        score_loss / (eps + score_loss.detach()) + \
    #        width_loss / (eps + width_loss.detach())
    end_points['loss/overall_loss'] = loss
    end_points['loss/grasp_loss'] = grasp_loss
    return loss, end_points


# def compute_objectness_loss(end_points, device):
#     criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
#     objectness_score = end_points['objectness_score']
#     objectness_label = end_points['objectness_label']
#     loss = criterion(objectness_score, objectness_label)
#     end_points['Objectness Loss'] = loss

#     objectness_pred = torch.argmax(objectness_score, 1)
#     end_points['Objectness Acc'] = (objectness_pred == objectness_label.long()).float().mean()
#     return loss, end_points


def compute_objectness_loss(end_points, device):
    """
    end_points needs:
      - objectness_score: (B, 2, N) logits   或 (B, N, 2)
      - objectness_label: (B, N) with {0,1}
    """
    logits = end_points["objectness_score"]
    labels = end_points.get("objectness_label_sel", end_points["objectness_label"]).long()

    # 兼容 (B,N,2)
    if logits.dim() == 3 and logits.size(-1) == 2 and logits.size(1) != 2:
        logits = logits.permute(0, 2, 1).contiguous()  # -> (B,2,N)

    # shape check
    assert logits.dim() == 3 and logits.size(1) == 2, f"objectness_score should be (B,2,N), got {tuple(logits.shape)}"
    assert labels.shape == (logits.size(0), logits.size(2)), f"objectness_label shape mismatch: {labels.shape} vs {(logits.size(0), logits.size(2))}"

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    loss = criterion(logits, labels)

    # metrics（全部保持 tensor，避免 train.py 里 .item 崩）
    with torch.no_grad():
        pred = logits.argmax(dim=1)  # (B,N)
        acc = (pred == labels).float().mean()

        tp = ((pred == 1) & (labels == 1)).sum().float()
        fp = ((pred == 1) & (labels == 0)).sum().float()
        fn = ((pred == 0) & (labels == 1)).sum().float()
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)

        end_points["acc/objectness_acc"] = acc
        end_points["prec/objectness_prec"] = prec
        end_points["recall/objectness_recall"] = rec
        end_points["count/object_pos_ratio"] = (labels == 1).float().mean()

    end_points["loss/objectness_loss"] = loss
    return loss, end_points

# def compute_graspness_loss(end_points, device):
#     criterion = nn.SmoothL1Loss(reduction='mean').to(device)
#     grasp_rot_graspness_pred = end_points['grasp_rot_graspness_pred']
#     grasp_graspness_label = end_points['batch_grasp_rot_graspness']
#     loss = criterion(grasp_rot_graspness_pred, grasp_graspness_label)
#     end_points['loss/rot_graspness_loss'] = loss
#     return loss, end_points


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