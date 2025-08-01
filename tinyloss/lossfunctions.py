import torch
import torch.nn.functional as F
from utils.util import reshape_output, build_targets

def obj_loss(obj_pred, obj_target):
    return F.binary_cross_entropy_with_logits(
        obj_pred.squeeze(2),  # (batch, 8, H, W)
        obj_target.squeeze(2),           # (batch, 8, H, W)
        reduction='mean'
    )


def cls_loss(cls_pred, cls_target, obj_target):
    pos_mask = obj_target > 0.5
    loss = F.binary_cross_entropy_with_logits(
        cls_pred.squeeze(2),  # (batch, 8, H, W)
        cls_target,           # (batch, 8, H, W)
        reduction='none'
    )

    return (loss * pos_mask).sum() / max(1, pos_mask.sum())


def ciou_loss(pred_boxes, gt_boxes):
    # pred_boxes, gt_boxes: (batch, 8, 4, H, W) as (l, t, r, b)
    # Convert to (x1, y1, x2, y2) format
    pred_x1 = pred_boxes[:, :, 0:1, :, :]
    pred_y1 = pred_boxes[:, :, 1:2, :, :]
    pred_x2 = pred_boxes[:, :, 2:3, :, :]
    pred_y2 = pred_boxes[:, :, 3:4, :, :]
    gt_x1 = gt_boxes[:, :, 0:1, :, :]
    gt_y1 = gt_boxes[:, :, 1:2, :, :]
    gt_x2 = gt_boxes[:, :, 2:3, :, :]
    gt_y2 = gt_boxes[:, :, 3:4, :, :]

    # Calculate intersection and union
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, 0) * torch.clamp(inter_y2 - inter_y1, 0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / (union_area + 1e-7)

    # Center distance
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    gt_center_x = (gt_x1 + gt_x2) / 2
    gt_center_y = (gt_y1 + gt_y2) / 2
    center_dist = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2

    # Enclosing box diagonal
    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_y1 = torch.min(pred_y1, gt_y1)
    enclose_x2 = torch.max(pred_x2, gt_x2)
    enclose_y2 = torch.max(pred_y2, gt_y2)
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan((gt_x2 - gt_x1) / (gt_y2 - gt_y1 + 1e-7)) -
                                          torch.atan((pred_x2 - pred_x1) / (pred_y2 - pred_y1 + 1e-7)), 2)
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist / (enclose_diag + 1e-7)) - alpha * v
    return 1 - ciou


def dfl_loss(box_pred, box_target, num_bins=4):
    # box_pred: (batch, 8, 4, num_bins, H, W)
    # box_target: (batch, 8, 4, H, W)
    bin_indices = torch.arange(num_bins, device=box_pred.device)
    bin_indices = bin_indices.view(1, 1, 1, num_bins, 1, 1)

    # Softmax over bins and compute expected value
    probs = F.softmax(box_pred, dim=3)
    coord_pred = (probs * bin_indices).sum(dim=3)  # (batch, 8, 4, H, W)

    # DFL: Match target to nearest bins
    target_val = box_target.unsqueeze(3)  # (batch, 8, 4, 1, H, W)
    floor_val = torch.floor(target_val).long()
    ceil_val = torch.ceil(target_val).long()
    floor_val = torch.clamp(floor_val, 0, num_bins - 1)
    ceil_val = torch.clamp(ceil_val, 0, num_bins - 1)

    # Weights for bin targets
    floor_weight = ceil_val - target_val
    ceil_weight = target_val - floor_val

    # Gather log-probabilities for the bins
    log_probs = F.log_softmax(box_pred, dim=3)
    floor_log_prob = torch.gather(log_probs, 3, floor_val)
    ceil_log_prob = torch.gather(log_probs, 3, ceil_val)

    loss = - (floor_weight * floor_log_prob + ceil_weight * ceil_log_prob)
    return loss.mean(dim=2, keepdim=True)  # Average over coordinates


def box_loss(box_pred, box_target, H, W, stride):
    # Convert predictions to boxes
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
    center_x = (grid_x.unsqueeze(0).unsqueeze(0) + 0.5) * stride / (W * stride)
    center_y = (grid_y.unsqueeze(0).unsqueeze(0) + 0.5) * stride / (H * stride)

    # Compute l, t, r, b from DFL
    bin_probs = F.softmax(box_pred.view(*box_pred.shape[:3], 4, H, W), dim=3)
    coord_vals = (bin_probs * torch.arange(4, device=box_pred.device).view(1, 1, 1, 4, 1, 1)).sum(dim=3)
    l_pred, t_pred, r_pred, b_pred = coord_vals.chunk(4, dim=2)

    # Predicted boxes (normalized)
    left_pred = center_x - l_pred.squeeze(2) / W
    top_pred = center_y - t_pred.squeeze(2) / H
    right_pred = center_x + r_pred.squeeze(2) / W
    bottom_pred = center_y + b_pred.squeeze(2) / H
    pred_boxes = torch.stack([left_pred, top_pred, right_pred, bottom_pred], dim=2)

    # GT boxes (from distances)
    l_gt, t_gt, r_gt, b_gt = box_target.chunk(4, dim=2)
    left_gt = center_x - l_gt / W
    top_gt = center_y - t_gt / H
    right_gt = center_x + r_gt / W
    bottom_gt = center_y + b_gt / H
    gt_boxes = torch.stack([left_gt, top_gt, right_gt, bottom_gt], dim=2)

    ciou = ciou_loss(pred_boxes, gt_boxes)
    dfl = dfl_loss(box_pred.view(*box_pred.shape[:3], 4, H, W), box_target)
    return ciou.mean() + dfl.mean()


def total_loss(outputs, targets, img_size=640):
    scales = [(80, 8), (40, 16), (20, 32)]  # (grid_size, stride)
    total_obj, total_cls, total_box = 0, 0, 0

    for (H, W), stride, out in zip(scales, [8, 16, 32], outputs):
        box_pred, obj_pred, cls_pred = reshape_output(out, num_anchors=8)
        obj_t, cls_t, box_t = build_targets(targets, H, W, stride)

        # Move targets to device
        obj_t = obj_t.to(out.device)
        cls_t = cls_t.to(out.device)
        box_t = box_t.to(out.device)

        total_obj += obj_loss(obj_pred, obj_t)
        total_cls += cls_loss(cls_pred, cls_t, obj_t)
        total_box += box_loss(box_pred, box_t, H, W, stride)

    return total_obj + total_cls + total_box


