import logging
import os
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def setup_logger(name='my_logger', log_file='/home/my_proj/logs/app.log', level=logging.DEBUG):
    # Create custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs if called multiple times

    # Formatter: includes file name, line number, level, and message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_layer_wise_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        print(f"{name:30} | {param.numel():7} params | requires_grad={param.requires_grad}")

def active_trainable_params(model):
    for param in model.parameters():
        param.requires_grad = True  # Enable gradients for training

def deactive_trainable_params(model):
    for param in model.parameters():
        param.requires_grad = True  # Enable gradients for training

def reshape_output(y, num_anchors=8, num_bins=4):
    batch, _, H, W = y.shape
    y = y.view(batch, num_anchors, -1, H, W)
    box_pred = y[:, :, :4 * num_bins, :, :]      # (batch, 8, 16, H, W)
    obj_pred = y[:, :, 4 * num_bins:4 * num_bins + 1, :, :]  # (batch, 8, 1, H, W)
    cls_pred = y[:, :, 4 * num_bins + 1:, :, :]  # (batch, 8, 1, H, W)
    return box_pred, obj_pred, cls_pred

def build_targets(targets, H, W, num_anchors=8):
    batch_size = len(targets)
    obj_target = torch.zeros(batch_size, num_anchors, H, W)
    cls_target = torch.zeros(batch_size, num_anchors, H, W)
    box_target = torch.zeros(batch_size, num_anchors, 4, H, W)

    for b in range(batch_size):
        for class_id, cx, cy, w, h in [targets[b]]:
            # class_id, cx, cy, w, h = obj
            # Calculate grid cell index
            i = min(int(cx * W), W - 1)
            j = min(int(cy * H), H - 1)

            # Mark all anchors in this cell as positive
            obj_target[b, :, j, i] = 1.0
            cls_target[b, :, j, i] = 1.0

            # Calculate box distances
            center_x = (i + 0.5) / W
            center_y = (j + 0.5) / H
            l_dist = (center_x - (cx - w / 2)) * W
            t_dist = (center_y - (cy - h / 2)) * H
            r_dist = ((cx + w / 2) - center_x) * W
            b_dist = ((cy + h / 2) - center_y) * H
            box_target[b, :, 0, j, i] = l_dist
            box_target[b, :, 1, j, i] = t_dist
            box_target[b, :, 2, j, i] = r_dist
            box_target[b, :, 3, j, i] = b_dist

    return obj_target, cls_target, box_target


def read_lables_file(lbl: str) -> np.ndarray:
    data = []
    # Read file using with open()
    with open(lbl, 'r') as file:
        for line in file:
            # Split each line into individual string values
            str_values = line.split()

            # Convert string values to floats
            float_values = [float(x) for x in str_values]

            # Append to data list
            data.append(float_values)

    # Convert to NumPy array
    arr = np.array(data)
    return arr

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y