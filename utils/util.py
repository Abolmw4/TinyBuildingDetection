import json
import logging
import os
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import amp
from tinyloss.lossfunctions import *

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

def load_json(json_dir: str="/home/my_proj/conf/config.json"):
    with open(json_dir, mode='r') as jsfile:
        config = json.load(jsfile)
    return config

def check_device(device: str) -> str:
    return "cuda" if (device == "cuda" or device == "CUDA") and torch.cuda.is_available() else "cpu"

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


def train_tinynet(model, data_loader=None, optimizer=None, dev="cuda"):
    down_factor = 2
    total_ls, obj_ls, cls_ls, box_ls, sr_ls = 0, 0, 0, 0, 0
    for i, (imgs, targets, path, _) in enumerate(tqdm(data_loader)):
        # Get the actual batch size
        batch_size = imgs.size(0)

        # Move data to device
        image = imgs.to(dev, non_blocking=True) / 255.0
        imgs = F.interpolate(image, size=[i // down_factor for i in image.size()[2:]], mode='bilinear', align_corners=True)

        optimizer.zero_grad()
        pred, output_sr = model(imgs)

        # Initialize batch losses
        batch_obj_loss = 0
        batch_cls_loss = 0
        batch_box_loss = 0
        sr_loss = 0
        for item, stride in zip(pred, [8, 16, 32]):
            batch, _, h, w = item.shape
            box_pred, obj_pred, cls_pred = reshape_output(item)

            # Pass the actual batch size to build_targets
            obj_target, cls_target, box_target = build_targets(
                targets=targets, H=h, W=w, batch_size=batch_size
            )

            # Move targets to device
            obj_target = obj_target.to(dev, non_blocking=True)
            cls_target = cls_target.to(dev, non_blocking=True)
            box_target = box_target.to(dev, non_blocking=True)

            # Calculate losses
            batch_obj_loss += obj_loss(obj_pred, obj_target)
            batch_cls_loss += cls_loss(cls_pred, cls_target, obj_target)
            batch_box_loss += box_loss(box_pred, box_target, h, w, stride)

        sr_loss += 0.1*torch.nn.L1Loss(reduction='mean')(output_sr, image)

        # Combine losses and backpropagate
        total_batch_loss = batch_obj_loss + batch_cls_loss + batch_box_loss + sr_loss
        total_batch_loss.backward()
        optimizer.step()

        # Accumulate detached losses for logging
        total_ls += total_batch_loss.detach().item()
        obj_ls += batch_obj_loss.detach().item()
        cls_ls += batch_cls_loss.detach().item()
        box_ls += batch_box_loss.detach().item()
        sr_ls += sr_loss.detach().item()

        # Clear intermediate variables
        del pred, output_sr, box_pred, obj_pred, cls_pred
        torch.cuda.empty_cache()
    return total_ls, obj_ls, cls_ls, box_ls, sr_ls


def validation_tinynet(model, device: str="cuda", data_loader=None):

    model.eval()
    down_factor = 2
    with torch.no_grad():
        for i, (imgs, targets, path, _) in enumerate(tqdm(data_loader)):
            # Get the actual batch size
            batch_size = imgs.size(0)

            # Move data to device
            image = imgs.to(device, non_blocking=True) / 255.0
            imgs = F.interpolate(image, size=[i // down_factor for i in image.size()[2:]], mode='bilinear',
                                     align_corners=True)

            pred, output_sr = model(imgs)

            # Initialize batch losses
            batch_obj_loss_val = 0
            batch_cls_loss_val = 0
            batch_box_loss_val = 0
            sr_loss_val = 0
            for item, stride in zip(pred, [8, 16, 32]):
                batch, _, h, w = item.shape
                box_pred, obj_pred, cls_pred = reshape_output(item)

                # Pass the actual batch size to build_targets
                obj_target, cls_target, box_target = build_targets(
                        targets=targets, H=h, W=w, batch_size=batch_size
                    )

                # Move targets to device
                obj_target = obj_target.to(device, non_blocking=True)
                cls_target = cls_target.to(device, non_blocking=True)
                box_target = box_target.to(device, non_blocking=True)

                # Calculate losses
                batch_obj_loss_val += obj_loss(obj_pred, obj_target)
                batch_cls_loss_val += cls_loss(cls_pred, cls_target, obj_target)
                batch_box_loss_val += box_loss(box_pred, box_target, h, w, stride)

            sr_loss_val += 0.1 * torch.nn.L1Loss(reduction='mean')(output_sr, image)

