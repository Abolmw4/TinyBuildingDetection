import logging
import os
import torch.nn as nn

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

