"""
工具函数集合
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dtype(dtype_str: str) -> torch.dtype:
    """字符串→torch dtype"""
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float16)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计参数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num: int) -> str:
    """友好显示参数量"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """获取设备"""
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def freeze_module(module: torch.nn.Module):
    """冻结模块参数"""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    """解冻模块参数"""
    module.train()
    for param in module.parameters():
        param.requires_grad = True


def print_model_info(model: torch.nn.Module, prefix: str = ""):
    """打印模型信息"""
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model)
    print(f"{prefix}Trainable: {format_params(trainable)}")
    print(f"{prefix}Total:     {format_params(total)}")
    print(f"{prefix}Ratio:     {100*trainable/total:.2f}%")