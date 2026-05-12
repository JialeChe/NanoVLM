"""
NanoVLM 训练配置
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class Stage1Config:
    """Stage 1 训练配置：只训练投影层(MLP)"""
    # 训练轮数
    num_epochs: int = 1
    # 全局 batch size
    per_device_batch_size: int = 1
    # 梯度累积步数（模拟更大batch）
    gradient_accumulation_steps: int = 8
    # 学习率
    learning_rate: float = 2e-3
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    # 预热步数比例
    warmup_ratio: float = 0.03
    # 权重衰减
    weight_decay: float = 0.0
    # 混合精度
    use_fp16: bool = True
    # 每多少步保存一次
    save_steps: int = 5000
    # 每多少步记录日志
    logging_steps: int = 10
    # 最大序列长度 (视觉token + 文本token)
    max_seq_length: int = 2048
    # 训练的数据类型
    torch_dtype: str = "float16"
    # 优化器
    optimizer: str = "adamw"


@dataclass
class Stage2Config:
    """Stage 2 训练配置：训练投影层 + 语言模型(LoRA)"""
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    use_fp16: bool = True
    save_steps: int = 4000
    logging_steps: int = 10
    max_seq_length: int = 2048
    torch_dtype: str = "float16"
    optimizer: str = "adamw"

    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")


@dataclass
class TrainingConfig:
    """总训练配置"""
    # 输出目录
    output_dir: str = "./checkpoints"
    # Stage 1 配置
    stage1: Stage1Config = field(default_factory=Stage1Config)
    # Stage 2 配置
    stage2: Stage2Config = field(default_factory=Stage2Config)
    # 数据配置
    data_path: str = "./data/llava_pretrain/blip_laion_cc_sbu_558k.json"
    # 随机种子
    seed: int = 42
    # TensorBoard 配置
    use_tensorboard: bool = True
    # DeepSpeed 配置 (可选，暂不使用)
    use_deepspeed: bool = False
    # 梯度裁剪
    max_grad_norm: float = 1.0
