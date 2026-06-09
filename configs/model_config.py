"""
NanoVLM 模型配置
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionConfig:
    """视觉编码器配置"""
    # HuggingFace 模型 ID
    model_name_or_path: str = "./models/siglip-so400m-patch14-384"
    # 本地缓存路径 (下载到项目 models/ 目录下)
    local_cache_dir: str = "./models/siglip-so400m-patch14-384"
    # 输入图像尺寸
    image_size: int = 384
    # 是否冻结参数（训练 Stage 1 和 Stage 2 均冻结）
    freeze: bool = True


@dataclass
class LanguageConfig:
    """语言模型配置"""
    # HuggingFace 模型 ID
    model_name_or_path: str = "./models/Qwen2-0.5B-Instruct"
    # 本地缓存路径 (下载到项目 models/ 目录下)
    local_cache_dir: str = "./models/Qwen2-0.5B-Instruct"
    # 模型嵌入维度 (Qwen2-0.5B-Instruct 的实际 hidden_size = 896)
    hidden_size: int = 896
    # Stage 1 是否冻结
    freeze_stage1: bool = True
    # Stage 2 使用 LoRA 微调
    use_lora_stage2: bool = True


@dataclass
class ConnectorConfig:
    """跨模态连接器配置"""
    connector_type: str = "mlp"  # 仅支持 "mlp" 类型
    # 视觉编码器输出维度 (SigLIP so400m: 1152)
    vision_hidden_size: int = 1152
    # 语言模型嵌入维度 (Qwen2-0.5B 实际 hidden_size=896，运行时自动修正)
    llm_hidden_size: int = 896
    # MLP 隐藏层维度
    mlp_hidden_size: int = 2048
    # MLP 层数
    mlp_depth: int = 2
    # 激活函数
    activation: str = "gelu"


@dataclass
class AnyResConfig:
    """AnyRes 动态高分辨率配置（LLaVA-NeXT 1.6）

    可通过 enabled=False 完全回退到原始单图模式，保留学习路线。
    """
    # 是否启用 AnyRes。False 时等价于原始 LLaVA 1.0/1.5 单图模式。
    enabled: bool = False

    # 每个 tile 的基准分辨率（与 VisionEncoder 的 image_size 一致）
    base_size: int = 384

    # 候选 grid 配置 (rows, cols)。
    # (1,1) = 缩略图模式（等价于原始单图），(2,2) = 4 tile，(3,3) = 9 tile
    grid_configs: tuple = (
        (1, 1),
        (1, 2), (2, 1),
        (2, 2),
        (1, 3), (3, 1),
        (2, 3), (3, 2),
        (3, 3),
    )

    # 训练/推理时限制最大 tile 数（不含缩略图），控制显存
    max_tiles: int = 4


@dataclass
class NanoVLMConfig:
    """NanoVLM 总配置"""
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    anyres: AnyResConfig = field(default_factory=AnyResConfig)

    # 特殊 token
    image_token: str = "<image>"
    image_token_id: Optional[int] = None  # 将在模型加载时设置

    def __post_init__(self):
        """初始化后验证配置一致性"""
        assert self.connector.connector_type == "mlp", \
            f"Unsupported connector type: {self.connector.connector_type}"
        assert self.connector.mlp_depth >= 1, \
            f"MLP depth must be >= 1, got {self.connector.mlp_depth}"