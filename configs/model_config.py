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
    # 模型嵌入维度 (Qwen2-0.5B 的 hidden_size)
    hidden_size: int = 1024
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
    # 语言模型嵌入维度
    llm_hidden_size: int = 1024
    # MLP 隐藏层维度
    mlp_hidden_size: int = 2048
    # MLP 层数
    mlp_depth: int = 2
    # 激活函数
    activation: str = "gelu"


@dataclass
class NanoVLMConfig:
    """NanoVLM 总配置"""
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)

    # 特殊 token
    image_token: str = "<image>"
    image_token_id: Optional[int] = None  # 将在模型加载时设置

    def __post_init__(self):
        """初始化后验证配置一致性"""
        assert self.connector.connector_type == "mlp", \
            f"Unsupported connector type: {self.connector.connector_type}"
        assert self.connector.mlp_depth >= 1, \
            f"MLP depth must be >= 1, got {self.connector.mlp_depth}"