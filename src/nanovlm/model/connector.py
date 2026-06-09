"""
跨模态连接器（MLP Projector）
- 这是VLM最核心的组件之一
- 作用：将视觉特征向量投影到语言模型的embedding空间
- 2层MLP + GELU激活，简单高效
"""

from typing import Optional
import torch
import torch.nn as nn
from configs.model_config import ConnectorConfig


class Connector(nn.Module):
    """
    MLP 投影连接器

    输入:  (B, num_vision_tokens, vision_hidden_size)  # CLIP输出: 1024维
    输出:  (B, num_vision_tokens, llm_hidden_size)      # Qwen输入: 1536维

    架构:
        Linear(1024 → 2048) → GELU → Linear(2048 → 1536)
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.config = config

        # 构建MLP层
        layers = []
        input_dim = config.vision_hidden_size  # 1024

        for i in range(config.mlp_depth):
            if i < config.mlp_depth - 1:
                # 隐藏层: input_dim → mlp_hidden_size
                layers.append(nn.Linear(input_dim, config.mlp_hidden_size))
                input_dim = config.mlp_hidden_size
            else:
                # 输出层: mlp_hidden_size → llm_hidden_size
                layers.append(nn.Linear(input_dim, config.llm_hidden_size))

            # 每层后面加激活（除最后一层外）
            if i < config.mlp_depth - 1:
                if config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported activation: {config.activation}")

        self.mlp = nn.Sequential(*layers)

        # 输出归一化：将视觉 embedding 约束到与 LLM token embedding 相同的量级
        # Qwen2 内部使用 RMSNorm，这里保持一致。没有这个层，Connector 输出可能
        # 达到 ±5700，远超 token embedding 的 ±1 范围，导致 FP16 attention 溢出。
        self.norm = nn.RMSNorm(config.llm_hidden_size, eps=1e-6)

        # 初始化权重（xavier正态分布）
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, num_vision_tokens, vision_hidden_size)

        Returns:
            language_embeddings: (B, num_vision_tokens, llm_hidden_size)
                MLP 投影 → RMSNorm → 输出，值域与 LLM token embedding 一致
        """
        x = self.mlp(vision_features)
        x = self.norm(x)
        return x

    def get_output_dim(self) -> int:
        """返回输出维度"""
        return self.config.llm_hidden_size