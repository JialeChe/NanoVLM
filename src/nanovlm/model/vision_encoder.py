"""
视觉编码器：支持 CLIP ViT / SigLIP 等多种视觉骨干
- 负责将图像像素转换为视觉特征向量序列
- 输入: (B, 3, H, W) - 统一使用 config.image_size
- 输出: (B, num_patches, vision_hidden_size)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from configs.model_config import VisionConfig


class VisionEncoder(nn.Module):
    """
    通用视觉编码器封装

    输入图像 → [Vision Backbone] → patch特征序列
        (B,3,H,W) → (B, num_patches, hidden_size)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        print(f"[VisionEncoder] Loading vision model: {config.model_name_or_path}")
        print(f"[VisionEncoder] Local cache dir: {config.local_cache_dir}")

        # 加载预训练视觉模型 (自动检测 CLIP / SigLIP 等)
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
            dtype=torch.float16,
            local_files_only=True
        )

        self.vision_model = getattr(self.model, "vision_model", self.model)
        self.vision_config = getattr(self.model.config, "vision_config", self.vision_model.config)
        model_type = getattr(self.vision_config, "model_type", "")
        self.has_cls_token = model_type.startswith("clip")

        # 冻结所有参数
        if config.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # 图像预处理器
        self.processor = AutoImageProcessor.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
            local_files_only=True
        )

        self.hidden_size = self.vision_config.hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            pixel_values: (B, 3, H, W) - 预处理后的图像张量
            output_hidden_states: 是否返回中间层特征

        Returns:
            last_hidden_state: (B, num_patches + 1, hidden_size)
                - [:, 0, :] 是 CLS token
                - [:, 1:, :] 是 patch tokens
        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
        else:
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        return outputs.last_hidden_state

    def get_num_patches(self) -> int:
        """估算输出patch数"""
        image_size = self.config.image_size  # e.g. 384
        patch_size = getattr(self.vision_config, "patch_size", None)
        if patch_size is None:
            raise AttributeError(
                f"Cannot find patch_size in {type(self.vision_config).__name__}."
            )
        num_patches = (image_size // patch_size) ** 2  # e.g. (384/14)^2 = 729
        return num_patches

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def get_device(self):
        return next(self.model.parameters()).device