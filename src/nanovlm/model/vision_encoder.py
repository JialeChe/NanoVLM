"""
视觉编码器：基于 CLIP ViT-L/14@336px
- 负责将图像像素转换为视觉特征向量序列
- 输入: (B, 3, 336, 336) 或支持动态尺寸
- 输出: (B, num_patches, vision_hidden_size)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from configs.model_config import VisionConfig


class VisionEncoder(nn.Module):
    """
    CLIP ViT 视觉编码器封装

    输入图像 → [CLIP ViT] → patch特征序列
        (B,3,H,W) → (B, num_patches, 1024)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        print(f"[VisionEncoder] Loading CLIP vision model: {config.model_name_or_path}")
        print(f"[VisionEncoder] Local cache dir: {config.local_cache_dir}")

        # 加载预训练 CLIP 视觉模型
        self.model = CLIPVisionModel.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
            torch_dtype=torch.float16,
        )

        # 冻结所有参数
        if config.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # 图像预处理器
        self.processor = CLIPImageProcessor.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
        )

        self.hidden_size = self.model.config.hidden_size  # 1024

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
                num_patches = (image_size / patch_size)^2 = (336/14)^2 = 576
        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.model(
                    pixel_values=pixel_values,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
        else:
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        return outputs.last_hidden_state

    def get_num_patches(self) -> int:
        """估算输出patch数 — 仅供外部参考"""
        image_size = self.config.image_size  # 336
        patch_size = self.model.config.patch_size  # 14
        num_patches = (image_size // patch_size) ** 2  # 576
        return num_patches

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def get_device(self):
        return next(self.model.parameters()).device