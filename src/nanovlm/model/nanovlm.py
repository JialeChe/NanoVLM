"""
NanoVLM 完整模型
组合 VisionEncoder + Connector(MLP) + LanguageModel
实现：图像 → 视觉特征 → 投影 → 拼接文本embedding → 语言模型 → 输出
"""

from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from PIL import Image

from configs.model_config import NanoVLMConfig, VisionConfig, LanguageConfig, ConnectorConfig
from .vision_encoder import VisionEncoder
from .language_model import LanguageModelWrapper
from .connector import Connector


class NanoVLM(nn.Module):
    """
    NanoVLM: 轻量级视觉-语言模型

    前向传播流程:
    1. 图像 → VisionEncoder → patch features (B, N_v, 1024)
    2. patch features → Connector(MLP) → visual embeddings (B, N_v, 1536)
    3. 文本 → Tokenizer → input_ids (B, L)
    4. 替换 <image> token 位置为 visual embeddings
    5. 拼接后的 embeddings → LanguageModel → logits / loss
    """

    def __init__(self, config: NanoVLMConfig = None):
        super().__init__()

        if config is None:
            config = NanoVLMConfig()

        self.config = config

        # 三大模块
        self.vision_encoder = VisionEncoder(config.vision)
        self.language_model = LanguageModelWrapper(config.language)

        # 用真实加载到的模型维度覆盖手工配置，避免本地权重与配置不一致。
        self.config.connector.vision_hidden_size = self.vision_encoder.hidden_size
        self.config.connector.llm_hidden_size = self.language_model.hidden_size
        self.connector = Connector(config.connector)

        # 设置 <image> token
        self._setup_image_token()

        # 当前训练阶段
        self._stage = "stage1"

    def _setup_image_token(self):
        """
        设置图像占位 token

        我们使用 tokenizer 中已有的特殊 token 作为 <image> 标记，
        或者添加一个新的特殊 token
        """
        tokenizer = self.language_model.tokenizer
        image_token = self.config.image_token  # "<image>"

        # 尝试添加特殊 token，如果已存在则获取其 id
        special_tokens = tokenizer.additional_special_tokens

        if image_token not in special_tokens and image_token not in tokenizer.get_vocab():
            # 添加特殊token
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            # 调整语言模型的embedding层大小
            self.language_model.model.resize_token_embeddings(len(tokenizer))
            print(f"[NanoVLM] Added special token: {image_token}")
        else:
            print(f"[NanoVLM] Token '{image_token}' already exists in tokenizer")

        # 获取 image_token id
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.config.image_token_id = self.image_token_id
        print(f"[NanoVLM] Image token id: {self.image_token_id}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码为视觉特征，并通过投影层

        Args:
            images: (B, 3, H, W) - 预处理后的图像

        Returns:
            visual_embeddings: (B, num_patches, llm_hidden_size)
        """
        # 1. 通过视觉编码器: (B, 3, H, W) → (B, num_patches+1, 1024)
        vision_features = self.vision_encoder(images)

        # CLIP 系列包含 CLS token，SigLIP 视觉塔通常直接返回 patch tokens。
        if self.vision_encoder.has_cls_token:
            patch_features = vision_features[:, 1:, :]
        else:
            patch_features = vision_features

        # 3. 通过MLP投影到LLM空间
        visual_embeddings = self.connector(patch_features)

        return visual_embeddings

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        visual_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建混合 embedding 序列

        将文本的 token embedding 与视觉 embedding 融合：
        - 找到 input_ids 中所有 <image> token 的位置
        - 用 visual_embeddings 替换这些位置

        Args:
            input_ids: (B, L) - 文本token序列（包含<image>占位符）
            visual_embeddings: (B, N_v, llm_hidden_size) - 视觉embedding
            attention_mask: (B, L) - 注意力mask

        Returns:
            inputs_embeds: (B, L', llm_hidden_size) - 混合后的embedding
            attention_mask: (B, L') - 调整后的attention_mask
        """
        # 获取文本embedding层
        embed_tokens = self.language_model.get_input_embeddings()

        # 将 input_ids 转换为文本 embedding
        text_embeds = embed_tokens(input_ids)  # (B, L, llm_hidden_size)

        # 找到所有 <image> token 的位置
        batch_size, seq_len = input_ids.shape
        image_token_mask = (input_ids == self.image_token_id)  # (B, L)

        # 对于每个样本，在 <image> 位置填入视觉 embedding
        for b in range(batch_size):
            image_positions = image_token_mask[b].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_positions)

            if num_image_tokens == 0:
                continue

            # 视觉embedding的数量
            num_vis_tokens = visual_embeddings.shape[1]  # 例如 576

            if num_image_tokens != num_vis_tokens:
                # 如果<image> token数量不等于视觉token数量，需要处理
                # 最简单的方式：将多个<image>展平为多个视觉token
                # 这里假设 input_ids 中 <image> 的数量和视觉patch数一致
                # 如果不一致，做截断或填充
                if num_image_tokens < num_vis_tokens:
                    # 视觉token更多：截断多余的视觉token
                    visual_embeds_used = visual_embeddings[b, :num_image_tokens, :]
                else:
                    # <image> token更多：复制最后的视觉token填充
                    visual_embeds_used = visual_embeddings[b]
                    # 前面num_vis_tokens个位置用视觉embedding
                    # 多余的<image>位置用零填充（或padding embedding）
                    pass
            else:
                visual_embeds_used = visual_embeddings[b]

            # 替换: 在 <image> token 的位置填入视觉 embedding
            text_embeds[b, image_positions] = visual_embeds_used.to(text_embeds.dtype)

        return text_embeds, attention_mask

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播

        Args:
            pixel_values: (B, 3, H, W) - 预处理后的图像（别名: images）
            input_ids: (B, L) - 文本token序列
            attention_mask: (B, L) - 注意力mask
            labels: (B, L) - 训练标签

        Returns:
            dict with keys:
                - loss: 交叉熵损失（训练时）
                - logits: (B, L, vocab_size)
        """
        # 统一 images 和 pixel_values
        if images is not None:
            pixel_values = images

        # 1. 编码图像
        if pixel_values is not None:
            visual_embeddings = self.encode_images(pixel_values)
        else:
            visual_embeddings = None

        # 2. 构建混合embedding
        if visual_embeddings is not None and input_ids is not None:
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(
                input_ids=input_ids,
                visual_embeddings=visual_embeddings,
                attention_mask=attention_mask,
            )

            # 3. 通过语言模型
            outputs = self.language_model(
                input_ids=None,  # 不使用input_ids，直接传embedding
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        else:
            # 纯文本模式（无图像）
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    def set_stage(self, stage: str):
        """设置训练阶段"""
        self._stage = stage
        self.language_model.set_stage(stage)
        print(f"[NanoVLM] Set stage to: {stage}")

    def get_trainable_parameters(self):
        """获取可训练参数数量"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[NanoVLM] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        return trainable

    def save_pretrained(self, save_dir: str):
        """保存模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 保存 connector (这是我们训练的主要部分)
        torch.save(self.connector.state_dict(), os.path.join(save_dir, "connector.bin"))

        # 保存配置
        import json
        from dataclasses import asdict
        config_dict = {
            "vision": asdict(self.config.vision),
            "language": asdict(self.config.language),
            "connector": asdict(self.config.connector),
            "image_token": self.config.image_token,
            "image_token_id": self.config.image_token_id,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # 保存 tokenizer
        self.language_model.tokenizer.save_pretrained(save_dir)

        print(f"[NanoVLM] Model saved to: {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        """加载保存的模型"""
        import os
        import json

        # 加载配置
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config_dict = json.load(f)

        config = NanoVLMConfig(
            vision=VisionConfig(**config_dict["vision"]),
            language=LanguageConfig(**config_dict["language"]),
            connector=ConnectorConfig(**config_dict["connector"]),
        )
        config.image_token = config_dict["image_token"]
        config.image_token_id = config_dict["image_token_id"]

        # 创建模型
        model = cls(config, **kwargs)

        # 加载 connector 权重
        connector_path = os.path.join(save_dir, "connector.bin")
        if os.path.exists(connector_path):
            model.connector.load_state_dict(torch.load(connector_path, map_location="cpu"))
            print(f"[NanoVLM] Loaded connector from: {connector_path}")

        return model