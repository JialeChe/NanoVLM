"""
NanoVLM 完整模型
组合 VisionEncoder + Connector(MLP) + LanguageModel
实现：图像 → 视觉特征 → 投影 → 拼接文本embedding → 语言模型 → 输出
"""

from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from PIL import Image

from configs.model_config import NanoVLMConfig, VisionConfig, LanguageConfig, ConnectorConfig, AnyResConfig
from .vision_encoder import VisionEncoder
from .language_model import LanguageModelWrapper
from .connector import Connector
from .anyres_processor import AnyResProcessor


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

        # 对齐 Connector dtype 与语言模型一致（FP16），避免 Half+Float 混算报错。
        # Vision Encoder 输出 FP16，Connector 默认 FP32，Linear 不兼容。
        self.connector = self.connector.to(dtype=self.language_model.dtype)

        # 设置 <image> token
        self._setup_image_token()

        # AnyRes 动态高分辨率处理器（LLaVA-NeXT 1.6）
        # enabled=False 时等价于原始单图模式，保留学习路线
        self.anyres_processor = AnyResProcessor(
            base_size=config.anyres.base_size,
            image_processor=self.vision_encoder.processor,
            grid_configs=list(config.anyres.grid_configs),
            max_tiles=config.anyres.max_tiles,
            enabled=config.anyres.enabled,
        )

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

        # 数值稳定性守卫：检测并钳制极端值（防止 FP16 attention 溢出）
        if self.training:
            vis_max = visual_embeddings.abs().max().item()
            if vis_max > 50:  # FP16 安全阈值，远低于 65504
                print(f"[WARNING] Connector output max={vis_max:.1f}, clamping to ±50")
                visual_embeddings = visual_embeddings.clamp(-50, 50)

        return visual_embeddings

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        visual_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建混合 embedding 序列

        将文本的 token embedding 与视觉 embedding 融合：
        - 找到 input_ids 中所有 <image> token 的位置
        - 用 visual_embeddings 替换这些位置

        visual_embeddings 支持两种格式（向后兼容）：
        - Tensor (B, N_v, hidden)：原始单图模式，每样本等长
        - List[Tensor]：AnyRes 模式，每样本可以不等长 [(N_v1, hidden), (N_v2, hidden), ...]

        Args:
            input_ids: (B, L) - 文本token序列（包含<image>占位符）
            visual_embeddings: (B, N_v, hidden) 或 List[(N_vi, hidden)]
            attention_mask: (B, L) - 注意力mask

        Returns:
            inputs_embeds: (B, L, llm_hidden_size) - 混合后的embedding
            attention_mask: (B, L) - attention_mask
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

            # visual_embeddings[b] 对 Tensor 和 List 都适用
            vis_emb = visual_embeddings[b]  # (N_v_i, hidden)
            num_vis_tokens = vis_emb.shape[0]

            if num_image_tokens != num_vis_tokens:
                # 如果<image> token数量不等于视觉token数量，做截断或填充
                if num_image_tokens < num_vis_tokens:
                    # 视觉token更多：截断多余的视觉token
                    vis_emb = vis_emb[:num_image_tokens]
                else:
                    # <image> token更多：多余位置填零
                    padding = torch.zeros(
                        num_image_tokens - num_vis_tokens, text_embeds.shape[-1],
                        device=text_embeds.device, dtype=text_embeds.dtype,
                    )
                    vis_emb = torch.cat([vis_emb, padding], dim=0)
            elif num_image_tokens == num_vis_tokens:
                pass  # 数量匹配，直接使用

            # 替换: 在 <image> token 的位置填入视觉 embedding
            text_embeds = text_embeds.clone()
            text_embeds[b, image_positions] = vis_emb.to(text_embeds.dtype)

        return text_embeds, attention_mask

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_counts: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播

        Args:
            pixel_values: 图像张量
                - 原始单图模式: (B, 3, H, W)
                - AnyRes 模式: (total_sub_images, 3, H, W) — 所有样本的全部 tile 展平
            input_ids: (B, L) - 文本token序列
            attention_mask: (B, L) - 注意力mask
            labels: (B, L) - 训练标签
            image_counts: List[int] — 每个样本包含几个 sub-image（缩略图+tiles）。
                为 None 时使用原始单图模式（向后兼容）。

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
            # 原始单图模式: (B, 729, 896)
            # AnyRes 展平后: (total_sub_images, 729, 896)

            # AnyRes 路径：按 image_counts 拆分并拼接每个样本的 visual tokens
            if image_counts is not None and self.config.anyres.enabled:
                vis_emb_list = []
                start = 0
                for count in image_counts:
                    # 取出该样本的 count 个 sub-image 的 visual tokens
                    sample_vis = visual_embeddings[start:start + count]
                    # 拼接为一个序列: (count, 729, 896) → (count * 729, 896)
                    sample_vis = sample_vis.reshape(-1, sample_vis.shape[-1])
                    vis_emb_list.append(sample_vis)
                    start += count
                visual_embeddings = vis_emb_list
                # visual_embeddings 现在是 List[(N_v1, 896), (N_v2, 896), ...]
            # 否则 visual_embeddings 保持为 (B, 729, 896) — 原始单图模式
        else:
            visual_embeddings = None

        # 2. 构建混合embedding
        if visual_embeddings is not None and input_ids is not None:
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(
                input_ids=input_ids,
                visual_embeddings=visual_embeddings,
                attention_mask=attention_mask,
            )

            # NaN 诊断 1：inputs_embeds 是否已含 NaN
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                print(f"[DIAG] NaN/Inf in inputs_embeds BEFORE LLM! "
                      f"shape={inputs_embeds.shape}, max={inputs_embeds.abs().max().item():.1f}", flush=True)

            # 3. 通过语言模型
            outputs = self.language_model(
                input_ids=None,  # 不使用input_ids，直接传embedding
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

            # NaN 诊断 2：logits 是否含 NaN
            if labels is not None:
                logits = outputs.logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_ratio = torch.isnan(logits).float().mean().item()
                    print(f"[DIAG] NaN/Inf in LOGITS! nan_ratio={nan_ratio:.4f}, "
                          f"logits_max={logits[~torch.isnan(logits)].abs().max().item():.1f}, "
                          f"inputs_embeds_max={inputs_embeds.abs().max().item():.1f}", flush=True)

            # 防御：labels 全为 -100 时（AnyRes 把文本挤出了 max_seq_length），返回 0 loss
            if labels is not None and (labels != -100).sum() == 0:
                outputs.loss = torch.tensor(0.0, device=outputs.logits.device,
                                            dtype=outputs.logits.dtype, requires_grad=True)
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

    def save_pretrained(self, save_dir: str, save_config: bool = True, save_tokenizer: bool = True):
        """保存模型

        Args:
            save_dir: 保存目录
            save_config: 是否保存 config.json（训练中不变的静态文件，只需保存一次）
            save_tokenizer: 是否保存 tokenizer 文件（训练中不变的静态文件，只需保存一次）
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 保存 connector (这是我们训练的主要部分)
        torch.save(self.connector.state_dict(), os.path.join(save_dir, "connector.bin"))

        # 保存 LoRA 权重（Stage 2）
        try:
            from peft import PeftModel
            if isinstance(self.language_model.model, PeftModel):
                lora_dir = os.path.join(save_dir, "lora")
                self.language_model.model.save_pretrained(lora_dir)
        except ImportError:
            pass

        # 保存配置（训练中不变，仅首次保存即可）
        if save_config:
            import json
            from dataclasses import asdict
            config_dict = {
                "vision": asdict(self.config.vision),
                "language": asdict(self.config.language),
                "connector": asdict(self.config.connector),
                "anyres": asdict(self.config.anyres),
                "image_token": self.config.image_token,
                "image_token_id": self.config.image_token_id,
            }
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)

        # 保存 tokenizer（训练中不变，仅首次保存即可）
        if save_tokenizer:
            self.language_model.tokenizer.save_pretrained(save_dir)

        print(f"[NanoVLM] Model saved to: {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, base_dir: str = None, **kwargs):
        """加载保存的模型

        Args:
            save_dir: 模型保存目录（包含 connector.bin，可能也包含 config.json 和 tokenizer）
            base_dir: 静态文件目录（config.json 和 tokenizer）。若 save_dir 中找不到则回退到此目录。
                     默认与 save_dir 相同，兼容完整保存的 checkpoint。
        """
        import os
        import json

        # 查找 config.json：先在 save_dir 找，再回退到 base_dir
        config_path = os.path.join(save_dir, "config.json")
        if not os.path.exists(config_path) and base_dir is not None:
            config_path = os.path.join(base_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.json not found in {save_dir}"
                + (f" or {base_dir}" if base_dir else "")
            )

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = NanoVLMConfig(
            vision=VisionConfig(**config_dict["vision"]),
            language=LanguageConfig(**config_dict["language"]),
            connector=ConnectorConfig(**config_dict["connector"]),
            # anyres 向后兼容：旧 checkpoint 没有此字段时使用默认值
            anyres=AnyResConfig(**config_dict["anyres"]) if "anyres" in config_dict else AnyResConfig(),
        )
        config.image_token = config_dict["image_token"]
        config.image_token_id = config_dict["image_token_id"]

        # 创建模型
        model = cls(config, **kwargs)

        # 加载 connector 权重
        connector_path = os.path.join(save_dir, "connector.bin")
        if os.path.exists(connector_path):
            state_dict = torch.load(connector_path, map_location="cpu")
            # strict=False: 兼容旧 checkpoint（没有 norm.weight 时使用新初始化的默认值）
            missing, unexpected = model.connector.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[NanoVLM] Connector missing keys (using defaults): {missing}")
            # connector 训练时保存为 float32，需要转换到与视觉/语言模型一致的 dtype
            model.connector.to(model.language_model.dtype)
            print(f"[NanoVLM] Loaded connector from: {connector_path}")

        # 加载 LoRA 权重（Stage 2）
        lora_dir = os.path.join(save_dir, "lora")
        if os.path.exists(lora_dir):
            try:
                from peft import PeftModel
                model.language_model.model = PeftModel.from_pretrained(
                    model.language_model.model, lora_dir
                )
                print(f"[NanoVLM] Loaded LoRA weights from: {lora_dir}")
            except ImportError:
                print("[NanoVLM] Warning: peft not installed, skipping LoRA loading")

        return model