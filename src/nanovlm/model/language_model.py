"""
语言模型封装：基于 Qwen2.5-1.5B-Instruct
- 负责接收视觉token + 文本token的拼接序列，进行自回归生成
- 支持 Stage1 全量冻结 + Stage2 LoRA微调
"""

from typing import Optional, Dict, List
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from configs.model_config import LanguageConfig


class LanguageModelWrapper(nn.Module):
    """
    Qwen2.5 语言模型封装

    输入: input_ids (B, seq_len) + 可选 attention_mask
    输出: logits (B, seq_len, vocab_size)
    """

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config

        print(f"[LanguageModel] Loading LM: {config.model_name_or_path}")
        print(f"[LanguageModel] Local cache dir: {config.local_cache_dir}")

        # 加载预训练语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # 加载 tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.local_cache_dir,
            trust_remote_code=True,
            use_fast=False,
        )

        # 确保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = config.hidden_size  # 1536
        self._stage = "stage1"  # 当前训练阶段

    def set_stage(self, stage: str):
        """
        设置训练阶段，控制参数冻结/解冻

        Stage 1: 语言模型完全冻结
        Stage 2: 语言模型可训练（通常配合LoRA使用）
        """
        self._stage = stage

        if stage == "stage1":
            # Stage 1: 冻结语言模型
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        elif stage == "stage2":
            # Stage 2: 解冻语言模型
            # 注意：如果使用LoRA，会在Trainer中通过peft包装
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: (B, seq_len) - 包含视觉投影后的embedding位置用占位符token填充
            attention_mask: (B, seq_len) - 注意力mask
            labels: (B, seq_len) - 用于计算loss的标签

        Returns:
            CausalLMOutput with logits and optional loss
        """
        if self._stage == "stage1":
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                    **kwargs,
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                **kwargs,
            )

        return outputs

    def get_input_embeddings(self) -> nn.Embedding:
        """获取模型的input embedding层，用于将视觉特征注入"""
        return self.model.get_input_embeddings()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype