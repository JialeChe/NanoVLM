"""
多模态对话数据集加载
支持 LLaVA 格式的 JSON 数据文件

数据格式:
[
  {
    "id": "...",
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\n问题"},
      {"from": "gpt", "value": "回答"}
    ]
  },
  ...
]
"""

import os
import json
from typing import List, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from .conversation import Conversation


class LLaVADataset(Dataset):
    """LLaVA格式多模态对话数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        image_token_id: int,
        num_image_tokens: int,
        max_seq_length: int = 2048,
        image_base_dir: Optional[str] = None,
    ):
        """
        Args:
            data_path: JSON 数据文件路径
            tokenizer: 语言模型 tokenizer
            image_processor: CLIP 图像预处理器
            image_token_id: <image> token 的 id
            num_image_tokens: 图像占位 token 数量
            max_seq_length: 最大序列长度
            image_base_dir: 图像文件根目录（如果JSON中用的是相对路径）
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.max_seq_length = max_seq_length

        # 加载 JSON 数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"[Dataset] Loaded {len(self.data)} samples from {data_path}")

        # 确定图像目录
        if image_base_dir:
            self.image_base_dir = image_base_dir
        else:
            # 默认用 data 文件所在目录的 images 子目录
            self.image_base_dir = os.path.join(os.path.dirname(data_path), "images")

        # 创建对话管理器
        self.conversation = Conversation(sep_style="qwen")

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path: str) -> Image.Image:
        """加载并返回 PIL Image"""
        # 如果是相对路径，拼接到 image_base_dir
        if not os.path.isabs(image_path):
            full_path = os.path.join(self.image_base_dir, image_path)
        else:
            full_path = image_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert("RGB")
        return image

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """使用 CLIP processor 预处理图像"""
        # CLIPImageProcessor 返回 {pixel_values: (1, 3, H, W)}
        processed = self.image_processor(
            images=image,
            return_tensors="pt",
        )
        return processed["pixel_values"].squeeze(0)  # (3, H, W)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # 1. 加载并处理图像
        image_path = sample.get("image", None)
        if image_path:
            image = self.load_image(image_path)
            pixel_values = self.process_image(image)
        else:
            # 无图像样本（纯文本对话）
            pixel_values = None

        # 2. 处理对话文本
        conversations = sample["conversations"]
        tokenized = self.conversation.tokenize_conversation(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
            image_token_id=self.image_token_id,
            num_image_tokens=self.num_image_tokens,
            is_training=True,
        )

        result = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result


def create_dummy_data(data_dir: str = "./data"):
    """创建示例数据（用于测试）"""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)

    # 创建一个简单的示例 JSON
    sample_data = [
        {
            "id": "sample_001",
            "image": "sample.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nPlease describe this image in detail."},
                {"from": "gpt", "value": "This image shows a beautiful landscape with mountains in the background and a lake in the foreground. The sky is blue with scattered clouds."},
            ],
        },
        {
            "id": "sample_002",
            "image": "sample.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nHow many people are in this image?"},
                {"from": "gpt", "value": "There are three people visible in this image."},
            ],
        },
    ]

    data_path = os.path.join(data_dir, "llava_instruct_sample.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"[Dataset] Created sample data at: {data_path}")
    return data_path