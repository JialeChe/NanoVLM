"""
LLaVA 风格的对话格式处理

标准格式:
{
    "id": "sample_001",
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\n请描述这张图片。"},
        {"from": "gpt", "value": "这是一张...的图片。"}
    ]
}

输入序列构建:
    <image> token × N_visual + <|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>
"""

from typing import List, Dict, Optional, Tuple
import torch


class Conversation:
    """对话管理器"""

    def __init__(
        self,
        system_message: str = "You are a helpful vision-language assistant.",
        roles: Tuple[str, str] = ("human", "gpt"),
        sep_style: str = "qwen",  # 分隔符风格
    ):
        """
        Args:
            system_message: 系统提示
            roles: (提问者, 回答者) 角色名
            sep_style: 分隔符风格 ("qwen" | "llava")
        """
        self.system_message = system_message
        self.roles = roles
        self.sep_style = sep_style

        # Qwen2.5 chat template tokens
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"

    def apply_chat_template(
        self,
        conversations: List[Dict[str, str]],
        tokenizer,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        将对话列表转为模型输入文本

        Args:
            conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            tokenizer: 对应的tokenizer
            add_generation_prompt: 是否添加生成提示（推理时使用）

        Returns:
            格式化后的文本字符串
        """
        if self.sep_style == "qwen":
            return self._qwen_style(conversations, add_generation_prompt)
        else:
            # 通用风格
            return self._generic_style(conversations, add_generation_prompt)

    def _qwen_style(
        self,
        conversations: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Qwen2.5 ChatML 格式"""
        system = f"{self.im_start}system\n{self.system_message}{self.im_end}\n"
        text = system

        for turn in conversations:
            role = turn["from"]
            content = turn["value"]

            if role == self.roles[0]:  # human
                text += f"{self.im_start}user\n{content}{self.im_end}\n"
            elif role == self.roles[1]:  # gpt
                text += f"{self.im_start}assistant\n{content}{self.im_end}\n"

        if add_generation_prompt:
            text += f"{self.im_start}assistant\n"

        return text

    def _generic_style(
        self,
        conversations: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """通用格式"""
        text = f"### System:\n{self.system_message}\n\n"

        for turn in conversations:
            role = turn["from"]
            content = turn["value"]

            if role == self.roles[0]:
                text += f"### Human:\n{content}\n\n"
            elif role == self.roles[1]:
                text += f"### Assistant:\n{content}\n\n"

        if add_generation_prompt:
            text += "### Assistant:\n"

        return text

    def tokenize_conversation(
        self,
        conversations: List[Dict[str, str]],
        tokenizer,
        max_length: int,
        image_token_id: int,
        num_image_tokens: int,
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        将对话转为训练用的 input_ids 和 labels

        训练时:
            - labels 中 user 部分设为 -100（不计算loss）
            - labels 中 assistant 部分为真实token

        Args:
            conversations: 对话列表
            tokenizer: tokenizer 对象
            max_length: 最大序列长度
            image_token_id: <image> token 的 id
            num_image_tokens: 图像占位 token 数量（=视觉patch数）
            is_training: 是否训练模式

        Returns:
            dict: {input_ids, labels, attention_mask}
        """
        # 构建对话文本
        text = self.apply_chat_template(conversations, tokenizer)

        # 将 <image> 替换为多个 image token
        # 原始文本中有一个 <image>，我们需要替换为 num_image_tokens 个 image_token_id
        image_placeholder = "<image>"
        image_tokens_str = "".join([tokenizer.decode([image_token_id])] * num_image_tokens)

        # 如果文本中有 <image>，替换为多个 image token
        if image_placeholder in text:
            text = text.replace(image_placeholder, image_tokens_str, 1)

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = [1] * len(input_ids)

        # 创建 labels
        labels = input_ids.copy()

        if is_training:
            # 找到 assistant 回复的起始位置
            # 使用 im_start + assistant 来定位
            assistant_start_marker = f"{self.im_start}assistant\n"
            assistant_start_tokens = tokenizer.encode(assistant_start_marker, add_special_tokens=False)

            # 在 input_ids 中搜索 assistant 开始位置
            ass_start_pos = None
            for i in range(len(input_ids) - len(assistant_start_tokens) + 1):
                if input_ids[i:i + len(assistant_start_tokens)] == assistant_start_tokens:
                    ass_start_pos = i + len(assistant_start_tokens)
                    break

            if ass_start_pos is not None:
                # assistant 之前的部分（system + user）设置为 -100
                for i in range(ass_start_pos):
                    labels[i] = -100
            else:
                # 如果没找到 assistant 标记，全部设为 -100（不太优雅但安全）
                labels = [-100] * len(input_ids)

        # Padding 到 max_length
        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }