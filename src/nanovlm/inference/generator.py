"""
NanoVLM 推理生成器
支持单图问答、批量推理、流式生成
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from PIL import Image

from ..model.nanovlm import NanoVLM
from ..data.conversation import Conversation


class VLMGenerator:
    """NanoVLM 推理生成器"""

    def __init__(
        self,
        model: NanoVLM,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ):
        """
        Args:
            model: NanoVLM 模型实例
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: nucleus sampling 阈值
            top_k: top-k sampling 阈值
            do_sample: 是否采样（False=贪心解码）
            repetition_penalty: 重复惩罚系数
        """
        self.model = model
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        self.device = next(model.parameters()).device
        self.tokenizer = model.language_model.tokenizer
        self.image_processor = model.vision_encoder.processor
        self.image_token_id = model.image_token_id

        self.conversation = Conversation(sep_style="qwen")

    def generate(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        question: Optional[str] = None,
        conversations: Optional[list] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        根据图像和问题生成回答

        支持两种图像处理模式：
        - 原始单图：resize 到 384×384（向后兼容）
        - AnyRes：动态高分辨率切分（需 anyres.enabled=True）

        Args:
            image: PIL Image 或图片路径
            question: 用户问题
            conversations: 预构建的对话列表（如果提供则忽略 question）
            max_new_tokens: 覆盖默认最大生成长度

        Returns:
            模型生成的文本回答
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # 检查是否启用 AnyRes
        use_anyres = (
            hasattr(self.model, 'anyres_processor') and
            self.model.anyres_processor.enabled
        )

        # 0. 预先确定 visual token 数量（需要先处理好图像才能知道）
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            if use_anyres:
                # AnyRes 路径：动态切分图像
                pixel_values, num_sub_images = self.model.anyres_processor.process(image)
                # pixel_values: (N_sub, 3, 384, 384)
                # num_vis_tokens = num_sub_images × 729
                num_vis_tokens_per_patch = self.model.vision_encoder.get_num_patches()
                num_vis_tokens = num_sub_images * num_vis_tokens_per_patch
                pixel_values = pixel_values.to(self.device)
                image_counts = [num_sub_images]
            else:
                # 原始单图路径
                processed = self.image_processor(
                    images=image,
                    return_tensors="pt",
                )
                pixel_values = processed["pixel_values"].to(self.device)
                # pixel_values: (1, 3, 384, 384)
                num_vis_tokens = self.model.vision_encoder.get_num_patches()
                image_counts = None
        else:
            pixel_values = None
            num_vis_tokens = self.model.vision_encoder.get_num_patches()
            image_counts = None

        # 1. 构建对话
        if conversations is None:
            if question is None:
                raise ValueError("Either 'question' or 'conversations' must be provided")

            if "<image>" not in question:
                question = f"<image>\n{question}"
            conversations = [
                {"from": "human", "value": question},
            ]

        # 构建输入文本（包含 system + user，添加 generation prompt）
        prompt = self.conversation.apply_chat_template(
            conversations=conversations,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )

        # 替换 <image> 为 num_vis_tokens 个 image_token
        image_token_str = self.tokenizer.decode([self.image_token_id])
        prompt = prompt.replace("<image>", image_token_str * num_vis_tokens)

        # 2. Tokenize 输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 3. 编码图像获取视觉 embedding
        if pixel_values is not None:
            with torch.no_grad():
                visual_embeddings = self.model.encode_images(pixel_values)
            # visual_embeddings:
            #   原始单图: (1, 729, 896)
            #   AnyRes:   (N_sub, 729, 896)

            # AnyRes 路径：将 sub-image embeddings 拼接为单样本序列
            if use_anyres and image_counts is not None:
                vis_emb_list = []
                start = 0
                for count in image_counts:
                    sample_vis = visual_embeddings[start:start + count]
                    sample_vis = sample_vis.reshape(-1, sample_vis.shape[-1])
                    vis_emb_list.append(sample_vis)
                    start += count
                visual_embeddings = vis_emb_list
                # visual_embeddings = [(count*729, 896)]
        else:
            visual_embeddings = None

        # 4. 构建混合 embedding
        if visual_embeddings is not None:
            inputs_embeds, attention_mask = self.model.prepare_inputs_embeds(
                input_ids=input_ids,
                visual_embeddings=visual_embeddings,
                attention_mask=attention_mask,
            )
        else:
            embed_tokens = self.model.language_model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)

        # 5. 自回归生成
        generated_ids = self._generate_from_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            input_ids=input_ids,
        )

        # 6. 解码
        output_ids = generated_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 清理：去除多余的结束标记
        response = response.replace("<|im_end|>", "").strip()

        return response

    def _generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """从 embedding 开始自回归生成"""
        lm_model = self.model.language_model.model
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    outputs = lm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                else:
                    # 只用最后一个 token 的 embedding
                    current_embed = lm_model.get_input_embeddings()(
                        generated[:, -1:]
                    )
                    outputs = lm_model(
                        inputs_embeds=current_embed,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            logits = outputs.logits[:, -1, :]

            # 温度调节
            if self.temperature > 0 and self.do_sample:
                logits = logits / self.temperature

            # Top-K 过滤
            if self.top_k > 0 and self.do_sample:
                top_k_values, _ = torch.topk(logits, self.top_k, dim=-1)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_top_k] = -float("inf")

            # Top-P 过滤
            if self.top_p < 1.0 and self.do_sample:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                for b in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float("inf")

            # 采样或贪婪解码
            if self.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 更新
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.shape[0], 1, device=attention_mask.device)],
                dim=-1,
            )
            past_key_values = outputs.past_key_values

            # 遇到结束 token 则停止
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated

    def chat(self):
        """交互式对话（命令行）"""
        print("\n" + "=" * 60)
        print("NanoVLM Interactive Chat")
        print("Type 'exit' or 'quit' to stop")
        print("Type 'image: <path>' to load an image")
        print("=" * 60 + "\n")

        current_image = None

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if user_input.lower().startswith("image:"):
                    image_path = user_input[len("image:"):].strip()
                    try:
                        current_image = Image.open(image_path).convert("RGB")
                        print(f"[Image loaded: {image_path}]")
                    except Exception as e:
                        print(f"[Error loading image: {e}]")
                    continue

                if user_input:
                    response = self.generate(
                        image=current_image,
                        question=user_input,
                    )
                    print(f"Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"[Error: {e}]")