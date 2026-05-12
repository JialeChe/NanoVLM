"""
推理
    # 交互式对话
    python scripts/inference.py --model_path ./checkpoints/stage2_epoch_1 --interactive
    # 单次推理
    python scripts/inference.py --model_path ./checkpoints/stage2_epoch_1 --image cat.jpg --question "What is in this image?"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from configs.model_config import NanoVLMConfig
from src.nanovlm.utils.utils import set_seed, get_device
from src.nanovlm.model.nanovlm import NanoVLM
from src.nanovlm.inference.generator import VLMGenerator


def main():
    parser = argparse.ArgumentParser(description="NanoVLM Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Please describe this image in detail.",
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 加载模型
    print("Loading NanoVLM model...")
    model_config = NanoVLMConfig()

    if args.model_path and os.path.exists(args.model_path):
        # 从检查点加载
        model = NanoVLM.from_pretrained(args.model_path)
        print(f"Model loaded from: {args.model_path}")
    else:
        # 从头创建（仅用于测试，connector未训练）
        model = NanoVLM(model_config)
        print("Model created from scratch (untrained)")

    model.to(device)
    model.eval()

    # 创建生成器
    generator = VLMGenerator(
        model=model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # 推理
    if args.interactive:
        generator.chat()
    else:
        if args.image:
            image = Image.open(args.image).convert("RGB")
        else:
            image = None

        print(f"\nQuestion: {args.question}")
        print("Generating response...")

        response = generator.generate(
            image=image,
            question=args.question,
        )

        print(f"\nResponse:\n{response}")

        # 如果有图片，也进行描述
        if image is None:
            print("(No image provided — this was a text-only query)")


if __name__ == "__main__":
    main()