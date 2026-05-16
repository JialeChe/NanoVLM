#!/usr/bin/env python3
"""
NanoVLM 训练入口脚本
用法:
    python scripts/train.py --config configs/training_config.py --data_path data/llava_instruct.json
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.model_config import NanoVLMConfig
from configs.training_config import TrainingConfig
from src.nanovlm.utils.utils import set_seed, get_device
from src.nanovlm.model.nanovlm import NanoVLM
from src.nanovlm.data.dataset import LLaVADataset, create_dummy_data
from src.nanovlm.training.trainer import NanoVLMTrainer


def main():
    parser = argparse.ArgumentParser(description="NanoVLM Training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/llava_pretrain/blip_laion_cc_sbu_558k.json",
        help="Path to training data (LLaVA format JSON)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Base directory for images (if relative paths in JSON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage1,stage2",
        help="Training stages: stage1,stage2 or stage1+stage2",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
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
    parser.add_argument(
        "--create_dummy",
        action="store_true",
        help="Create dummy training data for testing",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory (e.g., checkpoints/stage1_step_55000)",
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 创建哑数据（如果需要）
    if args.create_dummy or not os.path.exists(args.data_path):
        args.data_path = create_dummy_data(os.path.dirname(args.data_path) or "./data")
        print("Created dummy data for testing")

    # 加载配置
    model_config = NanoVLMConfig()
    training_config = TrainingConfig()
    training_config.output_dir = args.output_dir

    print("\n" + "=" * 60)
    print("NanoVLM Model Configuration")
    print("=" * 60)
    print(f"  Vision: {model_config.vision.model_name_or_path}")
    print(f"  Language: {model_config.language.model_name_or_path}")
    print(f"  Image size: {model_config.vision.image_size}")
    print(f"  MLP hidden: {model_config.connector.mlp_hidden_size}")
    print("=" * 60)

    # 创建模型
    print("\nLoading model...")
    model = NanoVLM(model_config)
    model.to(device)

    # 获取必要的参数
    image_token_id = model.image_token_id
    num_image_tokens = model.vision_encoder.get_num_patches()
    print(f"Image token ID: {image_token_id}")
    print(f"Num visual tokens: {num_image_tokens}")

    # 加载数据集
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = LLaVADataset(
        data_path=args.data_path,
        tokenizer=model.language_model.tokenizer,
        image_processor=model.vision_encoder.processor,
        image_token_id=image_token_id,
        num_image_tokens=num_image_tokens,
        max_seq_length=args.max_seq_length,
        image_base_dir=args.image_dir,
    )
    print(f"Dataset size: {len(dataset)}")

    # 创建训练器
    trainer = NanoVLMTrainer(
        model=model,
        config=training_config,
        train_dataset=dataset,
    )

    # 开始训练
    stages = [s.strip() for s in args.stage.split(",")]
    trainer.train(stages=stages, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()