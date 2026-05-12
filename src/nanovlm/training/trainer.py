"""
NanoVLM 训练器
支持两阶段训练：Stage1（仅训练MLP）+ Stage2（训练MLP + LLM LoRA）
集成 TensorBoard 日志记录
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict
from tqdm import tqdm

from configs.training_config import TrainingConfig, Stage1Config, Stage2Config
from ..utils.utils import set_seed, get_dtype, ensure_dir


class NanoVLMTrainer:
    """NanoVLM 训练器"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset,
        val_dataset=None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.current_epoch = 0

        # 输出目录
        self.output_dir = config.output_dir
        ensure_dir(self.output_dir)

        # TensorBoard
        if config.use_tensorboard:
            log_dir = os.path.join(self.output_dir, "tb_logs")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[Trainer] TensorBoard logs will be saved to: {log_dir}")
        else:
            self.writer = None

    def _create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """创建 DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """自定义 batch 合并函数"""
        # 收集所有键
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            values = [item[key] for item in batch if key in item]

            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                # Padding tensor
                max_len = max(v.shape[0] for v in values if v.dim() > 0)
                padded = []
                for v in values:
                    if v.dim() == 0:
                        padded.append(v.unsqueeze(0))
                        continue
                    if v.shape[0] < max_len:
                        pad = torch.zeros(max_len - v.shape[0], dtype=v.dtype)
                        padded.append(torch.cat([v, pad]))
                    else:
                        padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                collated[key] = values

        return collated

    def _get_optimizer_and_scheduler(
        self,
        stage_config,
        total_steps: int,
    ):
        """创建优化器和学习率调度器"""
        # 只优化 requires_grad=True 的参数
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        optimizer = AdamW(
            trainable_params,
            lr=stage_config.learning_rate,
            weight_decay=stage_config.weight_decay,
        )

        # 预热 + 余弦衰减
        warmup_steps = int(total_steps * stage_config.warmup_ratio)
        main_steps = total_steps - warmup_steps

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        return optimizer, scheduler

    def _log_tb(self, tag: str, value: float, step: int):
        """记录标量到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def train_stage1(self, stage1_config: Stage1Config = None):
        """
        Stage 1 训练：仅训练MLP投影层

        冻结：Vision Encoder + Language Model
        训练：Connector (MLP Projector)
        """
        if stage1_config is None:
            stage1_config = self.config.stage1

        print("\n" + "=" * 60)
        print("Stage 1: Training Connector (MLP Projector)")
        print("=" * 60)
        print(f"  Epochs: {stage1_config.num_epochs}")
        print(f"  Batch size: {stage1_config.per_device_batch_size}")
        print(f"  Learning rate: {stage1_config.learning_rate}")
        print(f"  Gradient accumulation: {stage1_config.gradient_accumulation_steps}")
        print(f"  TensorBoard: {'enabled' if self.writer else 'disabled'}")
        print("=" * 60)

        # 设置 Stage 1
        self.model.set_stage("stage1")
        self.model.train()
        self.model.get_trainable_parameters()

        # 创建 DataLoader
        train_loader = self._create_dataloader(
            self.train_dataset,
            batch_size=stage1_config.per_device_batch_size,
            shuffle=True,
        )

        # 计算总步数
        total_steps = (
            len(train_loader)
            * stage1_config.num_epochs
            // stage1_config.gradient_accumulation_steps
        )

        # 创建优化器和调度器
        optimizer, scheduler = self._get_optimizer_and_scheduler(
            stage1_config, total_steps
        )

        # 混合精度
        scaler = torch.amp.GradScaler('cuda', enabled=stage1_config.use_fp16)

        # 保存一次静态文件（config + tokenizer），后续 checkpoint 不再重复保存
        self.model.save_pretrained(self.output_dir)

        # 训练循环
        stage_prefix = "stage1"
        for epoch in range(stage1_config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{stage1_config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                # 将数据移到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch.get("pixel_values")

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)

                # 混合精度前向传播
                with torch.amp.autocast('cuda', enabled=stage1_config.use_fp16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]

                # 梯度累积
                loss = loss / stage1_config.gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                if (step + 1) % stage1_config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # 优化器步
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    self.global_step += 1

                    # TensorBoard 日志（实际优化步级）
                    current_lr = scheduler.get_last_lr()[0]
                    self._log_tb(f"{stage_prefix}/loss", loss.item() * stage1_config.gradient_accumulation_steps, self.global_step)
                    self._log_tb(f"{stage_prefix}/lr", current_lr, self.global_step)
                    self._log_tb(f"{stage_prefix}/grad_norm", grad_norm.item(), self.global_step)

                # 日志
                epoch_loss += loss.item() * stage1_config.gradient_accumulation_steps
                avg_loss = epoch_loss / (step + 1)

                progress_bar.set_postfix({
                    "loss": f"{loss.item() * stage1_config.gradient_accumulation_steps:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # 每 logging_steps 记录 (数据步级)
                if (step + 1) % stage1_config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    self._log_tb(f"{stage_prefix}/loss_per_step", loss.item() * stage1_config.gradient_accumulation_steps, epoch * len(train_loader) + step)

                # 定期保存（仅保存可训练权重，config/tokenizer 已在训练开始时保存）
                if self.global_step > 0 and self.global_step % stage1_config.save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"stage1_step_{self.global_step}")
                    self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
                    print(f"\n[Stage1] Checkpoint saved to: {save_path}")

            # Epoch 结束保存
            save_path = os.path.join(self.output_dir, f"stage1_epoch_{epoch+1}")
            self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
            self._log_tb(f"{stage_prefix}/epoch_avg_loss", avg_loss, epoch + 1)
            print(f"\nStage 1 Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

        print("\nStage 1 training completed!")
        return self.model

    def train_stage2(self, stage2_config: Stage2Config = None):
        """
        Stage 2 训练：训练MLP + 语言模型(LoRA)

        冻结：Vision Encoder
        训练：Connector + Language Model (LoRA)
        """
        if stage2_config is None:
            stage2_config = self.config.stage2

        print("\n" + "=" * 60)
        print("Stage 2: Training Connector + Language Model (LoRA)")
        print("=" * 60)
        print(f"  Epochs: {stage2_config.num_epochs}")
        print(f"  Batch size: {stage2_config.per_device_batch_size}")
        print(f"  Learning rate: {stage2_config.learning_rate}")
        print(f"  LoRA r={stage2_config.lora_r}, alpha={stage2_config.lora_alpha}")
        print(f"  TensorBoard: {'enabled' if self.writer else 'disabled'}")
        print("=" * 60)

        # 设置 Stage 2
        self.model.set_stage("stage2")

        # 应用 LoRA（如果启用）
        if stage2_config.lora_r > 0:
            self._apply_lora(stage2_config)

        self.model.train()
        self.model.get_trainable_parameters()

        # 创建 DataLoader
        train_loader = self._create_dataloader(
            self.train_dataset,
            batch_size=stage2_config.per_device_batch_size,
            shuffle=True,
        )

        # 计算总步数
        total_steps = (
            len(train_loader)
            * stage2_config.num_epochs
            // stage2_config.gradient_accumulation_steps
        )

        # 创建优化器和调度器
        optimizer, scheduler = self._get_optimizer_and_scheduler(
            stage2_config, total_steps
        )

        # 混合精度
        scaler = torch.amp.GradScaler('cuda', enabled=stage2_config.use_fp16)

        # 保存一次静态文件（config + tokenizer），后续 checkpoint 不再重复保存
        # 注意：Stage 2 会额外保存 LoRA 权重到 checkpoint/lora/
        self.model.save_pretrained(self.output_dir)

        # 训练循环
        stage_prefix = "stage2"
        for epoch in range(stage2_config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{stage2_config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch.get("pixel_values")

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)

                with torch.amp.autocast('cuda', enabled=stage2_config.use_fp16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]

                loss = loss / stage2_config.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % stage2_config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1

                    # TensorBoard 日志（实际优化步级）
                    current_lr = scheduler.get_last_lr()[0]
                    self._log_tb(f"{stage_prefix}/loss", loss.item() * stage2_config.gradient_accumulation_steps, self.global_step)
                    self._log_tb(f"{stage_prefix}/lr", current_lr, self.global_step)
                    self._log_tb(f"{stage_prefix}/grad_norm", grad_norm.item(), self.global_step)

                epoch_loss += loss.item() * stage2_config.gradient_accumulation_steps
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * stage2_config.gradient_accumulation_steps:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                })

                # 每 logging_steps 记录 (数据步级)
                if (step + 1) % stage2_config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    self._log_tb(f"{stage_prefix}/loss_per_step", loss.item() * stage2_config.gradient_accumulation_steps, epoch * len(train_loader) + step)

                # 定期保存（仅保存可训练权重，config/tokenizer 已在训练开始时保存；含 LoRA）
                if self.global_step > 0 and self.global_step % stage2_config.save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"stage2_step_{self.global_step}")
                    self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
                    print(f"\n[Stage2] Checkpoint saved to: {save_path}")

            # Epoch 结束保存
            save_path = os.path.join(self.output_dir, f"stage2_epoch_{epoch+1}")
            self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
            self._log_tb(f"{stage_prefix}/epoch_avg_loss", avg_loss, epoch + 1)
            print(f"\nStage 2 Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

        print("\nStage 2 training completed!")
        return self.model

    def _apply_lora(self, stage2_config: Stage2Config):
        """对语言模型应用LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=stage2_config.lora_r,
                lora_alpha=stage2_config.lora_alpha,
                lora_dropout=stage2_config.lora_dropout,
                target_modules=list(stage2_config.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )

            # 对语言模型应用LoRA
            self.model.language_model.model = get_peft_model(
                self.model.language_model.model,
                lora_config,
            )

            print(f"[LoRA] Applied LoRA to language model")
            print(f"       r={stage2_config.lora_r}, alpha={stage2_config.lora_alpha}")
            print(f"       target_modules={stage2_config.lora_target_modules}")

        except ImportError:
            print("[Warning] peft not installed, skipping LoRA. Using full fine-tuning instead.")
            print("         Install with: pip install peft")

    def train(self, stages: list = None):
        """完整训练流程"""
        if stages is None:
            stages = ["stage1", "stage2"]

        if "stage1" in stages:
            self.train_stage1()

        if "stage2" in stages:
            self.train_stage2()

        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Model saved to: {self.output_dir}")
        print(f"TensorBoard logs: {os.path.join(self.output_dir, 'tb_logs')}")
        print("=" * 60)