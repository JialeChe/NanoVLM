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
        sampler=None,
    ) -> DataLoader:
        """创建 DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """自定义 batch 合并函数

        支持两种模式：
        - 原始单图：pixel_values 形状一致 → stack 为 (B, 3, H, W)
        - AnyRes：pixel_values 形状各异 → cat 为 (total_sub_images, 3, H, W)
                              + image_counts: List[int]
        """
        keys = batch[0].keys()
        collated = {}

        # 检测是否是 AnyRes 模式（存在 image_counts 字段）
        is_anyres = "image_counts" in keys

        for key in keys:
            values = [item[key] for item in batch if key in item]

            if len(values) == 0:
                continue

            if key == "pixel_values":
                if is_anyres:
                    # AnyRes 模式：每样本的 pixel_values 形状为 (N_i, 3, H, W)
                    # → 沿 dim=0 拼接为 (total_sub_images, 3, H, W)
                    collated[key] = torch.cat(values, dim=0)
                else:
                    # 原始单图模式：每样本 (3, H, W) → stack 为 (B, 3, H, W)
                    collated[key] = torch.stack(values)

            elif key == "image_counts":
                # 记录每个样本的 sub-image 数量（List[int]）
                collated[key] = values

            elif isinstance(values[0], torch.Tensor):
                # 文本 tensor 的 padding（保持原有逻辑）
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

    def _ensure_fp32_trainable_params(self, context: str):
        """Keep trainable parameters in FP32 when using GradScaler with fp16 AMP."""
        converted = 0
        converted_params = 0
        for param in self.model.parameters():
            if param.requires_grad and param.dtype == torch.float16:
                param.data = param.data.float()
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
                converted += param.numel()
                converted_params += 1

        if converted_params > 0:
            print(
                f"[Trainer] Converted {converted_params} trainable {context} tensors "
                f"({converted:,} params) from fp16 to fp32 for AMP GradScaler"
            )

    def _save_training_state(self, save_dir: str, optimizer, scheduler, scaler, step_in_epoch: int):
        """保存优化器/调度器/混合精度缩放器等训练状态，用于断点续训"""
        training_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'step_in_epoch': step_in_epoch,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler.pt"))

    def _load_training_state(self, checkpoint_dir: str, optimizer, scheduler, scaler) -> int:
        """加载训练状态，返回当前 epoch 内的 batch 位置"""
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu")
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location="cpu")
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location="cpu")
        )
        scaler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location="cpu")
        )
        self.global_step = state['global_step']
        self.current_epoch = state['current_epoch']
        return state['step_in_epoch']

    def _log_tb(self, tag: str, value: float, step: int):
        """记录标量到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def train_stage1(self, stage1_config: Stage1Config = None, resume_from: str = None):
        """
        Stage 1 训练：仅训练MLP投影层

        冻结：Vision Encoder + Language Model
        训练：Connector (MLP Projector)

        Args:
            stage1_config: Stage1 训练配置
            resume_from: 断点续训的 checkpoint 目录路径（如 ./checkpoints/stage1_step_25000）
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
        if resume_from:
            print(f"  Resume from: {resume_from}")
        print("=" * 60)

        # 设置 Stage 1
        self.model.set_stage("stage1")
        self.model.train()
        self.model.get_trainable_parameters()
        if stage1_config.use_fp16:
            self._ensure_fp32_trainable_params("stage1")

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

        # ---- 断点续训 ----
        batches_to_skip = 0
        if resume_from is not None:
            connector_path = os.path.join(resume_from, "connector.bin")
            if os.path.exists(connector_path):
                self.model.connector.load_state_dict(
                    torch.load(connector_path, map_location=self.device),
                    strict=False,  # 兼容旧 checkpoint（无 norm.weight）
                )
                print(f"[Trainer] Loaded connector weights from: {connector_path}")
            else:
                print(f"[Warning] connector.bin not found in {resume_from}, skipping weight load")

            state_path = os.path.join(resume_from, "training_state.pt")
            opt_path = os.path.join(resume_from, "optimizer.pt")
            if os.path.exists(state_path) and os.path.exists(opt_path):
                self._load_training_state(
                    resume_from, optimizer, scheduler, scaler
                )
                print(f"[Trainer] Resumed at global_step={self.global_step}, epoch={self.current_epoch}")
            else:
                # 旧版 checkpoint：尝试从目录名解析 step（如 stage1_step_25000）
                import re
                match = re.search(r'step[_\s](\d+)', os.path.basename(resume_from), re.IGNORECASE)
                if match:
                    self.global_step = int(match.group(1))
                    print(f"[Trainer] Parsed global_step={self.global_step} from checkpoint path")
                    print(f"[Trainer] Optimizer/scheduler will restart from scratch (LR schedule resets)")
                else:
                    print(f"[Trainer] Could not determine step; starting from beginning (global_step=0)")

            batches_to_skip = self.global_step * stage1_config.gradient_accumulation_steps
        # ---- 结束断点续训 ----

        # 保存一次静态文件（config + tokenizer），后续 checkpoint 不再重复保存
        if resume_from is None:
            self.model.save_pretrained(self.output_dir)
        else:
            if not os.path.exists(os.path.join(self.output_dir, "config.json")):
                self.model.save_pretrained(self.output_dir)

        # 训练循环
        stage_prefix = "stage1"
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, stage1_config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            # 断点续训首个 epoch：跳过已处理 batch；后续 epoch 使用全量数据
            if batches_to_skip > 0 and epoch == start_epoch:
                from torch.utils.data import SubsetRandomSampler
                indices = list(range(batches_to_skip, len(self.train_dataset)))
                epoch_loader = self._create_dataloader(
                    self.train_dataset,
                    batch_size=stage1_config.per_device_batch_size,
                    sampler=SubsetRandomSampler(indices),
                )
                print(f"[Trainer] Skipping {batches_to_skip} batches, training on {len(indices)} remaining samples")
            else:
                epoch_loader = self._create_dataloader(
                    self.train_dataset,
                    batch_size=stage1_config.per_device_batch_size,
                    shuffle=True,
                )

            progress_bar = tqdm(epoch_loader, desc=f"Stage1 Epoch {epoch+1}/{stage1_config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                # 将数据移到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch.get("pixel_values")

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)

                image_counts = batch.get("image_counts", None)

                # 混合精度前向传播
                with torch.amp.autocast('cuda', enabled=stage1_config.use_fp16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_counts=image_counts,
                    )
                    loss = outputs["loss"]

                # 检测到 NaN 时跳过本次更新，避免污染权重
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARNING] NaN/Inf loss detected at step {step}, epoch {epoch}. Skipping.")
                    print(f"  pixel_values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  image_counts: {image_counts}")
                    optimizer.zero_grad()
                    continue

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

                # 定期保存（可训练权重 + 优化器/调度器状态，用于断点续训）
                if self.global_step > 0 and self.global_step % stage1_config.save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"stage1_step_{self.global_step}")
                    self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
                    self._save_training_state(save_path, optimizer, scheduler, scaler, step)
                    print(f"\n[Stage1] Checkpoint saved to: {save_path} (step {self.global_step})")

            # Epoch 结束保存
            save_path = os.path.join(self.output_dir, f"stage1_epoch_{epoch+1}")
            self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
            self._save_training_state(save_path, optimizer, scheduler, scaler, step)
            self._log_tb(f"{stage_prefix}/epoch_avg_loss", avg_loss, epoch + 1)
            print(f"\nStage 1 Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

        print("\nStage 1 training completed!")
        return self.model

    def train_stage2(self, stage2_config: Stage2Config = None, resume_from: str = None):
        """
        Stage 2 训练：训练MLP + 语言模型(LoRA)

        冻结：Vision Encoder
        训练：Connector + Language Model (LoRA)

        Args:
            stage2_config: Stage2 训练配置
            resume_from: 断点续训的 checkpoint 目录。
                         - Stage 1 checkpoint: 仅加载 connector，LoRA 从头初始化
                         - Stage 2 checkpoint: 加载 connector + LoRA + 训练状态，精确续训
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
        if resume_from:
            print(f"  Resume from: {resume_from}")
        print("=" * 60)

        # ── 设置 Stage 2（必须在加载权重前完成，LoRA 需要先包装模型）──────
        self.model.set_stage("stage2")

        # 应用 LoRA（如果启用）— 先建立 PEFT wrapper，再用 checkpoint 权重覆盖
        if stage2_config.lora_r > 0:
            self._apply_lora(stage2_config)

        self.model.train()
        self.model.get_trainable_parameters()
        if stage2_config.use_fp16:
            self._ensure_fp32_trainable_params("stage2")

        # NaN 权重检测
        nan_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                nan_params.append(name)
        if nan_params:
            print(f"[CRITICAL] {len(nan_params)} NaN params detected BEFORE training: {nan_params[:5]}...")
            print(f"  → Weights are corrupted. Must reload checkpoint or reinitialize LoRA.")
        else:
            print(f"[Trainer] Weight check OK — no NaN in trainable params")

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

        # ── 断点续训 ──────────────────────────────────────────────────
        batches_to_skip = 0
        if resume_from is not None:
            # 1) 加载 connector 权重
            connector_path = os.path.join(resume_from, "connector.bin")
            if os.path.exists(connector_path):
                self.model.connector.load_state_dict(
                    torch.load(connector_path, map_location=self.device),
                    strict=False,
                )
                print(f"[Trainer] Loaded connector weights from: {connector_path}")
            else:
                print(f"[Warning] connector.bin not found in {resume_from}")

            # 2) 加载 LoRA 权重（仅 Stage 2 checkpoint 才包含 lora/ 子目录）
            lora_dir = os.path.join(resume_from, "lora")
            if os.path.exists(lora_dir):
                try:
                    from peft import PeftModel
                    self.model.language_model.model = PeftModel.from_pretrained(
                        self.model.language_model.model, lora_dir
                    )
                    print(f"[Trainer] Loaded LoRA weights from: {lora_dir}")
                except ImportError:
                    print("[Warning] peft not installed, skipping LoRA weight load")
            else:
                # 从 Stage 1 checkpoint 恢复 → 无 LoRA 权重，使用刚初始化的
                print(f"[Trainer] lora/ not found → loading Stage 1 connector only; "
                      f"LoRA adapter initialized from scratch")

            # 3) 加载训练状态（optimizer / scheduler / scaler / global_step）
            state_path = os.path.join(resume_from, "training_state.pt")
            opt_path = os.path.join(resume_from, "optimizer.pt")
            if os.path.exists(state_path) and os.path.exists(opt_path):
                self._load_training_state(resume_from, optimizer, scheduler, scaler)
                print(f"[Trainer] Resumed at global_step={self.global_step}, epoch={self.current_epoch}")
            else:
                # 兼容旧 checkpoint（无训练状态）：从目录名解析 step
                import re
                match = re.search(r'step[_\s](\d+)', os.path.basename(resume_from), re.IGNORECASE)
                if match:
                    self.global_step = int(match.group(1))
                    print(f"[Trainer] Parsed global_step={self.global_step} from checkpoint path")
                    print(f"[Trainer] Optimizer/scheduler restarting from scratch (LR schedule resets)")
                else:
                    print(f"[Trainer] No training state found; starting from beginning (global_step=0)")

            batches_to_skip = self.global_step * stage2_config.gradient_accumulation_steps
        # ── 结束断点续训 ──────────────────────────────────────────────

        # 保存一次静态文件（config + tokenizer），后续 checkpoint 不再重复保存
        if resume_from is None:
            self.model.save_pretrained(self.output_dir)
        else:
            if not os.path.exists(os.path.join(self.output_dir, "config.json")):
                self.model.save_pretrained(self.output_dir)

        # ── 训练循环 ──────────────────────────────────────────────────
        stage_prefix = "stage2"
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, stage2_config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            # 断点续训首个 epoch：跳过已处理的 batch
            if batches_to_skip > 0 and epoch == start_epoch:
                from torch.utils.data import SubsetRandomSampler
                indices = list(range(batches_to_skip, len(self.train_dataset)))
                epoch_loader = self._create_dataloader(
                    self.train_dataset,
                    batch_size=stage2_config.per_device_batch_size,
                    sampler=SubsetRandomSampler(indices),
                )
                print(f"[Trainer] Skipping {batches_to_skip} batches, "
                      f"training on {len(indices)} remaining samples")
            else:
                epoch_loader = self._create_dataloader(
                    self.train_dataset,
                    batch_size=stage2_config.per_device_batch_size,
                    shuffle=True,
                )

            progress_bar = tqdm(epoch_loader, desc=f"Stage2 Epoch {epoch+1}/{stage2_config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch.get("pixel_values")
                image_counts = batch.get("image_counts", None)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)

                with torch.amp.autocast('cuda', enabled=stage2_config.use_fp16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_counts=image_counts,
                    )
                    loss = outputs["loss"]

                # 检测到 NaN 时跳过本次更新
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARNING] NaN/Inf loss at Stage2 step {step}, epoch {epoch}. Skipping.")
                    print(f"  pixel_values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  image_counts: {image_counts}")
                    optimizer.zero_grad()
                    continue

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

                # 定期保存（可训练权重 + LoRA + 优化器/调度器状态，用于断点续训）
                if self.global_step > 0 and self.global_step % stage2_config.save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"stage2_step_{self.global_step}")
                    self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
                    self._save_training_state(save_path, optimizer, scheduler, scaler, step)
                    print(f"\n[Stage2] Checkpoint saved to: {save_path} (step {self.global_step})")

            # Epoch 结束保存
            save_path = os.path.join(self.output_dir, f"stage2_epoch_{epoch+1}")
            self.model.save_pretrained(save_path, save_config=False, save_tokenizer=False)
            self._save_training_state(save_path, optimizer, scheduler, scaler, step)
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

    def train(self, stages: list = None, resume_from_checkpoint: str = None):
        """完整训练流程"""
        if stages is None:
            stages = ["stage1", "stage2"]

        if "stage1" in stages:
            self.train_stage1(resume_from=resume_from_checkpoint)

        if "stage2" in stages:
            # stage2 可单独从 stage1 checkpoint 恢复 connector 权重
            stage2_resume = resume_from_checkpoint if "stage1" not in stages else None
            self.train_stage2(resume_from=stage2_resume)

        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Model saved to: {self.output_dir}")
        print(f"TensorBoard logs: {os.path.join(self.output_dir, 'tb_logs')}")
        print("=" * 60)
