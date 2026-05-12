# NanoVLM

**NanoVLM** — 一个精简、可学习的 Vision-Language Model（视觉语言模型）。 约 **1B 总参数**， 在单张消费级 GPU 上即可完成两阶段训练。

> 🎯 初衷：轻量 VLM，便于理解架构&训练流程

---

## 总体架构

```
┌──────────┐      ┌──────────────┐      ┌──────────────────────┐
│  Image   │────▶ │ SigLIP ViT   │────▶ │  MLP Projector       │
│  输入    │      │ (视觉编码器)  │      │  视觉 → 语言 维度     │
└──────────┘      └──────────────┘      └──────────┬───────────┘
                                                   │ visual tokens
                                                   ▼
                              ┌───────────────────────────────────┐
                              │         Qwen2-0.5B                │
                              │       (语言模型 + LoRA)            │
                              │         生成文本回答                │
                              └───────────────────────────────────┘
```

* **Vision Encoder**: SigLIP (google/siglip-so400m-patch14-384) — 约 400M 参数，384×384 输入
* **Connector**: 2层MLP — 将视觉特征投影到语言空间
* **Language Model**: Qwen2-0.5B-Instruct — 约 0.5B 参数，支持对话

---

## 模型规模

| 组件 | 参数量 (约) |
|------|------------|
| SigLIP Vision Encoder | ~400M |
| MLP Connector | ~50M |
| Qwen2-0.5B (含LoRA) | ~500M |
| **总计** | **~1B** |

其中 Stage1 可训练参数 ~50M，Stage2（LoRA r=8）可训练参数 ~60M。

---

## 项目结构

```
NanoVLM/
├── configs/                     # 配置文件
│   ├── __init__.py
│   ├── model_config.py          # 模型架构配置
│   └── training_config.py       # 训练超参数配置
├── src/nanovlm/
│   ├── __init__.py
│   ├── model/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── vision_encoder.py    # SigLIP 视觉编码器
│   │   ├── language_model.py    # Qwen2 语言模型
│   │   ├── connector.py         # MLP 连接器
│   │   └── nanovlm.py           # 完整 VLM 组装 & 加载/保存
│   ├── data/                    # 数据处理
│   │   ├── __init__.py
│   │   ├── conversation.py      # 对话模板 (Qwen/Llama格式)
│   │   └── dataset.py           # LLaVA格式数据集
│   ├── training/                # 训练系统
│   │   ├── __init__.py
│   │   └── trainer.py           # 两阶段训练器
│   ├── inference/               # 推理
│   │   ├── __init__.py
│   │   └── generator.py         # 生成器 & 交互式对话
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       └── utils.py             # 随机种子/参数统计/设备管理
├── scripts/                     # 入口脚本
│   ├── download_models.py       # 下载预训练权重
│   ├── train.py                 # 训练入口
│   └── inference.py             # 推理入口
├── models/                      # 预训练权重(需下载)
├── checkpoints/                 # 训练检查点
├── data/                        # 训练数据
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练权重

```bash
# 下载全部（~2GB）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py

# 单独下载
python scripts/download_models.py --vision_only
python scripts/download_models.py --language_only
```

### 3. 准备数据

数据格式使用 LLaVA 标准格式 JSON：

```json
[
  {
    "id": "000001",
    "image": "coco/train2017/000000000001.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\n请描述这张图片。"},
      {"from": "gpt", "value": "图中展示了一个..."}
    ]
  }
]
```

也可以用哑数据快速测试训练流程：

```bash
python scripts/train.py --create_dummy
```

### 4. 训练

```bash
# 标准两阶段训练
python scripts/train.py \
  --data_path data/llava_instruct.json \
  --image_dir data/images \
  --output_dir ./checkpoints \
  --max_seq_length 2048

# 仅训练 Stage1
python scripts/train.py --stage stage1 --data_path data/llava_558k.json

# 仅训练 Stage2
python scripts/train.py --stage stage2 --data_path data/llava_instruct.json
```

**训练两阶段**：
- **Stage 1**：冻结视觉编码器 + 冻结语言模型，只训练 MLP Connector
- **Stage 2**：冻结视觉编码器，训练 MLP Connector + 语言模型（LoRA）

### 5. 推理

```bash
# 单次推理
python scripts/inference.py \
  --model_path ./checkpoints/stage2_epoch_1 \
  --image cat.jpg \
  --question "图中有什么？"

# 交互式对话
python scripts/inference.py \
  --model_path ./checkpoints/stage2_epoch_1 \
  --interactive
```

---

## 配置说明

### 模型配置 (`configs/model_config.py`)

```python
@dataclass
class VisionConfig:
    model_name_or_path: str = "models/siglip-so400m-patch14-384"
    image_size: int = 384

@dataclass
class LanguageConfig:
    model_name_or_path: str = "models/Qwen2-0.5B-Instruct"

@dataclass
class ConnectorConfig:
    vision_hidden_size: int = 1152
    language_hidden_size: int = 896
    mlp_hidden_size: int = 2048
```

### 训练配置 (`configs/training_config.py`)

```python
@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"
    max_grad_norm: float = 1.0
    stage1: Stage1Config  # Stage1 超参
    stage2: Stage2Config  # Stage2 超参
```

---

## 技术要点

### 为什么选择 SigLIP + Qwen2-0.5B？

| 特性 | 说明 |
|------|------|
| **SigLIP** | 优于 CLIP 的对比学习视觉编码器，400M 参数，384px 输入 |
| **Qwen2-0.5B** | 轻量中文/英文 LLM，支持 Instruct 对话格式 |
| **总参数量** | ~1B，可在 RTX 3090/4090 等 24GB 显存 GPU 上完整训练 |
| **训练效率** | Stage1 仅训练 MLP(~50M)，Stage2 LoRA(~10M)大幅节省显存 |

### 两阶段训练策略

* **Stage 1：Connector 预热**
  * 只训练 MLP Projector，让视觉token和语言token对齐
  * 显存需求：~8GB（batch size=1, fp16）
  * 数据量：558K 图文对（如 LLaVA-Pretrain）

* **Stage 2：指令微调**
  * 训练 MLP Projector + LoRA（语言模型）
  * 显存需求：~12GB（batch size=1, fp16）
  * 数据量：80K~150K 高质量指令数据（如 LLaVA-Instruct）

### LoRA 配置

```python
lora_r = 8
lora_alpha = 16
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_dropout = 0.05
```

---

## 依赖

```
torch >= 2.0.0
transformers >= 4.36.0
pillow
numpy
tqdm
peft >= 0.7.0
datasets
accelerate
```

完整列表见 [`requirements.txt`](requirements.txt)

---

## 参考资料

- [LLaVA](https://github.com/haotian-liu/LLaVA) — 多模态指令跟随
- [SigLIP](https://arxiv.org/abs/2303.15343) — Sigmoid Loss for Language Image Pre-training
- [Qwen2](https://arxiv.org/abs/2309.16609) — Qwen Technical Report
- [LoRA](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of LLMs

---

## License

MIT License