# PPAIN: Privacy-Preserving Adaptive Inpainting Network

**[中文说明](#中文) | [English](#english)**

---

## English

### Overview

**PPAIN** (Privacy-Preserving Adaptive Inpainting Network) is a deep learning framework for content-level privacy protection in aerial/drone imagery. Given an image and a binary mask specifying regions to protect, PPAIN generates a **sanitized image** that removes sensitive implicit information while preserving visual quality.

**Key idea**: PPAIN learns to inpaint sensitive regions in a way that not only looks realistic but also makes the content **undetectable by object detectors** — trained with a frozen YOLO11 as the attack signal.

### Architecture

```
Input Image + Mask → Context Encoder → Bottleneck → Adaptive Decoder → Sanitized Image
                            ↑                                    ↑
                            └──────────── Skip ─────────────────┘

Sensitivity Map (S) feeds into decoder at 3 stages via Sensitivity-Guided Attention (SGA)
```

| Component | Description |
|-----------|-------------|
| **Context Encoder** | Multi-scale CNN (1/2, 1/4 resolution) for spatial feature extraction |
| **Bottleneck** | Feature pyramid with attention-based fusion |
| **Sensitivity-Guided Attention (SGA)** | 3-stage attention modules conditioned on privacy sensitivity map |
| **Privacy Discriminator** | PatchGAN-style discriminator for adversarial training |

### Sensitivity Scoring

For each region, compute sensitivity score as:

```
S(o_i) = w_c · φ_s · φ_p · (0.5 + 0.5 · φ_d)
```

- **w_c**: Category weight (identity=0.9, text=0.7, location=0.85, etc.)
- **φ_p**: Position score, centered regions are more sensitive
- **φ_d**: Detection confidence from YOLO11
- **φ_s**: Size score, moderate-sized regions are more sensitive

### Training Loss

```
L_G = λ_rec · L_rec + λ_det · L_det + λ_adv · L_adv

L_rec  = L1 loss on non-masked regions
L_det  = max(0, YOLO11_confidence - ε)   # Detection suppression
L_adv  = -E[D(I_san)]                    # WGAN-GP adversarial

λ_rec=1.0, λ_det=0.5, λ_adv=0.1
```

**Two-phase training**:
1. **Phase 1 (Pretrain)**: 10 epochs, L1 reconstruction only
2. **Phase 2 (Adversarial)**: 20 epochs, WGAN-GP with detection suppression

### Usage

**Train PPAIN**:

```bash
# Single GPU training (Phase 1 → Phase 2)
python scripts/train_ppain.py \
    --data_dir datasets/VisDrone/images/train \
    --epochs_pretrain 10 \
    --epochs_train 20 \
    --batch_size 8

# Evaluate
python scripts/evaluate.py \
    --data_dir datasets/VisDrone/images/val \
    --checkpoint checkpoints/ppain_gen_final.pth
```

**Run inference**:

```python
import torch
from networks.ppain import PPAINGenerator

# Load model
generator = PPAINGenerator(in_channels=4, base_channels=64)
ckpt = torch.load('checkpoints/ppain_gen_final.pth', map_location='cpu')
generator.load_state_dict(ckpt)
generator.eval()

# Sanitize an image (B,4,H,W): [image, mask] concatenated
with torch.no_grad():
    image = torch.randn(1, 3, 256, 256)
    mask = (torch.rand(1, 1, 256, 256) > 0.7).float()
    model_input = torch.cat([image, mask], dim=1)          # (B,4,H,W)
    sanitized = generator(model_input)                    # (B,3,H,W)
    I_san = image * (1 - mask) + sanitized * mask         # blend
```

**Demo with visualization**:

```bash
# Process a batch of images
python scripts/demo.py --mode batch --image_dir datasets/VisDrone/images/train --max_images 20

# Or single image
python scripts/demo.py --mode image --image datasets/VisDrone/images/train/0000002_00005_d_0000014.jpg
```

### Dataset

**VisDrone2019** (aerial/drone imagery):
- Training: 6,471 images
- Validation: 548 images
- Test: 1,610 images

The dataset is extracted from `basic_exp_code_vis+aitod.zip`. Each image is processed with YOLO11 for object detection; bounding boxes are converted to masks and sensitivity scores using the formula above.

### Model Parameters

| Model | Parameters |
|-------|-----------|
| PPAINGenerator | ~10.8M |
| RegionDiscriminator | ~0.7M |
| **Total** | **~11.5M** |

### Citation

```bibtex
@article{ppain2025,
  title={PPAIN: Privacy-Preserving Adaptive Inpainting Network for Content-Level Protection in Aerial Imagery},
  author={PPAIN Authors},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025}
}
```

### License

MIT License

---

## 中文

### 概述

**PPAIN**（隐私保护自适应图像修复网络）是用于航拍/无人机图像内容级隐私保护的深度学习框架。给定一张图片和一个二值掩码（指定需要保护的区域），PPAIN 生成一张**脱敏图像**，在移除敏感隐式信息的同时保持视觉质量。

**核心思想**：PPAIN 不仅让修复区域看起来真实，还能让修复后的内容**被目标检测器无法识别** —— 通过冻结的 YOLO11 作为攻击信号来训练。

### 架构

```
输入图像 + 掩码 → 上下文编码器 → 瓶颈层 → 自适应解码器 → 脱敏图像
                        ↑                              ↑
                        └────────── 跳跃连接 ─────────────┘

敏感度图 (S) 通过敏感性引导注意力 (SGA) 在解码器 3 个阶段注入
```

| 组件 | 说明 |
|------|------|
| **上下文编码器** | 多尺度 CNN（1/2、1/4 分辨率）提取空间特征 |
| **瓶颈层** | 基于注意力融合的特征金字塔 |
| **敏感性引导注意力 (SGA)** | 3 阶段注意力模块，以敏感度图为条件 |
| **隐私判别器** | PatchGAN 风格判别器，用于对抗训练 |

### 敏感度评分

对每个区域，敏感度评分计算公式为：

```
S(o_i) = w_c · φ_s · φ_p · (0.5 + 0.5 · φ_d)
```

- **w_c**: 类别权重（身份=0.9，文本=0.7，位置=0.85 等）
- **φ_p**: 位置分数，中心区域敏感度更高
- **φ_d**: YOLO11 检测置信度
- **φ_s**: 大小分数，中等大小区域更敏感

### 训练损失

```
L_G = λ_rec · L_rec + λ_det · L_det + λ_adv · L_adv

L_rec  = 非掩码区域的 L1 损失
L_det  = max(0, YOLO11_confidence - ε)  # 检测抑制
L_adv  = -E[D(I_san)]                   # WGAN-GP 对抗损失

λ_rec=1.0, λ_det=0.5, λ_adv=0.1
```

**两阶段训练**：
1. **阶段 1（预训练）**：10 轮，仅 L1 重构损失
2. **阶段 2（对抗训练）**：20 轮，WGAN-GP + 检测抑制

### 使用方法

**训练 PPAIN**：

```bash
# 单卡训练（阶段1 → 阶段2）
python scripts/train_ppain.py \
    --data_dir datasets/VisDrone/images/train \
    --epochs_pretrain 10 \
    --epochs_train 20 \
    --batch_size 8

# 评估
python scripts/evaluate.py \
    --data_dir datasets/VisDrone/images/val \
    --checkpoint checkpoints/ppain_gen_final.pth
```

**推理示例**：

```python
import torch
from networks.ppain import PPAINGenerator

# 加载模型
generator = PPAINGenerator(in_channels=4, base_channels=64)
ckpt = torch.load('checkpoints/ppain_gen_final.pth', map_location='cpu')
generator.load_state_dict(ckpt)
generator.eval()

# 脱敏（输入 B,4,H,W：[图像, 掩码] 拼接）
with torch.no_grad():
    image = torch.randn(1, 3, 256, 256)
    mask = (torch.rand(1, 1, 256, 256) > 0.7).float()
    model_input = torch.cat([image, mask], dim=1)     # (B,4,H,W)
    sanitized = generator(model_input)               # (B,3,H,W)
    I_san = image * (1 - mask) + sanitized * mask     # 融合
```

**可视化演示**：

```bash
# 批量处理
python scripts/demo.py --mode batch --image_dir datasets/VisDrone/images/train --max_images 20

# 单张图片
python scripts/demo.py --mode image --image datasets/VisDrone/images/train/0000002_00005_d_0000014.jpg
```

### 数据集

**VisDrone2019**（航拍图像）：
- 训练集：6,471 张图片
- 验证集：548 张图片
- 测试集：1,610 张图片

训练集、验证集、测试集均来自 VisDrone2019 数据集。每张图片经过 YOLO11 目标检测，检测框转换为掩码并按论文公式计算敏感度图。

### 模型参数量

| 模型 | 参数量 |
|------|--------|
| PPAINGenerator | ~1080万 |
| RegionDiscriminator | ~66万 |
| **总计** | **~1146万** |

### 引用

如果这个项目对您的研究有帮助，请引用我们的论文：

> **[待发表]** PPAIN: Privacy-Preserving Adaptive Inpainting Network for Content-Level Protection in Aerial Imagery. *Engineering Applications of Artificial Intelligence*.

### 许可证

MIT License