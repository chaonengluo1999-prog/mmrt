# MMRT 系统架构

## 架构概览

MMRT (Multi-Modal Recognition Transformer) 采用 8 模块设计，实现从手语视频到中文文本的端到端翻译。

```
视频输入 (Video Frames)
    ↓
[Module 4] 视觉特征提取
    ResNet18 3D CNN + GCN
    输出: [T, 512] 特征序列
    ↓
[Module 2] 多尺度时序编码
    TemporalConv (K3/K5/P2)
    输出: [T, 1024] 时序特征
    ↓
[Module 2] 序列建模
    BiLSTM (2层, 双向)
    输出: [T, 1024] 上下文特征
    ↓
[Module 8] CTC 解码
    Beam Search (width=10)
    输出: Gloss 序列
    ↓
[LLM] DeepSeek 三级翻译
    Tier 1: 精确匹配
    Tier 2: 模糊匹配 + LLM
    Tier 3: 检索增强 LLM
    输出: 中文文本
```

## 模块详解

### Module 4: 视觉特征提取

**文件**: `modules/resnet_encoder.py`

- **ResNet18 3D CNN**: 提取时空特征
  - 输入: [B, 3, T, 224, 224]
  - 3D 卷积核捕获时序信息
  - 输出: [B, T, 512]

- **GCN (Graph Convolutional Network)**: 建模帧间关系
  - 位置编码 (Positional Embedding)
  - 图卷积层 (gcn_lib/)
  - 增强时序一致性

### Module 2: 多尺度时序编码

**文件**: `modules/temporal_conv.py`

- **Multi-scale TemporalConv**: 捕获不同时间尺度的模式
  - K3: 短期依赖 (kernel_size=3)
  - K5: 中期依赖 (kernel_size=5)
  - P2: 长期依赖 (pooling + conv)
  - 输出: [B, 1024, T]

### Module 2: 序列建模

**文件**: `modules/bilstm_temporal.py`

- **BiLSTM**: 双向长短期记忆网络
  - 2 层堆叠
  - 隐藏层维度: 1024
  - 双向建模上下文
  - 输出: [T, B, 1024]

### Module 8: CTC 解码

**文件**: `modules/ctc_decoder.py`

- **CTC Beam Search**: 序列解码
  - Beam width: 10
  - 自动对齐 (无需帧级标注)
  - 去重与空白符处理
  - 输出: Gloss 序列 (List[str])

- **Fallback**: 如果 ctcdecode 不可用，使用贪心解码

### LLM 后处理

**文件**: `modules/llm_postprocessor.py`

- **Tier 1 (精确匹配)**: 8.6%
  - 训练集查表
  - 100% 准确率

- **Tier 2 (模糊匹配 + LLM)**: 4.0%
  - 编辑距离 ≤1
  - LLM 精修翻译

- **Tier 3 (检索增强 LLM)**: 87.4%
  - Jaccard 相似度检索 Top-6
  - Few-shot LLM 翻译
  - DeepSeek API

## 数据流

### 推理流程

1. **视频预处理** (`modules/video_transforms.py`)
   - 读取视频帧
   - Resize 256x256
   - CenterCrop 224x224
   - 归一化 [-1, 1]

2. **特征提取**
   - ResNet18 提取帧级特征
   - GCN 建模时序关系

3. **时序编码**
   - TemporalConv 多尺度卷积
   - BiLSTM 序列建模

4. **解码**
   - CTC Beam Search
   - 输出 Gloss 序列

5. **翻译**
   - GlossTranslator 三级策略
   - 输出中文文本

### 训练流程 (已完成)

- **损失函数**: CTC Loss + Knowledge Distillation
- **优化器**: Adam (lr=1e-4)
- **数据集**: CSL-Daily 1815 训练样本
- **训练时长**: ~24h (单 GPU)

## 文件映射

| 功能模块 | 实现文件 | 说明 |
|---------|---------|------|
| 推理模型 | `models/mmrt_inference_model.py` | 完整推理流程 |
| 视觉编码 | `modules/resnet_encoder.py` | ResNet18 + GCN |
| 时序编码 | `modules/temporal_conv.py` | Multi-scale TemporalConv |
| 序列建模 | `modules/bilstm_temporal.py` | BiLSTM |
| CTC 解码 | `modules/ctc_decoder.py` | Beam Search |
| 损失函数 | `modules/criterions.py` | CTC + KD |
| 视频处理 | `modules/video_transforms.py` | 预处理 |
| LLM 翻译 | `modules/llm_postprocessor.py` | 三级翻译 |
| GCN 图卷积 | `modules/gcn_lib/` | 图神经网络 |
| 后端 API | `backend/main.py` | FastAPI 服务 |
| 评估脚本 | `scripts/evaluate_mmrt.py` | 性能评估 |

## 性能指标

- **Gloss 准确率**: 90.05% (WER 9.95%)
- **BLEU-4**: 36.35%
- **推理速度**: 1.19s/样本 (CPU)
- **模型大小**: 448MB
- **参数量**: 37M

## 技术栈

- **深度学习**: PyTorch 2.0+
- **后端**: FastAPI + Uvicorn
- **前端**: Vite + React 18
- **LLM**: DeepSeek API (OpenAI SDK)
- **TTS**: edge-tts

## 部署架构

```
Frontend (React)
    ↓ HTTP
Backend (FastAPI:8000)
    ↓
MMRT Model (PyTorch)
    ↓
DeepSeek API
```

---

*最后更新: 2026-04-20*
