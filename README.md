# MMRT - Multi-Modal Recognition Transformer

**多模态手语识别翻译系统** | Sign Language Recognition & Translation System

[![Accuracy](https://img.shields.io/badge/Gloss_Accuracy-90.05%25-brightgreen)](data/eval_results.json)
[![BLEU](https://img.shields.io/badge/BLEU--4-36.35%25-blue)](data/eval_results.json)
[![WER](https://img.shields.io/badge/WER-9.95%25-yellow)](data/eval_results.json)

---

## 📋 项目概述

MMRT 是一个基于深度学习的手语识别与翻译系统，采用多模态架构实现从视频到文本的端到端翻译。系统在 CSL-Daily 数据集上达到 **90.05%** 的 Gloss 识别准确率，满足合同要求的 ≥90% 指标。

### 核心特性

- **高精度识别**: Gloss WER 9.95%，准确率 90.05%
- **智能翻译**: DeepSeek LLM 三级翻译策略，BLEU-4 36.35%
- **实时推理**: CPU 1.2s/样本，支持 MPS/CUDA 加速
- **可视化界面**: Vite + React 现代化 Web 界面
- **完整文档**: 架构设计、API 文档、评估报告

---

## 🏗️ 系统架构

MMRT 采用 8 模块架构设计：

```
视频输入 (Video)
    ↓
[Module 4] 视觉特征提取 (ResNet18 + GCN)
    ↓
[Module 2] 多尺度时序编码 (TemporalConv)
    ↓
[Module 2] 序列建模 (BiLSTM)
    ↓
[Module 8] CTC 解码 (Beam Search)
    ↓
Gloss 序列 (Sign Language Glosses)
    ↓
[LLM] DeepSeek 三级翻译
    ↓
中文文本 (Chinese Text)
```

### 模块映射

| 模块 | 功能 | 实现文件 |
|------|------|----------|
| Module 4 | 视觉特征提取 | `modules/resnet_encoder.py` |
| Module 2 | 时序编码 | `modules/temporal_conv.py` |
| Module 2 | 序列建模 | `modules/bilstm_temporal.py` |
| Module 8 | CTC 解码 | `modules/ctc_decoder.py` |
| LLM | 后处理翻译 | `modules/llm_postprocessor.py` |

---

## 📊 评估结果

**数据集**: CSL-Daily (8:2 随机划分)  
**测试样本**: 454  
**评估时间**: 2026-04-19

### 核心指标

| 指标 | 数值 | 状态 |
|------|------|------|
| **Gloss 准确率** | **90.05%** | ✅ **达标** (≥90%) |
| Gloss WER | 9.95% | ✅ 优秀 |
| BLEU-4 | 36.35% | ✅ 良好 |
| 推理速度 | 1.19s/样本 (CPU) | ✅ 实用 |

### 翻译策略分布

- **Tier 1 (精确匹配)**: 8.6% - 训练集中完全匹配的 gloss 序列
- **Tier 2 (模糊+LLM)**: 4.0% - 编辑距离 ≤1 的模糊匹配 + LLM 精修
- **Tier 3 (LLM翻译)**: 87.4% - 检索增强的 LLM 翻译

详细评估报告: [`data/eval_results.json`](data/eval_results.json)

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- Node.js 18+ (前端)
- 8GB+ RAM
- CSL-Daily 数据集

### 安装依赖

```bash
# 1. 克隆项目
cd /path/to/mmrt

# 2. 安装 Python 依赖
pip install torch torchvision torchaudio
pip install opencv-python numpy pyyaml fastapi uvicorn edge-tts openai

# 3. 安装前端依赖
cd frontend
npm install
cd ..

# 4. 下载模型权重 (如果未包含)
# 模型文件: checkpoints/mmrt_best.pt (448MB)
```

### 数据集准备

```bash
# CSL-Daily 数据集应放置在:
~/Desktop/CSL-Daily/
  frames_512x512/
    S000002_P0004_T00/
      000000.jpg
      000001.jpg
      ...
```

### 运行评估

```bash
# 无 LLM 翻译 (仅 Gloss 识别)
python scripts/evaluate_mmrt.py --no-llm

# 完整评估 (含 LLM 翻译)
export DEEPSEEK_API_KEY=your_api_key_here
python scripts/evaluate_mmrt.py

# 快速测试 (30 样本)
python scripts/evaluate_mmrt.py --max-samples 30
```

### 启动服务

```bash
# 1. 启动后端 API (端口 8000)
python backend/main.py

# 2. 启动前端 (端口 5173)
cd frontend
npm run dev

# 3. 访问 http://localhost:5173
```

---

## 📡 API 文档

### POST `/api/predict`

手语视频识别与翻译

**请求**:
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@video.mp4"
```

**响应**:
```json
{
  "glosses": ["今天", "下雨", "爷爷", "钓", "鱼", "去", "不"],
  "text": "今天下雨，爷爷不去钓鱼。",
  "confidence": 0.92,
  "tier": 3,
  "processing_time": 1.23
}
```

### POST `/api/tts`

文本转语音 (多方言支持)

**请求**:
```json
{
  "text": "今天下雨，爷爷不去钓鱼。",
  "voice": "zh-CN-XiaoxiaoNeural"
}
```

**响应**: MP3 音频流

### GET `/api/stats`

系统统计信息

**响应**:
```json
{
  "model": "MMRT",
  "vocab_size": 2000,
  "architecture": "ResNet18 + TemporalConv + BiLSTM + CTC",
  "accuracy": 90.05,
  "dataset": "CSL-Daily"
}
```

---

## 🎨 前端界面

### 功能模块

1. **识别页** - 视频上传、实时推理、结果展示
2. **架构页** - 8 模块流程图、技术细节
3. **评估页** - 准确率报告、样本预测对比

### 技术栈

- **框架**: Vite + React 18
- **样式**: TailwindCSS
- **动画**: Framer Motion
- **构建**: 335KB JS + 27KB CSS (gzip)

### 本地开发

```bash
cd frontend
npm run dev      # 开发服务器 (热重载)
npm run build    # 生产构建
npm run preview  # 预览构建结果
```

---

## 📁 项目结构

```
mmrt/
├── backend/              # FastAPI 后端
│   └── main.py          # API 服务器
├── checkpoints/         # 模型权重
│   └── mmrt_best.pt     # 预训练模型 (448MB)
├── configs/             # 配置文件
│   ├── inference.yaml   # 推理配置
│   └── dataset.yaml     # 数据集配置
├── data/                # 数据文件
│   ├── csl_daily/       # CSL-Daily 元数据
│   │   ├── gloss_dict.npy
│   │   ├── train_8_2.json
│   │   └── test_8_2.json
│   └── eval_results.json # 评估结果
├── frontend/            # React 前端
│   ├── src/
│   │   ├── components/  # UI 组件
│   │   └── App.jsx      # 主应用
│   └── dist/            # 构建输出
├── models/              # 模型定义
│   └── mmrt_inference_model.py
├── modules/             # 核心模块
│   ├── resnet_encoder.py      # Module 4: 视觉编码
│   ├── temporal_conv.py       # Module 2: 时序编码
│   ├── bilstm_temporal.py     # Module 2: 序列建模
│   ├── ctc_decoder.py         # Module 8: CTC 解码
│   ├── llm_postprocessor.py   # LLM 翻译
│   ├── video_transforms.py    # 视频预处理
│   └── gcn_lib/               # GCN 图卷积
├── scripts/             # 工具脚本
│   └── evaluate_mmrt.py # 评估脚本
├── docs/                # 文档
│   ├── architecture.md  # 架构设计
│   └── evaluation.md    # 评估方法
└── README.md            # 本文件
```

---

## 🔬 技术细节

### 模型架构

**输入**: 视频帧序列 [T, 3, 224, 224]  
**输出**: Gloss 序列 + 中文文本

1. **视觉编码** (ResNet18 + GCN)
   - 3D 卷积提取时空特征
   - GCN 建模帧间关系
   - 输出: [T, 512] 特征序列

2. **时序编码** (Multi-scale TemporalConv)
   - K3/K5/P2 多尺度卷积
   - 捕获不同时间尺度的模式
   - 输出: [T, 1024] 时序特征

3. **序列建模** (BiLSTM)
   - 双向 LSTM 建模长程依赖
   - 2 层堆叠，隐藏层 1024
   - 输出: [T, 1024] 上下文特征

4. **CTC 解码** (Beam Search)
   - Beam width 10
   - 自动对齐与去重
   - 输出: Gloss 序列

5. **LLM 翻译** (DeepSeek)
   - Tier 1: 精确匹配 (训练集查表)
   - Tier 2: 模糊匹配 + LLM 精修
   - Tier 3: 检索增强 LLM 翻译

### 训练细节

- **数据集**: CSL-Daily 1815 训练样本
- **优化器**: Adam (lr=1e-4)
- **损失函数**: CTC Loss + Knowledge Distillation
- **数据增强**: 随机裁剪、时序抖动
- **训练时长**: ~24h (单 GPU)

---

## 🐛 故障排除

### 常见问题

**Q: MPS 设备报错 `max_pool3d_with_indices not implemented`**  
A: 设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1` 或使用 CPU/CUDA

**Q: 评估脚本找不到模块**  
A: 确保在 `mmrt/` 目录下运行，检查 `sys.path` 设置

**Q: LLM 翻译全部 Tier 0 (Fallback)**  
A: 检查 `openai` 包是否安装，API Key 是否正确

**Q: 前端无法连接后端**  
A: 检查后端是否在 8000 端口运行，CORS 配置是否正确

### 性能优化

- **CPU 推理**: 1.2s/样本，适合批量评估
- **MPS 推理**: 需要 fallback，性能不如 CPU
- **CUDA 推理**: 0.3s/样本 (推荐生产环境)

---

## 📄 许可证

本项目仅供学术研究和合同交付使用。

---

## 👥 联系方式

- **项目负责人**: 吴健
- **技术支持**: 查看 `docs/` 目录获取详细文档
- **问题反馈**: 提交 Issue 或联系项目组

---

## 🎯 交付清单

✅ **源代码**: 完整的 MMRT 系统代码  
✅ **模型权重**: `checkpoints/mmrt_best.pt` (448MB)  
✅ **评估报告**: Gloss 准确率 90.05% (≥90% ✓)  
✅ **可视化界面**: Vite + React 现代化 Web 界面  
✅ **API 文档**: FastAPI 自动生成文档 `/docs`  
✅ **技术文档**: 架构设计、评估方法、使用说明  

**合同要求达成**: ✅ 准确率 ≥90%，含源码+模型+文档+界面

---

*最后更新: 2026-04-19*
