import { motion } from 'framer-motion'

const MODULES = [
  { id: 1, name: '均衡手语理解大模型', sub: '多模态输入', color: '#6366f1', icon: '🧠' },
  { id: 2, name: '多模态信息增强', sub: 'LSTM-MLP + RL', color: '#8b5cf6', icon: '⚡' },
  { id: 3, name: '增量聚类管理', sub: 'K-means', color: '#a78bfa', icon: '📊' },
  { id: 4, name: '视觉特征提取', sub: 'ResNet18 + GCN', color: '#06b6d4', icon: '👁️' },
  { id: 5, name: 'BERT文本特征', sub: 'Transformer', color: '#14b8a6', icon: '📝' },
  { id: 6, name: 'EMG信号编解码', sub: 'Conv3x3', color: '#f59e0b', icon: '💪' },
  { id: 7, name: '多模态理解模型', sub: 'EfficientNet+EMG+Text', color: '#ec4899', icon: '🔗' },
  { id: 8, name: '跨模态Transformer', sub: '编码器-解码器', color: '#ef4444', icon: '🔄' },
]

const PIPELINE = [
  { label: '视频输入', color: '#6366f1' },
  { label: 'Module 4\nResNet18+GCN\n视觉特征', color: '#06b6d4' },
  { label: 'Module 2\n多尺度时序卷积\n+ BiLSTM', color: '#8b5cf6' },
  { label: 'Module 8\nCTC解码\n序列输出', color: '#ef4444' },
  { label: 'DeepSeek\nLLM翻译\n三级策略', color: '#14b8a6' },
  { label: '中文文本\n+ TTS语音', color: '#ec4899' },
]

export default function ArchDiagram() {
  return (
    <div className="space-y-8">
      {/* Pipeline flow */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <span className="text-2xl">🔄</span> 推理流水线
        </h3>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {PIPELINE.map((p, i) => (
            <div key={i} className="flex items-center gap-3">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
                className="w-32 py-4 px-3 rounded-xl text-center text-xs font-medium whitespace-pre-line leading-tight border"
                style={{
                  background: `${p.color}15`,
                  borderColor: `${p.color}30`,
                  color: `${p.color}dd`,
                }}>
                {p.label}
              </motion.div>
              {i < PIPELINE.length - 1 && (
                <span className="text-gray-600 text-lg">→</span>
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* 8 Modules grid */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <span className="text-2xl">🏗️</span> 8 核心模块
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {MODULES.map((m, i) => (
            <motion.div key={m.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * i }}
              whileHover={{ scale: 1.03, y: -4 }}
              className="p-5 rounded-xl border cursor-default transition-shadow hover:shadow-lg"
              style={{
                background: `${m.color}08`,
                borderColor: `${m.color}20`,
              }}>
              <div className="text-2xl mb-2">{m.icon}</div>
              <div className="text-xs text-gray-500 mb-1">Module {m.id}</div>
              <div className="text-sm font-bold mb-1" style={{ color: m.color }}>{m.name}</div>
              <div className="text-xs text-gray-500">{m.sub}</div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Technical details */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <span className="text-2xl">⚙️</span> 技术细节
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h4 className="font-semibold text-indigo-400 mb-3">视觉编码器 (Module 4)</h4>
            <ul className="space-y-1.5 text-gray-400">
              <li>• ResNet18 3D卷积骨干网络</li>
              <li>• Spatial Graph Convolution (GCN) 空间增强</li>
              <li>• Temporal Graph 时序图建模</li>
              <li>• 输出: 512维帧级特征</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-purple-400 mb-3">时序建模 (Module 2)</h4>
            <ul className="space-y-1.5 text-gray-400">
              <li>• 多尺度1D时序卷积 (K5-P2-K5-P2)</li>
              <li>• 双向LSTM (2层, 1024隐藏维度)</li>
              <li>• 权重归一化分类器</li>
              <li>• 知识蒸馏损失 (T=8)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-red-400 mb-3">序列解码 (Module 8)</h4>
            <ul className="space-y-1.5 text-gray-400">
              <li>• CTC Beam Search (beam_width=10)</li>
              <li>• 双路径解码: Conv + LSTM</li>
              <li>• 1296类词汇表 (CSL-Daily)</li>
              <li>• 无需强制对齐</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-teal-400 mb-3">LLM翻译</h4>
            <ul className="space-y-1.5 text-gray-400">
              <li>• DeepSeek API 三级翻译策略</li>
              <li>• Tier 1: 精确匹配 (100%置信度)</li>
              <li>• Tier 2: 模糊匹配+LLM精修 (85%)</li>
              <li>• Tier 3: 检索增强LLM翻译 (60%)</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
