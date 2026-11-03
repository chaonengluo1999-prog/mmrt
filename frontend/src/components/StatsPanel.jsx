import { motion } from 'framer-motion'

export default function StatsPanel({ stats, status }) {
  return (
    <>
      {stats && (
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
          className="glass p-6">
          <h3 className="text-base font-bold mb-4 flex items-center gap-2">
            <span>📦</span> 数据集 · CSL-Daily
          </h3>
          <div className="space-y-2.5 text-sm">
            {[
              ['总样本', stats.dataset.total_samples],
              ['训练集', stats.dataset.train_samples],
              ['测试集', stats.dataset.test_samples],
              ['划分比例', stats.dataset.split_ratio],
              ['词汇表', `${stats.dataset.gloss_vocab_size} 词`],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-gray-500">{k}</span>
                <span className="font-semibold">{v}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1 }}
        className="glass p-6">
        <h3 className="text-base font-bold mb-4 flex items-center gap-2">
          <span>🧠</span> 模型信息
        </h3>
        <div className="space-y-2.5 text-sm">
          {[
            ['架构', 'MMRT'],
            ['视觉编码', 'ResNet18 + GCN'],
            ['时序建模', 'TemporalConv + BiLSTM'],
            ['解码器', 'CTC Beam Search'],
            ['翻译', 'DeepSeek LLM'],
            ['输入', 'RGB 224×224'],
            ['设备', status?.device?.toUpperCase() || 'CPU'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-gray-500">{k}</span>
              <span className="font-semibold">{v}</span>
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
        className="glass p-6">
        <h3 className="text-base font-bold mb-4 flex items-center gap-2">
          <span>⭐</span> 技术亮点
        </h3>
        <div className="space-y-2 text-sm">
          {[
            'ResNet18 + Spatial/Temporal GCN',
            '多尺度1D时序卷积',
            'CTC无需强制对齐',
            'DeepSeek三级翻译策略',
            '多方言TTS语音合成',
            'EMG肌电信号接口（预留）',
          ].map((h, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className="text-emerald-400 mt-0.5 text-xs">✓</span>
              <span className="text-gray-400">{h}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </>
  )
}
