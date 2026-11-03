import { motion } from 'framer-motion'

const EVAL_DATA = {
  accuracy: 90.05,
  wer: 9.95,
  bleu4: 36.35,
  tier1_pct: 8.6,
  tier2_pct: 4.0,
  tier3_pct: 87.4,
  fallback_pct: 0.0,
  total_test: 454,
  correct: 409,
  dataset: 'CSL-Daily',
  split: '8:2 随机划分',
  model: 'MMRT (ResNet18 + TemporalConv + BiLSTM + CTC)',
  samples: [
    { glosses: ['今天', '下雨', '爷爷', '钓', '鱼', '去', '不'], pred: '今天下雨，爷爷不去钓鱼。', ref: '今天下雨，爷爷不去钓鱼。', tier: 3, bleu: 1.0 },
    { glosses: ['谢谢'], pred: '谢谢。', ref: '谢谢！', tier: 2, bleu: 0.85 },
    { glosses: ['小', '明', '脑子', '聪明'], pred: '小明很聪明。', ref: '小明脑子很聪明。', tier: 3, bleu: 0.72 },
    { glosses: ['网站', '站', '封面', '东西', '多'], pred: '这个网站的首页内容很丰富。', ref: '网络上有许多杂志封面图片素材。', tier: 3, bleu: 0.0 },
    { glosses: ['怀孕', '去', '抽血', '天', '2'], pred: '怀孕后去抽血，第二天拿报告。', ref: '病人空腹到医院抽血，两天后拿报告。', tier: 3, bleu: 0.22 },
  ]
}

function Bar({ value, max = 100, color }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2 bg-white/[0.06] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(value / max) * 100}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
      <span className="text-sm font-bold w-12 text-right" style={{ color }}>{value}%</span>
    </div>
  )
}

export default function EvalReport() {
  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Gloss 准确率', value: `${EVAL_DATA.accuracy}%`, color: '#34d399', icon: '🎯' },
          { label: 'Gloss WER', value: `${EVAL_DATA.wer}%`, color: '#fbbf24', icon: '📉' },
          { label: 'BLEU-4', value: `${EVAL_DATA.bleu4}%`, color: '#6366f1', icon: '📊' },
          { label: '测试样本', value: EVAL_DATA.total_test.toLocaleString(), color: '#8b5cf6', icon: '📋' },
        ].map((c, i) => (
          <motion.div key={i}
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="glass p-5 text-center">
            <div className="text-2xl mb-2">{c.icon}</div>
            <div className="text-2xl font-black mb-1" style={{ color: c.color }}>{c.value}</div>
            <div className="text-xs text-gray-500">{c.label}</div>
          </motion.div>
        ))}
      </div>

      {/* Translation tier breakdown */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <span>📊</span> 翻译策略分布
        </h3>
        <div className="space-y-4">
          {[
            { label: 'Tier 1 精确匹配', value: EVAL_DATA.tier1_pct, color: '#34d399' },
            { label: 'Tier 2 模糊+LLM', value: EVAL_DATA.tier2_pct, color: '#6366f1' },
            { label: 'Tier 3 LLM翻译', value: EVAL_DATA.tier3_pct, color: '#8b5cf6' },
            { label: '回退拼接', value: EVAL_DATA.fallback_pct, color: '#f87171' },
          ].map((t, i) => (
            <div key={i}>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="text-gray-400">{t.label}</span>
              </div>
              <Bar value={t.value} color={t.color} />
            </div>
          ))}
        </div>
      </motion.div>

      {/* Eval config */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-5 flex items-center gap-2">
          <span>⚙️</span> 评估配置
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          {[
            ['数据集', EVAL_DATA.dataset],
            ['划分方式', EVAL_DATA.split],
            ['评估指标', 'Gloss Accuracy (1-WER) + BLEU-4'],
            ['模型', EVAL_DATA.model],
            ['翻译策略', 'DeepSeek 三级翻译'],
            ['目标准确率', '≥ 90%  ✓ 达标'],
          ].map(([k, v]) => (
            <div key={k} className="flex gap-3">
              <span className="text-gray-500 w-24 shrink-0">{k}</span>
              <span className="text-gray-200">{v}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Sample predictions */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="glass p-8">
        <h3 className="text-xl font-bold mb-5 flex items-center gap-2">
          <span>🔍</span> 样本预测示例
        </h3>
        <div className="space-y-3">
          {EVAL_DATA.samples.map((s, i) => (
            <div key={i}
              className={`p-4 rounded-xl border text-sm
                ${s.bleu >= 0.5
                  ? 'bg-emerald-500/5 border-emerald-500/15'
                  : 'bg-red-500/5 border-red-500/15'}`}>
              <div className="flex flex-wrap items-start gap-3 mb-2">
                <div className="flex flex-wrap gap-1">
                  {s.glosses.map((g, j) => (
                    <span key={j} className="tag tag-blue">{g}</span>
                  ))}
                </div>
                <span className={`tag ${['tag-green','tag-green','tag-blue','tag-purple'][s.tier]}`}>
                  Tier {s.tier}
                </span>
                <span className={s.bleu >= 0.5 ? 'text-emerald-400' : 'text-red-400'}>
                  BLEU {(s.bleu * 100).toFixed(0)}%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div><span className="text-gray-500">预测: </span><span className="text-gray-200">{s.pred}</span></div>
                <div><span className="text-gray-500">参考: </span><span className="text-gray-200">{s.ref}</span></div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
