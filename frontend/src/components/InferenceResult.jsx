import { motion } from 'framer-motion'
import ConfidenceGauge from './ConfidenceGauge'
import TTSPanel from './TTSPanel'

const TIER_LABELS = ['回退拼接', '精确匹配', '模糊+LLM', 'LLM翻译']
const TIER_COLORS = ['tag-pink', 'tag-green', 'tag-blue', 'tag-purple']

export default function InferenceResult({ result }) {
  if (!result) return null

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      className="glass p-8">
      <h3 className="text-xl font-bold mb-5 flex items-center gap-2">
        <span className="text-green-400 text-2xl">✅</span> 识别结果
      </h3>

      <div className="bg-gradient-to-br from-indigo-950/40 to-purple-950/40 rounded-2xl p-6 mb-5 border border-indigo-500/15">
        {/* Main result */}
        <div className="text-3xl md:text-4xl font-black grad-text mb-4 leading-snug break-all">
          {result.predicted_text || '（无法识别）'}
        </div>

        {/* Glosses */}
        {result.predicted_glosses?.length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-gray-500 mb-1.5">识别词汇 (Glosses)</div>
            <div className="flex flex-wrap gap-1.5">
              {result.predicted_glosses.map((g, i) => (
                <span key={i} className="tag tag-blue">{g}</span>
              ))}
            </div>
          </div>
        )}

        {/* Conv glosses comparison */}
        {result.conv_glosses?.length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-gray-500 mb-1.5">卷积路径 (Conv)</div>
            <div className="flex flex-wrap gap-1.5">
              {result.conv_glosses.map((g, i) => (
                <span key={i} className="tag tag-cyan">{g}</span>
              ))}
            </div>
          </div>
        )}

        {/* Metrics row */}
        <div className="flex flex-wrap items-center gap-4 mb-5">
          <span className="text-sm text-gray-400">
            ⏱️ {result.processing_time}s
          </span>
          {result.translation_tier != null && (
            <span className={`tag ${TIER_COLORS[result.translation_tier] || 'tag-blue'}`}>
              {TIER_LABELS[result.translation_tier] || '未知'}
            </span>
          )}
          {result.pipeline && (
            <span className="text-xs text-gray-500">
              帧提取 {result.pipeline.frame_extraction}s
              · 推理 {result.pipeline.mmrt_inference}s
              · 翻译 {result.pipeline.llm_translation}s
            </span>
          )}
        </div>

        {/* Confidence gauge */}
        <div className="flex items-center gap-6">
          <ConfidenceGauge value={result.confidence} />
          <div>
            <div className="text-sm text-gray-400">置信度</div>
            <div className="text-2xl font-bold" style={{
              color: result.confidence >= 0.8 ? '#34d399'
                : result.confidence >= 0.6 ? '#fbbf24' : '#f87171'
            }}>
              {Math.round(result.confidence * 100)}%
            </div>
          </div>
        </div>
      </div>

      <TTSPanel defaultText={result.predicted_text} />
    </motion.div>
  )
}
