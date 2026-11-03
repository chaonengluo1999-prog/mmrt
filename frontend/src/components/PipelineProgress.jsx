import { motion } from 'framer-motion'

const STEPS = [
  { label: '视频帧提取', icon: '🎞️' },
  { label: '视觉特征提取', icon: '👁️' },
  { label: '时序建模', icon: '⏱️' },
  { label: 'CTC 解码', icon: '🔤' },
  { label: 'LLM 翻译', icon: '🤖' },
]

export default function PipelineProgress({ step }) {
  return (
    <div className="mb-5 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center justify-between gap-1">
        {STEPS.map((s, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-1.5">
            <motion.div
              animate={{
                scale: step === i ? [1, 1.2, 1] : 1,
                opacity: step >= i ? 1 : 0.3,
              }}
              transition={{ duration: 0.5, repeat: step === i ? Infinity : 0 }}
              className={`w-10 h-10 rounded-full flex items-center justify-center text-lg
                ${step > i ? 'bg-emerald-500/20 border border-emerald-500/40'
                  : step === i ? 'bg-indigo-500/20 border border-indigo-500/50'
                  : 'bg-white/5 border border-white/10'}`}>
              {step > i ? '✓' : s.icon}
            </motion.div>
            <span className={`text-[10px] font-medium text-center leading-tight
              ${step >= i ? 'text-gray-300' : 'text-gray-600'}`}>
              {s.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
