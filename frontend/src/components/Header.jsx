import { motion } from 'framer-motion'

export default function Header({ status }) {
  const loaded = status?.model_loaded
  return (
    <header className="text-center mb-10">
      <motion.h1
        initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
        className="text-5xl md:text-7xl font-black mb-3 tracking-tight">
        <span className="grad-text">MMRT</span>
      </motion.h1>
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="text-lg md:text-xl text-gray-400 mb-1">
        多模态手语识别与翻译系统
      </motion.p>
      <p className="text-xs text-gray-600 mb-6">Multi-Modal Recognition Transformer</p>

      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="inline-flex flex-wrap items-center gap-3 glass px-6 py-3">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${loaded ? 'bg-emerald-400' : 'bg-amber-400'} pulse-ring`} />
          <span className="text-sm">{loaded ? '模型就绪' : '模型未加载'}</span>
        </div>
        {status && <span className="tag tag-blue">{String(status.device).toUpperCase()}</span>}
        <span className="tag tag-purple">CSL-Daily</span>
        <span className="tag tag-cyan">DeepSeek LLM</span>
      </motion.div>
    </header>
  )
}
