import { useState, useEffect } from 'react'
import { fetchStatus, fetchStats } from './api/client'
import Header from './components/Header'
import VideoUpload from './components/VideoUpload'
import InferenceResult from './components/InferenceResult'
import ArchDiagram from './components/ArchDiagram'
import StatsPanel from './components/StatsPanel'
import TTSPanel from './components/TTSPanel'
import EvalReport from './components/EvalReport'

const TABS = [
  { id: 'recognize', label: '识别', icon: '🤟' },
  { id: 'architecture', label: '架构', icon: '🧠' },
  { id: 'evaluation', label: '评估', icon: '📊' },
]

export default function App() {
  const [tab, setTab] = useState('recognize')
  const [status, setStatus] = useState(null)
  const [stats, setStats] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {})
    fetchStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div className="min-h-screen px-4 py-8 md:px-8 lg:px-12">
      <Header status={status} />

      {/* Tabs */}
      <div className="max-w-7xl mx-auto flex gap-2 mb-8 justify-center">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-5 py-2.5 rounded-xl text-sm font-semibold border transition-all
              ${tab === t.id ? 'tab-active' : 'border-white/10 text-gray-400 hover:text-white hover:border-white/20'}`}>
            <span className="mr-1.5">{t.icon}</span>{t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto">
        {tab === 'recognize' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <VideoUpload onResult={setResult} modelLoaded={status?.model_loaded} />
              {result && <InferenceResult result={result} />}
            </div>
            <div className="space-y-5">
              <StatsPanel stats={stats} status={status} />
              <TTSPanel defaultText="你好，欢迎使用手语识别系统" />
            </div>
          </div>
        )}
        {tab === 'architecture' && <ArchDiagram />}
        {tab === 'evaluation' && <EvalReport />}
      </div>

      <footer className="text-center mt-16 pb-8 text-gray-600 text-xs">
        MMRT Multi-Modal Recognition Transformer &middot; CSL-Daily &middot; 2026
      </footer>
    </div>
  )
}
