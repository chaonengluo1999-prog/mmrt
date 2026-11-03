import { useState, useRef } from 'react'
import { synthesizeSpeech } from '../api/client'

const DIALECTS = [
  { id: 'mandarin',  label: '普通话', flag: '🇨🇳' },
  { id: 'cantonese', label: '粤语',   flag: '🌟' },
  { id: 'taiwan',    label: '台湾腔', flag: '🏝️' },
  { id: 'sichuan',   label: '四川话', flag: '🌶️' },
  { id: 'northeast', label: '东北话', flag: '❄️' },
  { id: 'shanghai',  label: '上海话', flag: '🏙️' },
]

export default function TTSPanel({ defaultText = '' }) {
  const [dialect, setDialect] = useState('mandarin')
  const [loading, setLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const audioRef = useRef(null)

  const speak = async () => {
    if (!defaultText) return
    setLoading(true)
    setAudioUrl(null)
    try {
      const url = await synthesizeSpeech(defaultText, dialect)
      setAudioUrl(url)
      setTimeout(() => audioRef.current?.play(), 100)
    } catch (e) {
      alert('TTS失败: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mt-6 p-5 bg-white/[0.03] rounded-xl border border-white/[0.06]">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-blue-400">🔊</span>
        <span className="font-semibold text-sm">多方言语音合成</span>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {DIALECTS.map(d => (
          <button key={d.id} onClick={() => setDialect(d.id)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-all
              ${dialect === d.id
                ? 'bg-purple-600 text-white'
                : 'bg-white/[0.06] text-gray-400 hover:bg-white/10'}`}>
            {d.flag} {d.label}
          </button>
        ))}
      </div>
      <button onClick={speak} disabled={loading || !defaultText}
        className="btn-primary px-4 py-2 rounded-lg text-sm font-semibold flex items-center gap-2">
        {loading
          ? <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />合成中...</>
          : <>▶ 播放语音</>}
      </button>
      {audioUrl && (
        <audio ref={audioRef} src={audioUrl} controls
          className="mt-3 w-full" style={{ height: 36 }} />
      )}
    </div>
  )
}
