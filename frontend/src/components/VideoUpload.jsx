import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { predictVideo } from '../api/client'
import PipelineProgress from './PipelineProgress'

export default function VideoUpload({ onResult, modelLoaded }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(-1)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    onResult(null)
    setPreview(URL.createObjectURL(f))
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('video/')) handleFile(f)
  }, [])

  const predict = async () => {
    if (!file) return
    setLoading(true)
    onResult(null)
    try {
      setPipelineStep(0)
      await new Promise(r => setTimeout(r, 300))
      setPipelineStep(1)
      const result = await predictVideo(file)
      setPipelineStep(2)
      await new Promise(r => setTimeout(r, 200))
      setPipelineStep(3)
      await new Promise(r => setTimeout(r, 200))
      setPipelineStep(4)
      onResult(result)
    } catch (e) {
      alert('识别失败: ' + e.message)
    } finally {
      setLoading(false)
      setPipelineStep(-1)
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      className="glass p-8">
      <h2 className="text-xl font-bold mb-5 flex items-center gap-2">
        <span className="text-2xl">🎬</span> 视频上传与识别
      </h2>

      <div
        className={`upload-zone p-10 text-center rounded-2xl mb-5 ${dragOver ? 'drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}>
        {preview ? (
          <video src={preview} className="mx-auto max-h-52 rounded-xl mb-3" muted playsInline controls />
        ) : (
          <div className="text-6xl mb-4 opacity-40">🤟</div>
        )}
        <p className="text-base mb-1">{file ? file.name : '点击或拖拽上传手语视频'}</p>
        <p className="text-xs text-gray-500">支持 MP4 / AVI / MOV</p>
        <input ref={inputRef} type="file" accept="video/*"
          onChange={e => handleFile(e.target.files[0])} className="hidden" />
      </div>

      {loading && <PipelineProgress step={pipelineStep} />}

      <button onClick={predict} disabled={!file || loading || !modelLoaded}
        className="btn-primary w-full py-4 rounded-xl text-base font-bold flex items-center justify-center gap-3">
        {loading
          ? <><div className="spinner" /> 识别中，请稍候...</>
          : <><span className="text-lg">✨</span> 开始识别</>}
      </button>
    </motion.div>
  )
}
