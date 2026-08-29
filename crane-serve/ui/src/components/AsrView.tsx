import { DragEvent, useEffect, useRef, useState } from 'react'
import type { UiConfig } from '../main'

function wavBlob(parts: Float32Array[], sampleRate: number) {
  const size = parts.reduce((total, part) => total + part.length, 0), buffer = new ArrayBuffer(44 + size * 2), view = new DataView(buffer)
  const text = (at: number, value: string) => [...value].forEach((char, index) => view.setUint8(at + index, char.charCodeAt(0)))
  text(0, 'RIFF'); view.setUint32(4, 36 + size * 2, true); text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, size * 2, true)
  let offset = 44; for (const part of parts) for (const sample of part) { const value = Math.max(-1, Math.min(1, sample)); view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true); offset += 2 }
  return new Blob([buffer], { type: 'audio/wav' })
}

export function AsrView({ config }: { config: UiConfig }) {
  const [file, setFile] = useState<File | null>(null), [result, setResult] = useState('转写结果会显示在这里。'), [busy, setBusy] = useState(false), [recording, setRecording] = useState(false)
  const chunks = useRef<Float32Array[]>([]), sampleRate = useRef(16_000), interval = useRef<number | null>(null), transcribing = useRef(false), stream = useRef<MediaStream | null>(null), context = useRef<AudioContext | null>(null), processor = useRef<ScriptProcessorNode | null>(null), source = useRef<MediaStreamAudioSourceNode | null>(null)
  const sendAudio = async (blob: Blob, final = false) => {
    if (!blob.size || transcribing.current) return
    transcribing.current = true; setBusy(true)
    try { const form = new FormData(); form.append('file', blob, 'microphone.wav'); form.append('model', config.model_name); const response = await fetch('/v1/audio/transcriptions', { method: 'POST', body: form }); const data = await response.json(); if (!response.ok) throw new Error(data?.error?.message ?? '转写失败'); if (data.text || final) setResult(data.text || '（未识别到文字）') } catch (error) { if (final) setResult(`错误：${error instanceof Error ? error.message : '转写失败'}`) } finally { transcribing.current = false; setBusy(false) }
  }
  const transcribe = async () => { if (!file) return; setResult('正在处理音频…'); await sendAudio(file, true) }
  const startMic = async () => {
    if (!window.isSecureContext) { setResult('浏览器只允许在安全来源使用麦克风。请在本机通过 http://localhost:端口 打开；如果通过局域网 IP 或域名访问，请配置 HTTPS。'); return }
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) { setResult('当前浏览器没有提供麦克风录音 API。请检查浏览器权限或使用最新版 Chrome、Edge 或 Safari。'); return }
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ audio: true }); context.current = new AudioContext(); sampleRate.current = context.current.sampleRate; chunks.current = []
      source.current = context.current.createMediaStreamSource(stream.current); processor.current = context.current.createScriptProcessor(4096, 1, 1)
      processor.current.onaudioprocess = event => chunks.current.push(new Float32Array(event.inputBuffer.getChannelData(0)))
      source.current.connect(processor.current); processor.current.connect(context.current.destination); setRecording(true); setResult('正在聆听… 转写会每几秒自动刷新。')
      interval.current = window.setInterval(() => { if (chunks.current.length) void sendAudio(wavBlob(chunks.current, sampleRate.current)) }, 3500)
    } catch (error) { setResult(`无法打开麦克风：${error instanceof Error ? error.message : '请检查浏览器权限'}`) }
  }
  const stopMic = () => {
    if (interval.current) window.clearInterval(interval.current); interval.current = null; processor.current?.disconnect(); source.current?.disconnect(); stream.current?.getTracks().forEach(track => track.stop()); void context.current?.close(); processor.current = null; source.current = null; stream.current = null; context.current = null; setRecording(false)
    const finalBlob = wavBlob(chunks.current, sampleRate.current), submitFinal = () => transcribing.current ? window.setTimeout(submitFinal, 150) : void sendAudio(finalBlob, true); submitFinal()
  }
  useEffect(() => () => { if (processor.current) stopMic() }, [])
  const drop = (event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); setFile(event.dataTransfer.files[0] ?? null) }
  return <section className="flex min-h-[calc(100vh-56px)] justify-center px-5 pt-[13vh]"><div className="w-full max-w-[560px] text-center"><h1 className="text-3xl font-semibold tracking-tight">语音转文字</h1><p className="mt-2 text-sm text-zinc-500">上传音频，或直接使用麦克风让当前 ASR 模型实时听写。</p><div className="mt-7 flex justify-center gap-2"><button className={`rounded-lg px-4 py-2 text-sm font-medium text-white ${recording ? 'bg-red-600 hover:bg-red-700' : 'bg-zinc-900 hover:bg-zinc-700'}`} onClick={recording ? stopMic : startMic}>{recording ? '■ 停止听写' : '● 麦克风听写'}</button>{recording && <span className="self-center text-xs text-red-600">正在录音</span>}</div><label className="mt-7 flex cursor-pointer flex-col gap-2 rounded-lg border border-dashed border-zinc-300 px-4 py-8 transition hover:border-zinc-400 hover:bg-zinc-50" onDragOver={e => e.preventDefault()} onDrop={drop}><input hidden type="file" accept="audio/*" onChange={e => setFile(e.target.files?.[0] ?? null)}/><b className="text-sm">选择或拖放音频文件</b><span className="text-xs text-zinc-500">支持浏览器可读取的音频格式，最大 25 MB</span></label><div className="min-h-7 pt-2.5 text-xs text-zinc-500">{file?.name}</div><button className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40" disabled={!file || busy} onClick={transcribe}>{busy ? '正在转写…' : '开始转写'}</button><div className="mt-6 min-h-16 border-t border-zinc-200 pt-4 text-left text-sm whitespace-pre-wrap">{result}</div></div></section>
}
