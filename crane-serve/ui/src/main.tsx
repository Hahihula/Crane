import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ChatView } from './components/ChatView'
import { AsrView } from './components/AsrView'
import './styles.css'

export type UiConfig = { mode: 'chat' | 'asr'; multimodal: boolean; model_name: string; model_type: string }

function App() {
  const [config, setConfig] = useState<UiConfig | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => { fetch('/ui/config').then(r => r.ok ? r.json() : Promise.reject()).then(setConfig).catch(() => setError(true)) }, [])
  if (error) return <main className="grid min-h-screen place-content-center text-center text-zinc-500"><h1 className="text-xl font-semibold text-zinc-900">无法连接服务</h1><p className="mt-2">请确认 Crane 服务仍在运行。</p></main>
  if (!config) return <main className="grid min-h-screen place-content-center text-sm text-zinc-500">正在连接 Crane…</main>
  return <main className="min-h-screen bg-white text-zinc-900"><header className="flex h-14 items-center justify-between border-b border-zinc-200 px-6 md:px-[max(24px,calc((100vw-960px)/2))]"><div className="flex items-center gap-2 text-[17px] font-semibold tracking-tight"><span className="text-xl leading-none">⌁</span>Crane</div><span className="max-w-[52vw] truncate text-xs text-zinc-500">{config.model_name} · {config.model_type}</span></header>{config.mode === 'asr' ? <AsrView config={config} /> : <ChatView config={config} />}</main>
}
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
