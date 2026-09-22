import { Component, StrictMode, useEffect, useState, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { ChatView } from './components/ChatView'
import { AsrView } from './components/AsrView'
import { TtsView } from './components/TtsView'
import './styles.css'

export type UiConfig = { mode: 'chat' | 'asr' | 'tts'; multimodal: boolean; model_name: string; model_type: string }

class UiErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error) { console.error('Crane UI crashed', error) }
  render() {
    if (!this.state.error) return this.props.children
    return <main className="grid min-h-screen place-content-center px-6 text-center text-zinc-500"><h1 className="text-xl font-semibold text-zinc-900">Crane UI failed to start</h1><p className="mt-2 max-w-xl text-sm">{this.state.error.message}</p><button className="mx-auto mt-5 rounded-lg bg-zinc-900 px-4 py-2 text-sm text-white" onClick={() => { localStorage.removeItem('crane.chat.sessions.v1'); localStorage.removeItem('crane.chat.active-session.v1'); location.reload() }}>Reset local history</button></main>
  }
}

function App() {
  const [config, setConfig] = useState<UiConfig | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => { fetch('/ui/config').then(r => r.ok ? r.json() : Promise.reject()).then(setConfig).catch(() => setError(true)) }, [])
  if (error) return <main className="grid min-h-screen place-content-center text-center text-zinc-500"><h1 className="text-xl font-semibold text-zinc-900">无法连接服务</h1><p className="mt-2">请确认 Crane 服务仍在运行。</p></main>
  if (!config) return <main className="grid min-h-screen place-content-center text-sm text-zinc-500">正在连接 Crane…</main>
  return <main className="flex h-dvh flex-col overflow-hidden bg-white text-zinc-900">{config.mode === 'asr' ? <AsrView config={config} /> : config.mode === 'tts' ? <TtsView config={config} /> : <ChatView config={config} />}</main>
}
createRoot(document.getElementById('root')!).render(<StrictMode><UiErrorBoundary><App /></UiErrorBoundary></StrictMode>)
