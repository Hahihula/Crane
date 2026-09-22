import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { type ChatMessage, useChatHistory } from '../chat-history'
import type { UiConfig } from '../main'
import { useI18n } from '../i18n'
import { ConversationSidebar } from './ConversationSidebar'
import { Attachment, AttachmentList } from './ui/attachment'
import { Bubble, BubbleContent } from './ui/bubble'
import { Marker } from './ui/marker'
import { Message, MessageActions, MessageContent } from './ui/message'
import { MessageScroller } from './ui/message-scroller'

const id = () => globalThis.crypto?.randomUUID?.() ?? `message-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
const textOf = (content: ChatMessage['content']) => typeof content === 'string' ? content : content.filter(part => part.type === 'text').map(part => part.text ?? '').join('\n')

function Thinking({ text, active, labels }: { text: string; active: boolean; labels: { thinking: string; thought: string } }) {
  const [expanded, setExpanded] = useState(active)
  useEffect(() => { setExpanded(active) }, [active])
  if (!text && !active) return null
  // Keep the reasoning panel bounded without freezing the visible stream once it grows.
  // While generating, the latest tokens are what the user needs to see.
  const preview = text.length > 480
    ? active ? `…${text.slice(-480).trimStart()}` : `${text.slice(0, 480).trimEnd()}…`
    : text
  return <div className="mb-2 text-sm leading-6 text-zinc-400">
    <button type="button" className="flex items-center gap-2 text-left transition hover:text-zinc-700" onClick={() => !active && setExpanded(value => !value)}>
      {!active && <span className="text-zinc-400">{expanded ? '⌄' : '›'}</span>}
      <span className={active ? 'thinking-shimmer font-medium' : ''}>{active ? labels.thinking : labels.thought}</span>
    </button>
    {expanded && text && <div aria-live={active ? 'polite' : undefined} className="mt-1.5 max-w-2xl border-l border-zinc-200 pl-3 text-zinc-400">{preview}</div>}
  </div>
}

function SettingsPanel({ open, onClose, temperature, setTemperature, topP, setTopP, maxTokens, setMaxTokens, t, config }: {
  open: boolean; onClose: () => void; temperature: number; setTemperature: (value: number) => void; topP: number; setTopP: (value: number) => void; maxTokens: string; setMaxTokens: (value: string) => void; t: ReturnType<typeof useI18n>['t']; config: UiConfig
}) {
  if (!open) return null
  return <><button type="button" aria-label={t.closeSettings} onClick={onClose} className="fixed inset-0 z-40 cursor-default" /><section role="dialog" aria-label={t.settings} className="absolute right-3 top-11 z-50 w-[min(22rem,calc(100vw-1.5rem))] rounded-2xl border border-zinc-200 bg-white p-4 shadow-xl sm:right-6">
    <div className="mb-4 flex items-center justify-between"><h2 className="text-sm font-semibold text-zinc-900">{t.settings}</h2><button type="button" onClick={onClose} aria-label={t.closeSettings} className="grid h-7 w-7 place-items-center rounded-md text-zinc-400 hover:bg-zinc-100 hover:text-zinc-800">×</button></div>
    <div className="space-y-4 text-sm"><label className="block text-zinc-600"><span className="flex justify-between"><span>{t.temperature}</span><output className="tabular-nums text-zinc-400">{temperature.toFixed(1)}</output></span><input className="mt-2 w-full accent-zinc-900" type="range" min="0" max="2" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))} /></label><label className="block text-zinc-600"><span className="flex justify-between"><span>{t.topP}</span><output className="tabular-nums text-zinc-400">{topP.toFixed(2)}</output></span><input className="mt-2 w-full accent-zinc-900" type="range" min="0.05" max="1" step="0.05" value={topP} onChange={event => setTopP(Number(event.target.value))} /></label><label className="block text-zinc-600"><span>{t.maxTokens} <span className="text-zinc-400">({t.unlimited})</span></span><input inputMode="numeric" className="mt-2 h-9 w-full rounded-lg border border-zinc-200 px-2 text-zinc-900 outline-none focus:border-zinc-500" placeholder={t.unlimited} value={maxTokens} onChange={event => setMaxTokens(event.target.value.replace(/[^0-9]/g, ''))} /></label><p className="border-t border-zinc-100 pt-3 text-xs text-zinc-400"><span>{t.model}: </span>{config.model_name}</p></div>
  </section></>
}

function Markdown({ children }: { children: string }) {
  return <div className="chat-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown></div>
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: string }) {
  return <button type="button" onClick={onClick} aria-label={label} title={label} className="grid h-7 min-w-7 place-items-center rounded-md px-1.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900">{children}</button>
}

export function ChatView({ config }: { config: UiConfig }) {
  const { locale, t, toggle } = useI18n()
  const { sessions, activeSession, messages, setMessages, createSession, selectSession, deleteSession } = useChatHistory()
  const [input, setInput] = useState('')
  const [image, setImage] = useState<string | null>(null)
  const [imageName, setImageName] = useState('')
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.95)
  const [maxTokens, setMaxTokens] = useState('')
  const file = useRef<HTMLInputElement>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => {
    const node = textarea.current
    if (!node) return
    node.style.height = '0px'
    node.style.height = `${Math.min(node.scrollHeight, 180)}px`
  }, [input])

  const updateAssistant = (messageId: string, patch: Partial<ChatMessage>) => {
    setMessages(current => current.map(message => message.id === messageId ? { ...message, ...patch } : message))
  }

  const generate = async (conversation: ChatMessage[]) => {
    const assistantId = id()
    setMessages([...conversation, { id: assistantId, role: 'assistant', content: '', reasoning: '', status: 'streaming' }])
    setBusy(true)
    const abort = new AbortController()
    controller.current = abort
    let answer = '', reasoning = '', finishReason = ''
    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST', signal: abort.signal, headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: config.model_name, messages: conversation.map(({ role, content }) => ({ role, content })), stream: true, temperature, top_p: topP, ...(maxTokens ? { max_tokens: Number(maxTokens) } : {}) }),
      })
      if (!response.ok) { const data = await response.json(); throw new Error(data?.error?.message ?? t.requestFailed) }
      if (!response.headers.get('content-type')?.includes('text/event-stream')) {
        const data = await response.json()
        answer = data.choices?.[0]?.message?.content || ''
        finishReason = data.choices?.[0]?.finish_reason || 'stop'
        updateAssistant(assistantId, { content: answer || `（${t.empty}）`, status: 'complete', finishReason })
        return
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error(t.browserStream)
      const decoder = new TextDecoder()
      let pending = ''
      while (true) {
        const { done, value } = await reader.read()
        pending += decoder.decode(value ?? new Uint8Array(), { stream: !done })
        const lines = pending.split('\n'); pending = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (!payload || payload === '[DONE]') continue
          const chunk = JSON.parse(payload)
          if (chunk.error) throw new Error(chunk.error.message || t.requestFailed)
          const choice = chunk.choices?.[0]
          const delta = choice?.delta ?? {}
          answer += delta.content ?? ''
          reasoning += delta.reasoning_content ?? ''
          finishReason = choice?.finish_reason ?? finishReason
          updateAssistant(assistantId, { content: answer, reasoning, status: 'streaming', finishReason })
        }
        if (done) break
      }
      updateAssistant(assistantId, { content: answer || `（${t.empty}）`, reasoning, status: 'complete', finishReason: finishReason || 'stop' })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        updateAssistant(assistantId, { content: answer, reasoning, status: 'stopped', finishReason: 'cancelled' })
      } else {
        updateAssistant(assistantId, { content: answer || `${t.failed}：${error instanceof Error ? error.message : t.requestFailed}`, reasoning, status: 'error' })
      }
    } finally {
      if (controller.current === abort) controller.current = null
      setBusy(false)
    }
  }

  const submit = async (event?: FormEvent) => {
    event?.preventDefault()
    const text = input.trim()
    if (busy || (!text && !image)) return
    const content: ChatMessage['content'] = image ? [...(text ? [{ type: 'text', text }] : []), { type: 'image_url', image_url: { url: image } }] : text
    const user: ChatMessage = { id: id(), role: 'user', content, image: image ?? undefined }
    const next = [...messages.filter(message => message.status !== 'error' || textOf(message.content)), user]
    setInput(''); setImage(null); setImageName('')
    await generate(next)
  }

  const stop = () => controller.current?.abort()
  const newConversation = () => {
    if (busy) stop()
    if (messages.length === 0) return
    createSession(); setInput(''); setImage(null); setImageName('')
  }
  const retry = async (assistantId: string) => {
    if (busy) return
    const index = messages.findIndex(message => message.id === assistantId)
    if (index < 0) return
    await generate(messages.slice(0, index).filter(message => message.role === 'user' || message.status === 'complete'))
  }
  const copy = async (message: ChatMessage) => {
    await navigator.clipboard.writeText(textOf(message.content))
    setCopied(message.id); window.setTimeout(() => setCopied(null), 1400)
  }
  const chooseImage = (selected?: File) => {
    if (!selected?.type.startsWith('image/')) return
    const reader = new FileReader(); reader.onload = () => { setImage(String(reader.result)); setImageName(selected.name) }; reader.readAsDataURL(selected)
  }
  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit() }
  }

  const last = messages.at(-1)
  return <div className="flex min-h-0 flex-1 bg-white" lang={locale}>
    <ConversationSidebar sessions={sessions} activeId={activeSession?.id} open={historyOpen} desktopOpen={sidebarOpen} t={t} onClose={() => setHistoryOpen(false)} onNew={() => { newConversation(); setHistoryOpen(false) }} onSelect={id => { if (busy) stop(); selectSession(id) }} onDelete={id => { if (busy && id === activeSession?.id) stop(); deleteSession(id) }} />
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="relative flex h-12 shrink-0 items-center justify-between px-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-1"><button type="button" onClick={() => { setHistoryOpen(true); setSidebarOpen(true) }} aria-label={t.history} className="grid h-8 w-8 place-items-center rounded-lg text-lg text-zinc-600 hover:bg-zinc-100 md:hidden">☰</button><button type="button" onClick={() => setSidebarOpen(value => !value)} aria-label={sidebarOpen ? t.collapseSidebar : t.expandSidebar} title={sidebarOpen ? t.collapseSidebar : t.expandSidebar} className="hidden h-8 w-8 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100 md:grid">☰</button><span className="truncate text-sm text-zinc-400">{activeSession?.title || t.untitled}</span></div>
        <div className="flex items-center gap-1"><button type="button" onClick={() => setSettingsOpen(value => !value)} aria-label={t.settings} title={t.settings} className="grid h-8 w-8 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-950">⚙</button><button type="button" onClick={toggle} aria-label={t.language} className="rounded-lg px-2 py-1.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-950">{t.language}</button></div>
        <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} temperature={temperature} setTemperature={setTemperature} topP={topP} setTopP={setTopP} maxTokens={maxTokens} setMaxTokens={setMaxTokens} t={t} config={config} />
      </div>

    <MessageScroller dependency={`${messages.length}:${textOf(last?.content ?? '').length}:${last?.reasoning?.length ?? 0}`} scrollLabel={t.latest}>
      <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-3 py-8 sm:px-6">
        {messages.length === 0 ? <div className="grid flex-1 place-items-center py-20 text-center">
          <div><div className="mx-auto mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-zinc-950 text-lg font-semibold text-white shadow-lg">C</div><h1 className="text-2xl font-semibold tracking-tight text-zinc-950">{t.emptyTitle}</h1><p className="mt-2 text-sm text-zinc-500">{t.emptyHint}</p></div>
        </div> : <div className="space-y-7">
          {messages.map(message => <Message key={message.id} from={message.role} className="gap-0">
            <MessageContent className={message.role === 'user' ? 'order-first' : ''}>
              {message.image && <AttachmentList><Attachment preview={message.image} name={t.image} /></AttachmentList>}
              <Bubble className={message.role === 'user' ? 'rounded-[1.6rem] !px-3 !py-2 bg-zinc-900 text-white shadow-sm' : 'px-0 py-0 text-zinc-800'}>
                <BubbleContent>
                  {message.role === 'assistant' && <Thinking text={message.reasoning ?? ''} active={message.status === 'streaming' && !message.content} labels={t} />}
                  {textOf(message.content) ? <Markdown>{textOf(message.content)}</Markdown> : null}
                </BubbleContent>
              </Bubble>
              {message.role === 'assistant' && message.status !== 'streaming' && <>
                {message.status === 'stopped' && <Marker>{t.interrupted}</Marker>}
                {message.finishReason === 'length' && <Marker tone="danger">{t.lengthStop}</Marker>}
                <MessageActions><IconButton label={copied === message.id ? t.copied : t.copy} onClick={() => void copy(message)}>{copied === message.id ? '✓' : '⧉'}</IconButton><IconButton label={t.retry} onClick={() => void retry(message.id)}>↻</IconButton></MessageActions>
              </>}
            </MessageContent>
          </Message>)}
        </div>}
      </div>
    </MessageScroller>

      <div className="shrink-0 bg-gradient-to-t from-white via-white to-white/0 px-3 pb-3 pt-5 sm:px-6 sm:pb-5">
      <form onSubmit={submit} className="mx-auto max-w-3xl">
        {image && <AttachmentList><Attachment preview={image} name={imageName} onRemove={() => { setImage(null); setImageName('') }} removeLabel={t.removeAttachment} /></AttachmentList>}
        <div className="flex items-center gap-1 rounded-[1.75rem] border border-zinc-200 bg-white p-1.5 shadow-[0_8px_30px_rgb(0_0_0_/_0.08)] transition focus-within:border-zinc-400 focus-within:shadow-[0_10px_35px_rgb(0_0_0_/_0.11)]">
          {config.multimodal && <><button className="grid h-8 w-8 shrink-0 place-items-center rounded-xl text-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900" type="button" aria-label={t.attach} title={t.attach} onClick={() => file.current?.click()}>＋</button><input hidden ref={file} type="file" accept="image/*" onChange={event => chooseImage(event.target.files?.[0])} /></>}
          <textarea ref={textarea} rows={1} className="max-h-36 min-h-8 flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-[15px] leading-6 text-zinc-900 outline-none placeholder:text-zinc-400" value={input} onChange={event => setInput(event.target.value)} onKeyDown={onComposerKeyDown} placeholder={t.placeholder} disabled={busy} />
          {busy ? <button type="button" onClick={stop} aria-label={t.stop} title={t.stop} className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-zinc-950 text-white hover:bg-zinc-800"><span className="h-2.5 w-2.5 rounded-sm bg-white" /></button> : <button type="submit" disabled={!input.trim() && !image} aria-label={t.send} title={t.send} className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-zinc-950 text-base text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-400">↑</button>}
        </div>
        <div className="mt-2 flex items-center justify-between px-1 text-[11px] text-zinc-400"><span className="hidden sm:inline">{t.enterHint}</span><span className="mx-auto sm:mx-0">{t.disclaimer}</span></div>
      </form>
      </div>
    </div>
  </div>
}
