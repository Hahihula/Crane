import { FormEvent, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { UiConfig } from '../main'

type Message = { role: 'user' | 'assistant'; content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>; image?: string; reasoning?: string }

function splitThink(text: string) {
  const start = text.indexOf('<think>')
  if (start < 0) return { content: text, reasoning: '', active: false }
  const end = text.indexOf('</think>', start)
  return { content: text.slice(0, start) + (end < 0 ? '' : text.slice(end + 8)), reasoning: text.slice(start + 7, end < 0 ? undefined : end), active: end < 0 }
}

function Thinking({ text, open }: { text: string; open: boolean }) {
  const [expanded, setExpanded] = useState(open)
  const visible = open || expanded
  if (!text && !open) return null
  return <div className="mb-3 text-sm text-zinc-500"><button type="button" className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-800" onClick={() => setExpanded(value => !value)}><span className="text-xs">{visible ? '⌄' : '›'}</span>{open ? '正在思考…' : '已完成思考'}</button>{visible && <div className="mt-2 border-l border-zinc-200 pl-3 leading-6 text-zinc-500">{text || '…'}</div>}</div>
}

function Markdown({ children }: { children: string }) {
  return <div className="break-words [&_a]:text-blue-600 [&_a]:underline [&_blockquote]:my-3 [&_blockquote]:border-l-2 [&_blockquote]:border-zinc-300 [&_blockquote]:pl-3 [&_blockquote]:text-zinc-600 [&_code]:rounded [&_code]:bg-zinc-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13px] [&_h1]:mb-3 [&_h1]:text-2xl [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-5 [&_h2]:text-xl [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:font-semibold [&_li]:ml-5 [&_li]:list-disc [&_ol]:my-3 [&_ol]:list-decimal [&_p]:mb-3 [&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-zinc-900 [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-zinc-100 [&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-zinc-200 [&_td]:px-3 [&_td]:py-1.5 [&_th]:border [&_th]:border-zinc-200 [&_th]:bg-zinc-50 [&_th]:px-3 [&_th]:py-1.5 [&_ul]:my-3"><ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown></div>
}

export function ChatView({ config }: { config: UiConfig }) {
  const [messages, setMessages] = useState<Message[]>([]), [input, setInput] = useState(''), [image, setImage] = useState<string | null>(null), [busy, setBusy] = useState(false)
  const file = useRef<HTMLInputElement>(null)
  const submit = async (event: FormEvent) => {
    event.preventDefault(); const text = input.trim(); if (busy || (!text && !image)) return
    const content = image ? [...(text ? [{ type: 'text', text }] : []), { type: 'image_url', image_url: { url: image } }] : text
    const user: Message = { role: 'user', content, image: image ?? undefined }; const next = [...messages, user]
    setMessages([...next, { role: 'assistant', content: '' }]); setInput(''); setImage(null); setBusy(true)
    try {
      const response = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: config.model_name, messages: next.map(({ role, content }) => ({ role, content })), max_tokens: 1024, stream: true }) })
      if (!response.ok) { const data = await response.json(); throw new Error(data?.error?.message ?? '请求失败') }
      // Some VLM routes currently return a normal JSON response. Accept it as
      // a compatibility fallback; all streaming-capable models use SSE below.
      if (!response.headers.get('content-type')?.includes('text/event-stream')) {
        const data = await response.json(); setMessages([...next, { role: 'assistant', content: data.choices?.[0]?.message?.content || '（模型未返回文本）' }]); return
      }
      const reader = response.body?.getReader(); if (!reader) throw new Error('浏览器不支持流式响应')
      const decoder = new TextDecoder(); let pending = '', answer = '', reasoning = ''
      const update = () => setMessages([...next, { role: 'assistant', content: answer, reasoning }])
      while (true) {
        const { done, value } = await reader.read(); pending += decoder.decode(value ?? new Uint8Array(), { stream: !done })
        const lines = pending.split('\n'); pending = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim(); if (!payload || payload === '[DONE]') continue
          const chunk = JSON.parse(payload); const delta = chunk.choices?.[0]?.delta ?? {}; answer += delta.content ?? ''; reasoning += delta.reasoning_content ?? ''; update()
        }
        if (done) break
      }
      if (!answer) setMessages([...next, { role: 'assistant', content: '（模型未返回文本）', reasoning }])
    } catch (error) { setMessages([...next, { role: 'assistant', content: `错误：${error instanceof Error ? error.message : '请求失败'}` }]) } finally { setBusy(false) }
  }
  const chooseImage = (selected?: File) => { if (!selected?.type.startsWith('image/')) return; const reader = new FileReader(); reader.onload = () => setImage(String(reader.result)); reader.readAsDataURL(selected) }
  return <section className="flex h-[calc(100vh-56px)] min-h-[560px] flex-col bg-white"><div className="flex-1 overflow-y-auto px-5 pb-44 pt-10 md:px-[max(24px,calc((100vw-760px)/2))]">{messages.length === 0 && <div className="mx-auto mt-[18vh] max-w-xl text-center"><h1 className="text-3xl font-semibold tracking-tight text-zinc-900">有什么可以帮你的？</h1><p className="mt-2 text-sm text-zinc-500">{config.multimodal ? '支持文字对话，也可以在需要时添加图片。' : '向模型发送一条消息，开始对话。'}</p></div>}{messages.map((message, index) => { const raw = typeof message.content === 'string' ? message.content : message.content.filter(p => p.type === 'text').map(p => p.text).join(''); const thought = splitThink(raw); const thinking = message.reasoning || thought.reasoning; const thinkingOpen = busy && index === messages.length - 1 && (Boolean(thinking) || thought.active); return <article className="mx-auto mb-7 flex max-w-[760px] gap-3.5" key={index}><span className={`grid h-6 w-6 shrink-0 place-items-center rounded-md text-xs font-semibold ${message.role === 'assistant' ? 'bg-zinc-900 text-white' : 'bg-zinc-100 text-zinc-600'}`}>{message.role === 'assistant' ? '⌁' : '你'}</span><div className="min-w-0 flex-1 break-words pt-0.5 text-[15px] leading-7 text-zinc-800">{message.image && <img className="mb-2.5 block max-h-[270px] w-full max-w-[340px] rounded-md border border-zinc-200 object-contain" src={message.image} alt="已上传的图片" />}{message.role === 'assistant' && <Thinking text={thinking} open={thinkingOpen} />}{raw || thinking ? <Markdown>{message.role === 'assistant' ? thought.content : raw}</Markdown> : <span className="text-sm text-zinc-400">正在思考…</span>}</div></article>})}</div><form className="fixed bottom-0 left-1/2 z-10 w-[calc(100%-24px)] max-w-[760px] -translate-x-1/2 bg-gradient-to-b from-transparent from-0% via-white via-30% to-white pt-3 pb-3 md:w-[calc(100%-32px)] md:pb-5" onSubmit={submit}>{image && <div className="flex items-center gap-2 py-1.5 text-xs text-zinc-500"><img className="h-6 w-6 rounded object-cover" src={image} alt="图片预览"/><span>已添加图片</span><button className="text-lg leading-none text-zinc-500 hover:text-zinc-900" type="button" onClick={() => setImage(null)}>×</button></div>}<div className="flex items-end gap-2 rounded-xl border border-zinc-300 bg-white px-3 py-2 shadow-[0_2px_10px_rgb(0_0_0_/_0.05)] focus-within:border-zinc-400">{config.multimodal && <><button className="grid h-7 w-7 place-items-center rounded-md text-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900" type="button" aria-label="添加图片" onClick={() => file.current?.click()}>⌑</button><input hidden ref={file} type="file" accept="image/*" onChange={e => chooseImage(e.target.files?.[0])}/></>}<textarea className="min-h-6 flex-1 resize-none border-0 bg-transparent py-0.5 text-sm leading-6 text-zinc-900 outline-none placeholder:text-zinc-400" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit() } }} placeholder={`给 ${config.model_name} 发送消息`} rows={1}/><button className="grid h-7 w-7 place-items-center rounded-md bg-zinc-900 text-base text-white disabled:bg-zinc-300" disabled={busy || (!input.trim() && !image)} aria-label="发送" type="submit">↑</button></div><p className="mt-1.5 text-center text-[11px] text-zinc-400">Crane 本地推理 · Enter 发送，Shift + Enter 换行</p></form></section>
}
