import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'

export type ContentPart = { type: string; text?: string; image_url?: { url: string } }
export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string | ContentPart[]
  image?: string
  reasoning?: string
  status?: 'streaming' | 'complete' | 'stopped' | 'error'
  finishReason?: string
}

export type ChatSession = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

const STORAGE_KEY = 'crane.chat.sessions.v1'
const ACTIVE_KEY = 'crane.chat.active-session.v1'
// `crypto.randomUUID` is unavailable in some non-secure LAN/WebView contexts.
// History must never make the whole UI fail during its first render.
const newId = () => globalThis.crypto?.randomUUID?.() ?? `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
const storageGet = (key: string) => {
  try { return globalThis.localStorage?.getItem(key) ?? null } catch { return null }
}
const blankSession = (): ChatSession => ({ id: newId(), title: '', createdAt: Date.now(), updatedAt: Date.now(), messages: [] })
const textOf = (content: ChatMessage['content']) => typeof content === 'string' ? content : content.filter(part => part.type === 'text').map(part => part.text ?? '').join(' ')
const titleOf = (messages: ChatMessage[]) => {
  const first = messages.find(message => message.role === 'user')
  const title = first ? textOf(first.content).replace(/\s+/g, ' ').trim() : ''
  return title ? `${title.slice(0, 36)}${title.length > 36 ? '…' : ''}` : ''
}

function restoreSessions(): ChatSession[] {
  try {
    const parsed = JSON.parse(storageGet(STORAGE_KEY) || '[]')
    if (!Array.isArray(parsed)) return [blankSession()]
    const sessions = parsed.filter(session => session && typeof session.id === 'string' && Array.isArray(session.messages)) as ChatSession[]
    return sessions.length ? sessions.map(session => ({ ...session, messages: session.messages.map(message => message.status === 'streaming' ? { ...message, status: 'stopped' as const, finishReason: 'cancelled' } : message) })) : [blankSession()]
  } catch {
    return [blankSession()]
  }
}

export function useChatHistory() {
  const [sessions, setSessions] = useState<ChatSession[]>(restoreSessions)
  const [activeId, setActiveId] = useState(() => storageGet(ACTIVE_KEY) || '')
  const activeSession = useMemo(() => sessions.find(session => session.id === activeId) ?? sessions[0], [sessions, activeId])

  useEffect(() => {
    if (activeSession && activeSession.id !== activeId) setActiveId(activeSession.id)
  }, [activeId, activeSession])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
        if (activeSession) localStorage.setItem(ACTIVE_KEY, activeSession.id)
      } catch (error) {
        console.warn('Unable to persist Crane chat history', error)
      }
    }, 250)
    return () => window.clearTimeout(timer)
  }, [sessions, activeSession])

  const setMessages: Dispatch<SetStateAction<ChatMessage[]>> = action => {
    if (!activeSession) return
    setSessions(current => current.map(session => {
      if (session.id !== activeSession.id) return session
      const messages = typeof action === 'function' ? action(session.messages) : action
      return { ...session, messages, title: session.title || titleOf(messages), updatedAt: Date.now() }
    }))
  }

  const createSession = () => {
    const session = blankSession()
    setSessions(current => [session, ...current])
    setActiveId(session.id)
    return session.id
  }
  const selectSession = (id: string) => setActiveId(id)
  const deleteSession = (id: string) => {
    setSessions(current => {
      const remaining = current.filter(session => session.id !== id)
      if (remaining.length) {
        if (id === activeSession?.id) setActiveId(remaining[0].id)
        return remaining
      }
      const replacement = blankSession(); setActiveId(replacement.id); return [replacement]
    })
  }

  return {
    sessions: [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    activeSession,
    messages: activeSession?.messages ?? [],
    setMessages,
    createSession,
    selectSession,
    deleteSession,
  }
}
