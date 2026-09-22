import type { ChatSession } from '../chat-history'
import type { Translations } from '../i18n'

export function ConversationSidebar({ sessions, activeId, open, desktopOpen, t, onClose, onNew, onSelect, onDelete }: {
  sessions: ChatSession[]
  activeId?: string
  open: boolean
  desktopOpen: boolean
  t: Translations
  onClose: () => void
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  return <>
    {open && <button type="button" aria-label={t.closeHistory} onClick={onClose} className="fixed inset-0 z-30 bg-black/20 backdrop-blur-[1px] md:hidden" />}
    <aside className={`${open ? 'translate-x-0' : '-translate-x-full'} ${desktopOpen ? 'md:flex' : 'md:hidden'} fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-zinc-200 bg-zinc-50/95 p-3 shadow-xl transition-transform md:static md:z-auto md:w-64 md:translate-x-0 md:shadow-none`}>
      <div className="mb-3 flex items-center justify-between px-1 py-1 md:hidden"><span className="font-semibold">Crane</span><button type="button" onClick={onClose} className="grid h-8 w-8 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-200">×</button></div>
      <button type="button" onClick={onNew} className="flex h-10 items-center justify-center gap-2 rounded-xl border border-zinc-200 bg-white text-sm font-medium text-zinc-800 shadow-sm hover:bg-zinc-100">＋ {t.newChat}</button>
      <div className="mb-2 mt-5 px-2 text-[11px] font-medium uppercase tracking-wider text-zinc-400">{t.history}</div>
      <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {sessions.map(session => <div key={session.id} className={`group/session flex items-center rounded-xl ${session.id === activeId ? 'bg-white shadow-sm ring-1 ring-zinc-200' : 'hover:bg-zinc-200/60'}`}>
          <button type="button" onClick={() => { onSelect(session.id); onClose() }} className="min-w-0 flex-1 px-3 py-2.5 text-left">
            <div className="truncate text-sm text-zinc-800">{session.title || t.untitled}</div>
            <div className="mt-0.5 text-[11px] text-zinc-400">{new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(session.updatedAt)}</div>
          </button>
          <button type="button" onClick={() => onDelete(session.id)} aria-label={t.deleteChat} title={t.deleteChat} className="mr-1 grid h-8 w-8 shrink-0 place-items-center rounded-lg text-zinc-400 opacity-0 hover:bg-red-50 hover:text-red-600 group-hover/session:opacity-100 focus:opacity-100">×</button>
        </div>)}
      </nav>
      <div className="mt-3 border-t border-zinc-200 px-2 pt-3 text-[11px] leading-5 text-zinc-400">{t.localHistory}</div>
    </aside>
  </>
}
