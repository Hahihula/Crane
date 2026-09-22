import type { ReactNode } from 'react'

export function Attachment({ preview, name, onRemove, removeLabel }: { preview?: string; name?: string; onRemove?: () => void; removeLabel?: string }) {
  return <div className="relative inline-flex max-w-52 items-center gap-2 rounded-xl border border-zinc-200 bg-white p-1.5 pr-3 shadow-sm">
    {preview ? <img src={preview} alt={name || ''} className="h-12 w-12 rounded-lg object-cover" /> : <span className="grid h-10 w-10 place-items-center rounded-lg bg-zinc-100 text-zinc-500">＋</span>}
    {name && <span className="truncate text-xs text-zinc-600">{name}</span>}
    {onRemove && <button type="button" onClick={onRemove} aria-label={removeLabel} title={removeLabel} className="absolute -right-2 -top-2 grid h-5 w-5 place-items-center rounded-full border border-zinc-200 bg-white text-xs text-zinc-500 shadow-sm hover:text-zinc-950">×</button>}
  </div>
}

export function AttachmentList({ children }: { children: ReactNode }) {
  return <div className="mb-2 flex flex-wrap gap-2">{children}</div>
}
