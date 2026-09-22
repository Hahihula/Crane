import type { ReactNode } from 'react'

export function Marker({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'danger' | 'active' }) {
  const color = tone === 'danger' ? 'text-red-600' : tone === 'active' ? 'text-blue-600' : 'text-zinc-400'
  return <div className={`flex items-center gap-2 py-2 text-xs ${color}`}><span className="h-px flex-1 bg-current opacity-20" /><span>{children}</span><span className="h-px flex-1 bg-current opacity-20" /></div>
}
