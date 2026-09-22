import type { HTMLAttributes, ReactNode } from 'react'

type MessageProps = HTMLAttributes<HTMLDivElement> & { from: 'user' | 'assistant' }

export function Message({ from, className = '', ...props }: MessageProps) {
  return <article data-role={from} className={`group flex w-full gap-3 ${from === 'user' ? 'justify-end' : 'justify-start'} ${className}`} {...props} />
}

export function MessageAvatar({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full border border-zinc-200 bg-white text-[11px] font-semibold text-zinc-600 shadow-sm ${className}`}>{children}</div>
}

export function MessageContent({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`min-w-0 max-w-[min(46rem,calc(100%-2.75rem))] ${className}`}>{children}</div>
}

export function MessageActions({ children }: { children: ReactNode }) {
  return <div className="mt-1.5 flex min-h-7 items-center gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">{children}</div>
}
