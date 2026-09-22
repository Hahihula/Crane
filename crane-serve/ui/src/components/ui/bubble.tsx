import type { HTMLAttributes, ReactNode } from 'react'

export function Bubble({ className = '', ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-2xl px-4 py-3 text-[15px] leading-7 ${className}`} {...props} />
}

export function BubbleContent({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`min-w-0 ${className}`}>{children}</div>
}
