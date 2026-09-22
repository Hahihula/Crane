import { useEffect, useRef, useState, type ReactNode } from 'react'

export function MessageScroller({ children, dependency, scrollLabel }: { children: ReactNode; dependency: unknown; scrollLabel: string }) {
  const viewport = useRef<HTMLDivElement>(null)
  const [following, setFollowing] = useState(true)

  const scrollToLatest = (behavior: ScrollBehavior = 'smooth') => {
    const node = viewport.current
    if (!node) return
    node.scrollTo({ top: node.scrollHeight, behavior })
    setFollowing(true)
  }

  useEffect(() => {
    if (following) requestAnimationFrame(() => scrollToLatest('auto'))
  }, [dependency, following])

  return <div className="relative min-h-0 flex-1">
    <div ref={viewport} onScroll={event => {
      const node = event.currentTarget
      setFollowing(node.scrollHeight - node.scrollTop - node.clientHeight < 96)
    }} className="h-full overflow-y-auto overscroll-contain scroll-smooth">
      {children}
    </div>
    {!following && <button type="button" onClick={() => scrollToLatest()} aria-label={scrollLabel} title={scrollLabel} className="absolute bottom-4 left-1/2 grid h-9 w-9 -translate-x-1/2 place-items-center rounded-full border border-zinc-200 bg-white text-lg text-zinc-700 shadow-lg transition hover:-translate-y-0.5 hover:bg-zinc-50">↓</button>}
  </div>
}
