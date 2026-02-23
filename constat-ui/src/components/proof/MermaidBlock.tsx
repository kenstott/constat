import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'

let mermaidId = 0

export function MermaidBlock({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [svg, setSvg] = useState<string>('')

  useEffect(() => {
    const isDark = document.documentElement.classList.contains('dark')
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? 'dark' : 'default',
      flowchart: { curve: 'linear' },
      themeVariables: isDark
        ? { darkMode: true }
        : { primaryColor: '#fff', lineColor: '#6B7280' },
    })

    const id = `mermaid-${++mermaidId}`
    mermaid.render(id, chart.trim()).then(({ svg: rendered }) => {
      setSvg(rendered)
    }).catch((err) => {
      console.error('Mermaid render error:', err)
      setSvg(`<pre>${chart}</pre>`)
    })
  }, [chart])

  return (
    <>
      <style>{`
        .mermaid-container svg rect.background { stroke: none !important; stroke-width: 0 !important; fill: transparent !important; }
        .mermaid-container svg > rect { stroke: none !important; stroke-width: 0 !important; fill: transparent !important; }
        .mermaid-container svg > g > rect { stroke: none !important; stroke-width: 0 !important; fill: transparent !important; }
        .mermaid-container svg { max-width: 100%; height: auto; }
      `}</style>
      <div
        ref={containerRef}
        className="mermaid-container my-4 flex justify-center overflow-x-auto bg-white dark:bg-gray-900"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </>
  )
}
