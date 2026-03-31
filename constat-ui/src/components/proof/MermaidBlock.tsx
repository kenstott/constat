// Copyright (c) 2025 Kenneth Stott
// Canary: 426acf91-55b2-4ee5-bf27-7503ffd73344
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

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
    <div
      ref={containerRef}
      className="mermaid-container my-4 flex justify-center overflow-x-auto bg-white dark:bg-gray-900"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}