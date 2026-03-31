// Copyright (c) 2025 Kenneth Stott
// Canary: ad2018ad-640a-40c9-adb1-f5a8a8e36d48
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// AnnotationWidget - Image markup with canvas overlay for rectangle/circle/arrow/text annotations

import { useState, useRef, useCallback, useEffect } from 'react'
import { TrashIcon } from '@heroicons/react/24/outline'

type AnnotationTool = 'rectangle' | 'circle' | 'arrow' | 'text'

interface Annotation {
  tool: AnnotationTool
  coords: { x1: number; y1: number; x2: number; y2: number }
  text?: string
}

interface AnnotationWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: unknown
  onAnswer: (freeform: string, structured: { annotations: Annotation[] }) => void
}

const TOOL_LABELS: Record<AnnotationTool, string> = {
  rectangle: 'Rect',
  circle: 'Circle',
  arrow: 'Arrow',
  text: 'Text',
}

export function AnnotationWidget({ config, structuredValue, onAnswer }: AnnotationWidgetProps) {
  const imageUrl = config.imageUrl as string | undefined
  const width = (config.width as number) || 600
  const height = (config.height as number) || 400

  const [annotations, setAnnotations] = useState<Annotation[]>(() => {
    if (structuredValue && typeof structuredValue === 'object' && 'annotations' in (structuredValue as Record<string, unknown>)) {
      return (structuredValue as { annotations: Annotation[] }).annotations
    }
    return []
  })
  const [activeTool, setActiveTool] = useState<AnnotationTool>('rectangle')
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [currentCoords, setCurrentCoords] = useState<{ x: number; y: number } | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const emitAnswer = useCallback((newAnnotations: Annotation[]) => {
    const freeform = newAnnotations.length > 0
      ? `${newAnnotations.length} annotation(s): ${newAnnotations.map(a => a.tool + (a.text ? `: "${a.text}"` : '')).join(', ')}`
      : 'No annotations'
    onAnswer(freeform, { annotations: newAnnotations })
  }, [onAnswer])

  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: Math.round((e.clientX - rect.left) * (canvas.width / rect.width)),
      y: Math.round((e.clientY - rect.top) * (canvas.height / rect.height)),
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getCanvasPoint(e)
    setIsDrawing(true)
    setDrawStart(point)
    setCurrentCoords(point)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    setCurrentCoords(getCanvasPoint(e))
  }

  const handleMouseUp = () => {
    if (!isDrawing || !drawStart || !currentCoords) return
    setIsDrawing(false)

    const coords = {
      x1: drawStart.x,
      y1: drawStart.y,
      x2: currentCoords.x,
      y2: currentCoords.y,
    }

    // Minimum size check
    const dx = Math.abs(coords.x2 - coords.x1)
    const dy = Math.abs(coords.y2 - coords.y1)
    if (dx < 5 && dy < 5) return

    if (activeTool === 'text') {
      const text = prompt('Enter annotation text:')
      if (!text) return
      const newAnnotation: Annotation = { tool: activeTool, coords, text }
      const newAnnotations = [...annotations, newAnnotation]
      setAnnotations(newAnnotations)
      emitAnswer(newAnnotations)
    } else {
      const newAnnotation: Annotation = { tool: activeTool, coords }
      const newAnnotations = [...annotations, newAnnotation]
      setAnnotations(newAnnotations)
      emitAnswer(newAnnotations)
    }

    setDrawStart(null)
    setCurrentCoords(null)
  }

  // Draw annotations on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw background image if provided
    if (imageUrl) {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        drawAnnotations(ctx)
      }
      img.src = imageUrl
    } else {
      // Light grid background
      ctx.fillStyle = '#f9fafb'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.strokeStyle = '#e5e7eb'
      ctx.lineWidth = 1
      for (let x = 0; x < canvas.width; x += 20) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }
      for (let y = 0; y < canvas.height; y += 20) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }
      drawAnnotations(ctx)
    }

    function drawAnnotations(ctx: CanvasRenderingContext2D) {
      const allAnnotations = [...annotations]

      // Draw current drawing preview
      if (isDrawing && drawStart && currentCoords) {
        allAnnotations.push({
          tool: activeTool,
          coords: { x1: drawStart.x, y1: drawStart.y, x2: currentCoords.x, y2: currentCoords.y },
        })
      }

      for (const ann of allAnnotations) {
        ctx.strokeStyle = '#ef4444'
        ctx.lineWidth = 2
        ctx.fillStyle = 'rgba(239, 68, 68, 0.1)'

        const { x1, y1, x2, y2 } = ann.coords

        switch (ann.tool) {
          case 'rectangle':
            ctx.beginPath()
            ctx.rect(x1, y1, x2 - x1, y2 - y1)
            ctx.fill()
            ctx.stroke()
            break
          case 'circle': {
            const cx = (x1 + x2) / 2
            const cy = (y1 + y2) / 2
            const rx = Math.abs(x2 - x1) / 2
            const ry = Math.abs(y2 - y1) / 2
            ctx.beginPath()
            ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
            ctx.fill()
            ctx.stroke()
            break
          }
          case 'arrow': {
            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()
            // Arrowhead
            const angle = Math.atan2(y2 - y1, x2 - x1)
            const headLen = 12
            ctx.beginPath()
            ctx.moveTo(x2, y2)
            ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4))
            ctx.moveTo(x2, y2)
            ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4))
            ctx.stroke()
            break
          }
          case 'text':
            if (ann.text) {
              ctx.font = '14px sans-serif'
              ctx.fillStyle = '#ef4444'
              ctx.fillText(ann.text, x1, y1)
            }
            break
        }
      }
    }
  }, [annotations, isDrawing, drawStart, currentCoords, activeTool, imageUrl])

  const removeAnnotation = (index: number) => {
    const newAnnotations = annotations.filter((_, i) => i !== index)
    setAnnotations(newAnnotations)
    emitAnswer(newAnnotations)
  }

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center gap-1">
        {(Object.keys(TOOL_LABELS) as AnnotationTool[]).map(tool => (
          <button
            key={tool}
            onClick={() => setActiveTool(tool)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              activeTool === tool
                ? 'bg-primary-100 dark:bg-primary-900/40 text-primary-700 dark:text-primary-300 ring-1 ring-primary-500'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            {TOOL_LABELS[tool]}
          </button>
        ))}
        <div className="flex-1" />
        <span className="text-xs text-gray-400 dark:text-gray-500">
          {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Canvas */}
      <div
        ref={containerRef}
        className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
      >
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="w-full cursor-crosshair"
          style={{ aspectRatio: `${width}/${height}` }}
        />
      </div>

      {/* Annotation list */}
      {annotations.length > 0 && (
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {annotations.map((ann, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 dark:bg-gray-900/50 rounded text-xs text-gray-600 dark:text-gray-400"
            >
              <span className="font-medium">{ann.tool}</span>
              <span>({ann.coords.x1},{ann.coords.y1}) - ({ann.coords.x2},{ann.coords.y2})</span>
              {ann.text && <span className="italic">&quot;{ann.text}&quot;</span>}
              <button
                onClick={() => removeAnnotation(index)}
                className="ml-auto p-0.5 text-gray-400 hover:text-red-500 transition-colors"
              >
                <TrashIcon className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
