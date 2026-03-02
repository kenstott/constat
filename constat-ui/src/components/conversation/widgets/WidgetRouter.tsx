// WidgetRouter - Routes to the correct widget component based on question.widget?.type

import { ChoiceWidget } from './ChoiceWidget'
import { CurationWidget } from './CurationWidget'
import { RankingWidget } from './RankingWidget'
import { TableWidget } from './TableWidget'
import { MappingWidget } from './MappingWidget'
import { TreeWidget } from './TreeWidget'
import { AnnotationWidget } from './AnnotationWidget'

interface WidgetRouterProps {
  widget?: { type: string; config: Record<string, unknown> }
  suggestions: string[]
  value: string
  structuredValue?: unknown
  onAnswer: (freeform: string, structured?: unknown) => void
  customAnswer: string
  onCustomAnswerChange: (value: string) => void
  onSubmitShortcut?: () => void
}

/** Max-width class for the dialog based on widget type */
export function getWidgetMaxWidth(widgetType?: string): string {
  switch (widgetType) {
    case 'curation':
    case 'table':
      return 'max-w-2xl'
    case 'mapping':
    case 'annotation':
    case 'tree':
      return 'max-w-4xl'
    default:
      return 'max-w-lg'
  }
}

export function WidgetRouter({
  widget,
  suggestions,
  value,
  structuredValue,
  onAnswer,
  customAnswer,
  onCustomAnswerChange,
  onSubmitShortcut,
}: WidgetRouterProps) {
  const config = widget?.config || {}

  switch (widget?.type) {
    case 'curation':
      return (
        <CurationWidget
          config={config}
          value={value}
          structuredValue={structuredValue as { kept: string[]; removed: string[] } | undefined}
          onAnswer={onAnswer}
        />
      )
    case 'ranking':
      return (
        <RankingWidget
          config={config}
          value={value}
          structuredValue={structuredValue as { ranked: string[] } | undefined}
          onAnswer={onAnswer}
        />
      )
    case 'table':
      return (
        <TableWidget
          config={config}
          value={value}
          structuredValue={structuredValue as { rows: Record<string, unknown>[] } | undefined}
          onAnswer={onAnswer}
        />
      )
    case 'mapping':
      return (
        <MappingWidget
          config={config}
          value={value}
          structuredValue={structuredValue as { mappings: { left: string; right: string }[] } | undefined}
          onAnswer={onAnswer}
        />
      )
    case 'tree':
      return (
        <TreeWidget
          config={config}
          value={value}
          structuredValue={structuredValue}
          onAnswer={onAnswer}
        />
      )
    case 'annotation':
      return (
        <AnnotationWidget
          config={config}
          value={value}
          structuredValue={structuredValue}
          onAnswer={onAnswer}
        />
      )
    case 'choice':
    default:
      return (
        <ChoiceWidget
          config={config}
          suggestions={suggestions}
          value={value}
          onAnswer={onAnswer}
          customAnswer={customAnswer}
          onCustomAnswerChange={onCustomAnswerChange}
          onSubmitShortcut={onSubmitShortcut}
        />
      )
  }
}
