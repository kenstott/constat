interface ScopeBadgeProps {
  scope?: { level?: string; data_sources?: Array<{ name?: string; type?: string }> } | null
}

export function ScopeBadge({ scope }: ScopeBadgeProps) {
  if (!scope || !scope.level) return null

  const level = scope.level
  const sources = scope.data_sources || []

  if (level === 'instance') {
    const names = sources.map(s => s.name).filter(Boolean)
    if (names.length === 0) return null
    return (
      <>
        {names.map(name => (
          <span
            key={name}
            className="text-[10px] px-1.5 py-0.5 rounded bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 flex-shrink-0"
            title={`Instance-scoped: ${name}`}
          >
            {name}
          </span>
        ))}
      </>
    )
  }

  if (level === 'type') {
    const types = sources.map(s => s.type).filter(Boolean)
    if (types.length === 0) return null
    return (
      <>
        {types.map(t => (
          <span
            key={t}
            className="text-[10px] px-1.5 py-0.5 rounded bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 flex-shrink-0"
            title={`Type-scoped: ${t}`}
          >
            {t}
          </span>
        ))}
      </>
    )
  }

  // global
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 flex-shrink-0">
      global
    </span>
  )
}
