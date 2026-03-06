interface DomainBadgeProps {
  domain: string | null | undefined
  domainPath?: string | null
}

export function DomainBadge({ domain, domainPath }: DomainBadgeProps) {
  if (!domain) return null
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 flex-shrink-0"
      title={domainPath || domain}
    >
      {domainPath || domain}
    </span>
  )
}
