// Copyright (c) 2025 Kenneth Stott
// Canary: 395506ff-da86-47aa-9194-8982049fdd08
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useAuth } from '@/contexts/AuthContext'

interface DomainBadgeProps {
  domain: string | null | undefined
  domainPath?: string | null
}

export function DomainBadge({ domain, domainPath }: DomainBadgeProps) {
  const { userId } = useAuth()
  if (!domain) return null
  // Resolve personal domain: if domain matches the current user's UID, show "user"
  const label = domainPath || (domain === userId ? 'user' : domain)
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 flex-shrink-0"
      title={label}
    >
      {label}
    </span>
  )
}
