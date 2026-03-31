// Copyright (c) 2025 Kenneth Stott
// Canary: 93cd5786-134b-4179-9652-dd00e453d2d4
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery } from '@apollo/client'
import { CONFIG_QUERY, MY_PERMISSIONS_QUERY } from '@/graphql/operations/auth'

export function useConfig() {
  const { data, loading, error } = useQuery(CONFIG_QUERY)
  return { config: data?.config ?? null, loading, error }
}

export function usePermissions() {
  const { data, loading, error } = useQuery(MY_PERMISSIONS_QUERY)
  return { permissions: data?.myPermissions ?? null, loading, error }
}
