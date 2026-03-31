// Copyright (c) 2025 Kenneth Stott
// Canary: dd4a19fa-7e1f-4505-9bfc-640417b60c65
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery, useMutation } from '@apollo/client'
import { useSessionContext } from '@/contexts/SessionContext'
import {
  TABLES_QUERY,
  TABLE_DATA_QUERY,
  TABLE_VERSIONS_QUERY,
  TABLE_VERSION_DATA_QUERY,
  DELETE_TABLE,
  TOGGLE_TABLE_STAR,
  toTableInfo,
  toTableData,
  toTableVersions,
} from '@/graphql/operations/data'

export function useTables() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(TABLES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const tables = (data?.tables?.tables ?? []).map(toTableInfo)
  return { tables, total: data?.tables?.total ?? 0, loading, error }
}

export function useTableData(name: string, page = 1, pageSize = 100) {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(TABLE_DATA_QUERY, {
    variables: { sessionId: sessionId!, name, page, pageSize },
    skip: !sessionId || !name,
  })
  return {
    ...(data?.tableData ? toTableData(data.tableData) : { columns: [], data: [], total_rows: 0, page, page_size: pageSize, has_more: false, name }),
    loading,
    error,
    refetch,
  }
}

export function useTableVersions(name: string) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(TABLE_VERSIONS_QUERY, {
    variables: { sessionId: sessionId!, name },
    skip: !sessionId || !name,
  })
  return {
    versions: data?.tableVersions ? toTableVersions(data.tableVersions) : null,
    loading,
    error,
  }
}

export function useTableVersionData(name: string, version: number, page = 1, pageSize = 100) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(TABLE_VERSION_DATA_QUERY, {
    variables: { sessionId: sessionId!, name, version, page, pageSize },
    skip: !sessionId || !name,
  })
  return {
    ...(data?.tableVersionData ? toTableData(data.tableVersionData) : { columns: [], data: [], total_rows: 0, page, page_size: pageSize, has_more: false, name }),
    loading,
    error,
  }
}

export function useTableMutations() {
  const { sessionId } = useSessionContext()
  const [deleteTableMut] = useMutation(DELETE_TABLE, {
    refetchQueries: ['Tables'],
  })
  const [toggleStarMut] = useMutation(TOGGLE_TABLE_STAR)

  return {
    deleteTable: (name: string) =>
      deleteTableMut({ variables: { sessionId: sessionId!, name } }),
    toggleStar: (name: string) =>
      toggleStarMut({ variables: { sessionId: sessionId!, name } }),
  }
}
