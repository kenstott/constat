// Copyright (c) 2025 Kenneth Stott
// Canary: 72bfb06a-9e3d-4365-9c70-e25fe6a12ce6
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
  DATABASES_QUERY,
  DATABASE_TABLE_PREVIEW_QUERY,
  ADD_DATABASE,
  REMOVE_DATABASE,
  TEST_DATABASE,
  FILES_QUERY,
  UPLOAD_FILE_DATA_URI,
  DELETE_FILE,
  FILE_REFS_QUERY,
  ADD_FILE_REF,
  DELETE_FILE_REF,
  UPLOAD_DOCUMENTS,
  ADD_DOCUMENT_URI,
  ADD_EMAIL_SOURCE,
  REFRESH_DOCUMENTS,
  ADD_API,
  REMOVE_API,
  DATA_SOURCES_QUERY,
  USER_SOURCES_QUERY,
  REMOVE_USER_SOURCE,
  MOVE_SOURCE,
  toSessionDatabase,
  toDataSources,
  toDatabaseTablePreview,
  toUploadedFile,
  toFileRef,
} from '@/graphql/operations/sources'

export function useDatabases() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(DATABASES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const databases = (data?.databases?.databases ?? []).map(toSessionDatabase)
  return { databases, total: data?.databases?.total ?? 0, loading, error }
}

export function useDatabaseTablePreview(dbName: string, tableName: string, page = 1, pageSize = 100) {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(DATABASE_TABLE_PREVIEW_QUERY, {
    variables: { sessionId: sessionId!, dbName, tableName, page, pageSize },
    skip: !sessionId || !dbName || !tableName,
  })
  return {
    preview: data?.databaseTablePreview ? toDatabaseTablePreview(data.databaseTablePreview) : null,
    loading,
    error,
    refetch,
  }
}

export function useDatabaseMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['Databases', 'DataSources'] }
  const [addDatabaseMut] = useMutation(ADD_DATABASE, refetch)
  const [removeDatabaseMut] = useMutation(REMOVE_DATABASE, refetch)
  const [testDatabaseMut] = useMutation(TEST_DATABASE)

  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    addDatabase: (input: any) =>
      addDatabaseMut({ variables: { sessionId: sessionId!, input } }),
    removeDatabase: (name: string) =>
      removeDatabaseMut({ variables: { sessionId: sessionId!, name } }),
    testDatabase: (name: string) =>
      testDatabaseMut({ variables: { sessionId: sessionId!, name } }),
  }
}

export function useFiles() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(FILES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const files = (data?.files?.files ?? []).map(toUploadedFile)
  return { files, total: data?.files?.total ?? 0, loading, error }
}

export function useFileMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['Files'] }
  const [uploadFileDataUriMut] = useMutation(UPLOAD_FILE_DATA_URI, refetch)
  const [deleteFileMut] = useMutation(DELETE_FILE, refetch)

  return {
    uploadFileDataUri: (filename: string, dataUri: string) =>
      uploadFileDataUriMut({ variables: { sessionId: sessionId!, filename, dataUri } }),
    deleteFile: (fileId: string) =>
      deleteFileMut({ variables: { sessionId: sessionId!, fileId } }),
  }
}

export function useFileRefs() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(FILE_REFS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const fileRefs = (data?.fileRefs?.fileRefs ?? []).map(toFileRef)
  return { fileRefs, total: data?.fileRefs?.total ?? 0, loading, error }
}

export function useFileRefMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['FileRefs'] }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [addFileRefMut] = useMutation(ADD_FILE_REF, refetch)
  const [deleteFileRefMut] = useMutation(DELETE_FILE_REF, refetch)

  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    addFileRef: (input: any) =>
      addFileRefMut({ variables: { sessionId: sessionId!, input } }),
    deleteFileRef: (name: string) =>
      deleteFileRefMut({ variables: { sessionId: sessionId!, name } }),
  }
}

export function useDocumentMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['DataSources'] }
  const [uploadDocumentsMut] = useMutation(UPLOAD_DOCUMENTS, refetch)
  const [addDocumentUriMut] = useMutation(ADD_DOCUMENT_URI, refetch)
  const [addEmailSourceMut] = useMutation(ADD_EMAIL_SOURCE, refetch)
  const [refreshDocumentsMut] = useMutation(REFRESH_DOCUMENTS, refetch)

  return {
    uploadDocuments: (files: File[]) =>
      uploadDocumentsMut({ variables: { sessionId: sessionId!, files } }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    addDocumentUri: (input: any) =>
      addDocumentUriMut({ variables: { sessionId: sessionId!, input } }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    addEmailSource: (input: any) =>
      addEmailSourceMut({ variables: { sessionId: sessionId!, input } }),
    refreshDocuments: () =>
      refreshDocumentsMut({ variables: { sessionId: sessionId! } }),
  }
}

export function useApiMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['DataSources'] }
  const [addApiMut] = useMutation(ADD_API, refetch)
  const [removeApiMut] = useMutation(REMOVE_API, refetch)

  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    addApi: (input: any) =>
      addApiMut({ variables: { sessionId: sessionId!, input } }),
    removeApi: (name: string) =>
      removeApiMut({ variables: { sessionId: sessionId!, name } }),
  }
}

export function useDataSources() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(DATA_SOURCES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const sources = data?.dataSources ? toDataSources(data.dataSources) : { databases: [], apis: [], documents: [] }
  return { ...sources, loading, error }
}

export function useUserSources() {
  const { data, loading, error } = useQuery(USER_SOURCES_QUERY)
  return {
    databases: data?.userSources?.databases ?? [],
    apis: data?.userSources?.apis ?? [],
    documents: data?.userSources?.documents ?? [],
    loading,
    error,
  }
}

export function useUserSourceMutations() {
  const refetch = { refetchQueries: ['UserSources'] }
  const [removeUserSourceMut] = useMutation(REMOVE_USER_SOURCE, refetch)
  const [moveSourceMut] = useMutation(MOVE_SOURCE, refetch)

  return {
    removeUserSource: (sourceType: string, sourceName: string) =>
      removeUserSourceMut({ variables: { sourceType, sourceName } }),
    moveSource: (sourceType: string, sourceName: string, fromDomain: string, toDomain: string, sessionId?: string) =>
      moveSourceMut({ variables: { sourceType, sourceName, fromDomain, toDomain, sessionId } }),
  }
}
