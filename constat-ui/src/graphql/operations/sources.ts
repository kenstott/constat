// Copyright (c) 2025 Kenneth Stott
// Canary: 6f234eac-9b72-4036-b99d-2021b3e8a8d3
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const DATABASES_QUERY = gql`
  query Databases($sessionId: String!) {
    databases(sessionId: $sessionId) {
      databases {
        name
        type
        dialect
        description
        connected
        tableCount
        addedAt
        isDynamic
        fileId
        source
      }
      total
    }
  }
`

export const DATA_SOURCES_QUERY = gql`
  query DataSources($sessionId: String!) {
    dataSources(sessionId: $sessionId) {
      databases {
        name
        type
        dialect
        description
        uri
        connected
        tableCount
        addedAt
        isDynamic
        fileId
        source
      }
      apis {
        name
        type
        description
        baseUrl
        connected
        fromConfig
        source
      }
      documents {
        name
        type
        description
        path
        indexed
        fromConfig
        source
        followLinks
        maxDepth
        maxDocuments
        sameDomainOnly
        excludePatterns
      }
    }
  }
`

export const DATABASE_TABLE_PREVIEW_QUERY = gql`
  query DatabaseTablePreview($sessionId: String!, $dbName: String!, $tableName: String!, $page: Int, $pageSize: Int) {
    databaseTablePreview(sessionId: $sessionId, dbName: $dbName, tableName: $tableName, page: $page, pageSize: $pageSize) {
      database
      tableName
      columns
      data
      page
      pageSize
      totalRows
      hasMore
    }
  }
`

export const FILES_QUERY = gql`
  query Files($sessionId: String!) {
    files(sessionId: $sessionId) {
      files {
        id
        filename
        fileUri
        sizeBytes
        contentType
        uploadedAt
      }
      total
    }
  }
`

export const FILE_REFS_QUERY = gql`
  query FileRefs($sessionId: String!) {
    fileRefs(sessionId: $sessionId) {
      fileRefs {
        name
        uri
        hasAuth
        description
        addedAt
        sessionId
      }
      total
    }
  }
`

export const USER_SOURCES_QUERY = gql`
  query UserSources {
    userSources {
      databases
      apis
      documents
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const ADD_DATABASE = gql`
  mutation AddDatabase($sessionId: String!, $input: DatabaseAddInput!) {
    addDatabase(sessionId: $sessionId, input: $input) {
      name
      type
      dialect
      description
      connected
      tableCount
      addedAt
      isDynamic
      fileId
      source
    }
  }
`

export type DatabaseAddInput = {
  name: string
  type?: string
  uri?: string
  fileId?: string
  description?: string
  extraConfig?: Record<string, unknown>
}

export const REMOVE_DATABASE = gql`
  mutation RemoveDatabase($sessionId: String!, $name: String!) {
    removeDatabase(sessionId: $sessionId, name: $name) {
      status
    }
  }
`

export const TEST_DATABASE = gql`
  mutation TestDatabase($sessionId: String!, $name: String!) {
    testDatabase(sessionId: $sessionId, name: $name) {
      name
      connected
      tableCount
      error
    }
  }
`

export const ADD_API = gql`
  mutation AddApi($sessionId: String!, $input: ApiAddInput!) {
    addApi(sessionId: $sessionId, input: $input) {
      name
      type
      description
      baseUrl
      connected
      fromConfig
      source
    }
  }
`

export const REMOVE_API = gql`
  mutation RemoveApi($sessionId: String!, $name: String!) {
    removeApi(sessionId: $sessionId, name: $name) {
      status
    }
  }
`

export const UPLOAD_FILE = gql`
  mutation UploadFile($sessionId: String!, $file: Upload!) {
    uploadFile(sessionId: $sessionId, file: $file) {
      id
      filename
      fileUri
      sizeBytes
      contentType
      uploadedAt
    }
  }
`

export const UPLOAD_FILE_DATA_URI = gql`
  mutation UploadFileDataUri($sessionId: String!, $filename: String!, $dataUri: String!) {
    uploadFileDataUri(sessionId: $sessionId, filename: $filename, dataUri: $dataUri) {
      id
      filename
      fileUri
      sizeBytes
      contentType
      uploadedAt
    }
  }
`

export const DELETE_FILE = gql`
  mutation DeleteFile($sessionId: String!, $fileId: String!) {
    deleteFile(sessionId: $sessionId, fileId: $fileId) {
      status
    }
  }
`

export const ADD_FILE_REF = gql`
  mutation AddFileRef($sessionId: String!, $input: FileRefInput!) {
    addFileRef(sessionId: $sessionId, input: $input) {
      name
      uri
      hasAuth
      description
      addedAt
      sessionId
    }
  }
`

export const DELETE_FILE_REF = gql`
  mutation DeleteFileRef($sessionId: String!, $name: String!) {
    deleteFileRef(sessionId: $sessionId, name: $name) {
      status
    }
  }
`

export const UPLOAD_DOCUMENTS = gql`
  mutation UploadDocuments($sessionId: String!, $files: [Upload!]!) {
    uploadDocuments(sessionId: $sessionId, files: $files) {
      status
      acceptedCount
      databaseCount
      totalFiles
      results {
        filename
        name
        status
        reason
        path
      }
    }
  }
`

export const ADD_DOCUMENT_URI = gql`
  mutation AddDocumentUri($sessionId: String!, $input: DocumentUriInput!) {
    addDocumentUri(sessionId: $sessionId, input: $input) {
      status
      name
    }
  }
`

export const ADD_EMAIL_SOURCE = gql`
  mutation AddEmailSource($sessionId: String!, $input: EmailSourceInput!) {
    addEmailSource(sessionId: $sessionId, input: $input) {
      status
      name
    }
  }
`

export const REFRESH_DOCUMENTS = gql`
  mutation RefreshDocuments($sessionId: String!) {
    refreshDocuments(sessionId: $sessionId) {
      status
    }
  }
`

export const REMOVE_USER_SOURCE = gql`
  mutation RemoveUserSource($sourceType: String!, $sourceName: String!) {
    removeUserSource(sourceType: $sourceType, sourceName: $sourceName) {
      status
      name
      sourceType
    }
  }
`

export const MOVE_SOURCE = gql`
  mutation MoveSource($sourceType: String!, $sourceName: String!, $fromDomain: String!, $toDomain: String!, $sessionId: String) {
    moveSource(sourceType: $sourceType, sourceName: $sourceName, fromDomain: $fromDomain, toDomain: $toDomain, sessionId: $sessionId) {
      status
      name
      sourceType
    }
  }
`

export const UPDATE_DATABASE = gql`
  mutation UpdateDatabase($sessionId: String!, $input: DatabaseUpdateInput!) {
    updateDatabase(sessionId: $sessionId, input: $input) {
      name description type uri isDynamic source
    }
  }
`

export const UPDATE_API = gql`
  mutation UpdateApi($sessionId: String!, $input: ApiUpdateInput!) {
    updateApi(sessionId: $sessionId, input: $input) {
      name description type baseUrl isDynamic fromConfig source
    }
  }
`

export const UPDATE_DOCUMENT = gql`
  mutation UpdateDocument($sessionId: String!, $input: DocumentUpdateInput!) {
    updateDocument(sessionId: $sessionId, input: $input) {
      name description path followLinks maxDepth maxDocuments sameDomainOnly excludePatterns
    }
  }
`

export const VALIDATE_URI = gql`
  query ValidateUri($uri: String!) {
    validateUri(uri: $uri) {
      reachable
      error
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toSessionDatabase(gql: any) {
  return {
    name: gql.name,
    type: gql.type,
    dialect: gql.dialect ?? undefined,
    description: gql.description ?? undefined,
    connected: gql.connected,
    table_count: gql.tableCount ?? undefined,
    added_at: gql.addedAt,
    is_dynamic: gql.isDynamic,
    file_id: gql.fileId ?? undefined,
    source: gql.source ?? undefined,
    uri: gql.uri ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toSessionApi(gql: any) {
  return {
    name: gql.name,
    type: gql.type ?? undefined,
    description: gql.description ?? undefined,
    base_url: gql.baseUrl ?? undefined,
    connected: gql.connected,
    from_config: gql.fromConfig,
    source: gql.source,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toSessionDocument(gql: any) {
  return {
    name: gql.name,
    type: gql.type ?? undefined,
    description: gql.description ?? undefined,
    path: gql.path ?? undefined,
    indexed: gql.indexed,
    from_config: gql.fromConfig,
    source: gql.source,
    follow_links: gql.followLinks ?? false,
    max_depth: gql.maxDepth ?? 2,
    max_documents: gql.maxDocuments ?? 50,
    same_domain_only: gql.sameDomainOnly ?? true,
    exclude_patterns: gql.excludePatterns ?? [],
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDataSources(gql: any) {
  return {
    databases: (gql.databases ?? []).map(toSessionDatabase),
    apis: (gql.apis ?? []).map(toSessionApi),
    documents: (gql.documents ?? []).map(toSessionDocument),
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDatabaseTablePreview(gql: any) {
  return {
    database: gql.database,
    table_name: gql.tableName,
    columns: gql.columns,
    data: gql.data,
    page: gql.page,
    page_size: gql.pageSize,
    total_rows: gql.totalRows,
    has_more: gql.hasMore,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toUploadedFile(gql: any) {
  return {
    id: gql.id,
    filename: gql.filename,
    file_uri: gql.fileUri,
    size_bytes: gql.sizeBytes,
    content_type: gql.contentType,
    uploaded_at: gql.uploadedAt,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toFileRef(gql: any) {
  return {
    name: gql.name,
    uri: gql.uri,
    has_auth: gql.hasAuth,
    description: gql.description ?? undefined,
    added_at: gql.addedAt,
    session_id: gql.sessionId ?? undefined,
  }
}
