// Copyright (c) 2025 Kenneth Stott
// Canary: 8d6bb15b-26f6-419a-9871-1f70a0ec2f1e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState } from 'react'

// DB type categories
const FILE_TYPES = new Set(['duckdb', 'sqlite'])
const NOSQL_TYPES = new Set(['mongodb', 'elasticsearch', 'cassandra', 'dynamodb', 'cosmosdb', 'firestore', 'neo4j'])
const URI_TYPES = new Set(['custom'])

const DB_OPTIONS = [
  { value: 'postgresql', label: 'PostgreSQL' },
  { value: 'mysql', label: 'MySQL' },
  { value: 'mssql', label: 'SQL Server (MSSQL)' },
  { value: 'duckdb', label: 'DuckDB' },
  { value: 'sqlite', label: 'SQLite' },
  { value: 'mongodb', label: 'MongoDB' },
  { value: 'elasticsearch', label: 'Elasticsearch' },
  { value: 'cassandra', label: 'Cassandra' },
  { value: 'neo4j', label: 'Neo4j' },
  { value: 'dynamodb', label: 'DynamoDB' },
  { value: 'cosmosdb', label: 'Cosmos DB' },
  { value: 'firestore', label: 'Firestore' },
  { value: 'custom', label: 'Custom URI' },
]

const DEFAULT_PORTS: Record<string, string> = {
  postgresql: '5432',
  mysql: '3306',
  mssql: '1433',
  mongodb: '27017',
  elasticsearch: '9200',
  cassandra: '9042',
  neo4j: '7687',
}

function buildUri(type: string, fields: NetworkFields): string {
  const { host, port, username, password, database } = fields
  const user = username ? encodeURIComponent(username) : ''
  const pass = password ? encodeURIComponent(password) : ''
  const creds = user ? (pass ? `${user}:${pass}@` : `${user}@`) : ''
  const h = host || 'localhost'
  const p = port || DEFAULT_PORTS[type] || ''
  const portPart = p ? `:${p}` : ''
  const db = database ? `/${database}` : ''

  if (type === 'mongodb') return `mongodb://${creds}${h}${portPart}${db}`
  if (type === 'elasticsearch') return `http://${creds}${h}${portPart}`
  if (type === 'neo4j') return `bolt://${creds}${h}${portPart}${db}`
  if (type === 'cassandra') return `cassandra://${creds}${h}${portPart}${db}`
  if (type === 'postgresql') return `postgresql+psycopg2://${creds}${h}${portPart}${db}`
  if (type === 'mysql') return `mysql+pymysql://${creds}${h}${portPart}${db}`
  if (type === 'mssql') return `mssql+pyodbc://${creds}${h}${portPart}${db}?driver=ODBC+Driver+18+for+SQL+Server`
  return ''
}

interface NetworkFields {
  host: string
  port: string
  username: string
  password: string
  database: string
}

interface Props {
  onAdd: (name: string, uri: string, type: string) => void
  onCancel: () => void
  uploading: boolean
}

export function AddDatabaseModal({ onAdd, onCancel, uploading }: Props) {
  const [name, setName] = useState('')
  const [dbType, setDbType] = useState('')
  const [filePath, setFilePath] = useState('')
  const [customUri, setCustomUri] = useState('')
  const [net, setNet] = useState<NetworkFields>({ host: '', port: '', username: '', password: '', database: '' })

  const isFile = FILE_TYPES.has(dbType)
  const isUri = URI_TYPES.has(dbType)
  const isNetwork = dbType !== '' && !isFile && !isUri

  const handleSubmit = () => {
    if (!name || !dbType) return
    let uri = ''
    if (isFile) {
      uri = filePath
    } else if (isUri) {
      uri = customUri
    } else {
      uri = buildUri(dbType, net)
    }
    if (!uri) return
    onAdd(name, uri, dbType)
  }

  const canSubmit = () => {
    if (!name || !dbType) return false
    if (isFile) return filePath.length > 0
    if (isUri) return customUri.length > 0
    return net.host.length > 0
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const halfCls = 'flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const labelCls = 'text-xs text-gray-500 dark:text-gray-400 mb-0.5'

  return (
    <div className="space-y-3">
      <input
        type="text"
        placeholder="Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className={inputCls}
      />

      <select
        value={dbType}
        onChange={(e) => {
          setDbType(e.target.value)
          setNet({ host: '', port: DEFAULT_PORTS[e.target.value] || '', username: '', password: '', database: '' })
        }}
        className={inputCls}
      >
        <option value="">Select database type</option>
        {DB_OPTIONS.map(({ value, label }) => (
          <option key={value} value={value}>{label}</option>
        ))}
      </select>

      {isFile && (
        <div>
          <p className={labelCls}>{dbType === 'duckdb' ? 'DuckDB file path' : 'SQLite file path'}</p>
          <input
            type="text"
            placeholder={dbType === 'duckdb' ? '/path/to/file.duckdb' : '/path/to/file.db'}
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            className={inputCls}
          />
        </div>
      )}

      {isUri && (
        <div>
          <p className={labelCls}>Connection URI (SQLAlchemy format)</p>
          <input
            type="text"
            placeholder="dialect+driver://user:pass@host:port/db"
            value={customUri}
            onChange={(e) => setCustomUri(e.target.value)}
            className={inputCls}
          />
        </div>
      )}

      {isNetwork && !NOSQL_TYPES.has(dbType) && (
        <>
          <div className="flex gap-2">
            <div className="flex-[3]">
              <p className={labelCls}>Host</p>
              <input type="text" placeholder="localhost" value={net.host} onChange={(e) => setNet({ ...net, host: e.target.value })} className={halfCls} />
            </div>
            <div className="flex-1">
              <p className={labelCls}>Port</p>
              <input type="text" placeholder={DEFAULT_PORTS[dbType] || ''} value={net.port} onChange={(e) => setNet({ ...net, port: e.target.value })} className={halfCls} />
            </div>
          </div>
          <div>
            <p className={labelCls}>Database name</p>
            <input type="text" placeholder="database" value={net.database} onChange={(e) => setNet({ ...net, database: e.target.value })} className={inputCls} />
          </div>
          <div className="flex gap-2">
            <div className="flex-1">
              <p className={labelCls}>Username</p>
              <input type="text" placeholder="user" value={net.username} onChange={(e) => setNet({ ...net, username: e.target.value })} className={halfCls} />
            </div>
            <div className="flex-1">
              <p className={labelCls}>Password</p>
              <input type="password" placeholder="password" value={net.password} onChange={(e) => setNet({ ...net, password: e.target.value })} className={halfCls} />
            </div>
          </div>
        </>
      )}

      {isNetwork && NOSQL_TYPES.has(dbType) && dbType !== 'dynamodb' && dbType !== 'cosmosdb' && dbType !== 'firestore' && (
        <>
          <div className="flex gap-2">
            <div className="flex-[3]">
              <p className={labelCls}>Host</p>
              <input type="text" placeholder="localhost" value={net.host} onChange={(e) => setNet({ ...net, host: e.target.value })} className={halfCls} />
            </div>
            <div className="flex-1">
              <p className={labelCls}>Port</p>
              <input type="text" placeholder={DEFAULT_PORTS[dbType] || ''} value={net.port} onChange={(e) => setNet({ ...net, port: e.target.value })} className={halfCls} />
            </div>
          </div>
          {dbType !== 'elasticsearch' && (
            <div>
              <p className={labelCls}>Database / Keyspace</p>
              <input type="text" placeholder={dbType === 'cassandra' ? 'keyspace' : 'database'} value={net.database} onChange={(e) => setNet({ ...net, database: e.target.value })} className={inputCls} />
            </div>
          )}
          <div className="flex gap-2">
            <div className="flex-1">
              <p className={labelCls}>Username</p>
              <input type="text" placeholder="user" value={net.username} onChange={(e) => setNet({ ...net, username: e.target.value })} className={halfCls} />
            </div>
            <div className="flex-1">
              <p className={labelCls}>Password</p>
              <input type="password" placeholder="password" value={net.password} onChange={(e) => setNet({ ...net, password: e.target.value })} className={halfCls} />
            </div>
          </div>
        </>
      )}

      {isNetwork && (dbType === 'dynamodb' || dbType === 'cosmosdb' || dbType === 'firestore') && (
        <div>
          <p className={labelCls}>Connection URI / Endpoint</p>
          <input
            type="text"
            placeholder={
              dbType === 'dynamodb' ? 'http://localhost:8000 (or leave blank for AWS)' :
              dbType === 'cosmosdb' ? 'https://account.documents.azure.com:443/' :
              'projects/my-project/databases/(default)'
            }
            value={customUri}
            onChange={(e) => setCustomUri(e.target.value)}
            className={inputCls}
          />
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            {dbType === 'dynamodb' && 'AWS credentials will be read from environment / IAM role.'}
            {dbType === 'cosmosdb' && 'Account key will be read from AZURE_COSMOS_KEY env var.'}
            {dbType === 'firestore' && 'Credentials will be read from GOOGLE_APPLICATION_CREDENTIALS.'}
          </p>
        </div>
      )}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={uploading || !canSubmit()}
          className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {uploading && <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />}
          {uploading ? 'Adding...' : 'Add'}
        </button>
      </div>
    </div>
  )
}
