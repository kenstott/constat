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

const FILE_TYPES = new Set(['duckdb', 'sqlite'])
const URI_TYPES = new Set(['custom'])
const CLOUD_TYPES = new Set(['dynamodb', 'cosmosdb', 'firestore'])

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

type AuthOption = { value: string; label: string }

const DB_AUTH_OPTIONS: Record<string, AuthOption[]> = {
  elasticsearch: [
    { value: 'none', label: 'No Auth' },
    { value: 'basic', label: 'Username / Password' },
    { value: 'api_key', label: 'API Key' },
    { value: 'bearer', label: 'Bearer Token' },
  ],
  mongodb: [
    { value: 'none', label: 'No Auth' },
    { value: 'basic', label: 'Username / Password' },
  ],
  cassandra: [
    { value: 'none', label: 'No Auth' },
    { value: 'basic', label: 'Username / Password' },
  ],
  dynamodb: [
    { value: 'env', label: 'Environment / IAM Role' },
    { value: 'credentials', label: 'Access Key + Secret' },
  ],
  cosmosdb: [
    { value: 'env', label: 'AZURE_COSMOS_KEY env var' },
    { value: 'key', label: 'Account Key' },
  ],
  firestore: [
    { value: 'adc', label: 'Application Default Credentials' },
    { value: 'service_account', label: 'Service Account Key Path' },
  ],
}

function defaultAuth(dbType: string): string {
  const opts = DB_AUTH_OPTIONS[dbType]
  return opts ? opts[0].value : 'basic'
}

interface NetworkFields {
  host: string
  port: string
  username: string
  password: string
  database: string
}

interface Props {
  onAdd: (name: string, uri: string, type: string, extraConfig?: Record<string, unknown>) => void
  onCancel: () => void
  uploading: boolean
}

function buildNetworkUri(type: string, net: NetworkFields): string {
  const { host, port, username, password, database } = net
  const u = username ? encodeURIComponent(username) : ''
  const p = password ? encodeURIComponent(password) : ''
  const creds = u ? (p ? `${u}:${p}@` : `${u}@`) : ''
  const h = host || 'localhost'
  const portPart = port ? `:${port}` : ''
  const db = database ? `/${database}` : ''

  if (type === 'postgresql') return `postgresql+psycopg2://${creds}${h}${portPart}${db}`
  if (type === 'mysql') return `mysql+pymysql://${creds}${h}${portPart}${db}`
  if (type === 'mssql') return `mssql+pyodbc://${creds}${h}${portPart}${db}?driver=ODBC+Driver+18+for+SQL+Server`
  if (type === 'mongodb') return `mongodb://${creds}${h}${portPart}${db}`
  if (type === 'elasticsearch') return `http://${h}${portPart}`
  if (type === 'neo4j') return `bolt://${creds}${h}${portPart}${db}`
  if (type === 'cassandra') return `cassandra://${creds}${h}${portPart}${db}`
  return `${type}://${creds}${h}${portPart}${db}`
}

export function AddDatabaseModal({ onAdd, onCancel, uploading }: Props) {
  const [name, setName] = useState('')
  const [dbType, setDbType] = useState('')
  const [authType, setAuthType] = useState('')
  const [filePath, setFilePath] = useState('')
  const [customUri, setCustomUri] = useState('')
  const [net, setNet] = useState<NetworkFields>({ host: '', port: '', username: '', password: '', database: '' })
  // Extra auth fields
  const [apiKey, setApiKey] = useState('')
  const [bearerToken, setBearerToken] = useState('')
  const [region, setRegion] = useState('')
  const [accessKeyId, setAccessKeyId] = useState('')
  const [secretAccessKey, setSecretAccessKey] = useState('')
  const [endpoint, setEndpoint] = useState('')
  const [accountKey, setAccountKey] = useState('')
  const [project, setProject] = useState('')
  const [credentialsPath, setCredentialsPath] = useState('')

  const isFile = FILE_TYPES.has(dbType)
  const isUri = URI_TYPES.has(dbType)
  const isCloud = CLOUD_TYPES.has(dbType)
  const isNetwork = dbType !== '' && !isFile && !isUri && !isCloud
  const authOptions = DB_AUTH_OPTIONS[dbType] ?? null
  const hasAuthSelector = authOptions !== null && isNetwork

  function handleTypeChange(val: string) {
    setDbType(val)
    setNet({ host: '', port: DEFAULT_PORTS[val] || '', username: '', password: '', database: '' })
    setAuthType(defaultAuth(val))
    setApiKey(''); setBearerToken(''); setRegion(''); setAccessKeyId('')
    setSecretAccessKey(''); setEndpoint(''); setAccountKey(''); setProject(''); setCredentialsPath('')
  }

  function buildResult(): { uri: string; extra: Record<string, unknown> } {
    const extra: Record<string, unknown> = {}

    if (isFile) {
      // Build SQLAlchemy-compatible URIs: sqlite:///relative.db or sqlite:////absolute/path.db
      if (dbType === 'sqlite') {
        const uri = filePath.startsWith('sqlite:') ? filePath : `sqlite:///${filePath}`
        return { uri, extra }
      }
      if (dbType === 'duckdb') {
        const uri = filePath.startsWith('duckdb:') ? filePath : `duckdb:///${filePath}`
        return { uri, extra }
      }
      return { uri: filePath, extra }
    }
    if (isUri) return { uri: customUri, extra }

    if (isCloud) {
      if (dbType === 'dynamodb') {
        if (authType === 'credentials') {
          extra.aws_access_key_id = accessKeyId
          extra.aws_secret_access_key = secretAccessKey
          if (region) extra.region = region
          if (endpoint) extra.endpoint_url = endpoint
        } else {
          if (region) extra.region = region
          if (endpoint) extra.endpoint_url = endpoint
        }
        return { uri: endpoint || '', extra }
      }
      if (dbType === 'cosmosdb') {
        if (accountKey) extra.key = accountKey
        if (net.database) extra.database = net.database
        return { uri: endpoint, extra }
      }
      if (dbType === 'firestore') {
        if (credentialsPath) extra.credentials_path = credentialsPath
        if (net.database) extra.collection = net.database
        return { uri: `firestore://${project}`, extra }
      }
    }

    // Network types
    const uri = buildNetworkUri(dbType, net)
    if (dbType === 'elasticsearch') {
      if (authType === 'api_key') extra.api_key = apiKey
      else if (authType === 'bearer') extra.bearer_token = bearerToken
    }
    return { uri, extra }
  }

  function handleSubmit() {
    if (!name || !dbType) return
    const { uri, extra } = buildResult()
    if (!uri && !isCloud) return
    // SQLite/DuckDB use SQLAlchemy driver — report as 'sqlalchemy' so backend accepts the URI
    const effectiveType = (dbType === 'sqlite' || dbType === 'duckdb') ? 'sqlalchemy' : dbType
    onAdd(name, uri, effectiveType, Object.keys(extra).length ? extra : undefined)
  }

  function canSubmit() {
    if (!name || !dbType) return false
    if (isFile) return filePath.length > 0
    if (isUri) return customUri.length > 0
    if (isCloud) {
      if (dbType === 'dynamodb') return authType === 'env' || (accessKeyId.length > 0 && secretAccessKey.length > 0)
      if (dbType === 'cosmosdb') return endpoint.length > 0
      if (dbType === 'firestore') return project.length > 0
    }
    return net.host.length > 0
  }

  const input = 'w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const half = 'flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const label = 'text-xs text-gray-500 dark:text-gray-400 mb-0.5'
  const hint = 'text-xs text-gray-400 dark:text-gray-500 mt-1'

  return (
    <div className="space-y-3">
      <input type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} className={input} />

      <select
        value={dbType}
        onChange={(e) => handleTypeChange(e.target.value)}
        className={input}
      >
        <option value="">Select database type</option>
        {DB_OPTIONS.map(({ value, label: lbl }) => (
          <option key={value} value={value}>{lbl}</option>
        ))}
      </select>

      {/* File-based */}
      {isFile && (
        <div>
          <p className={label}>{dbType === 'duckdb' ? 'DuckDB file path' : 'SQLite file path'}</p>
          <input
            type="text"
            placeholder={dbType === 'duckdb' ? '/path/to/file.duckdb' : '/path/to/file.db'}
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            className={input}
          />
        </div>
      )}

      {/* Custom URI */}
      {isUri && (
        <div>
          <p className={label}>Connection URI (SQLAlchemy format)</p>
          <input type="text" placeholder="dialect+driver://user:pass@host:port/db" value={customUri} onChange={(e) => setCustomUri(e.target.value)} className={input} />
        </div>
      )}

      {/* Network: host + port */}
      {isNetwork && (
        <div className="flex gap-2">
          <div className="flex-[3]">
            <p className={label}>Host</p>
            <input type="text" placeholder="localhost" value={net.host} onChange={(e) => setNet({ ...net, host: e.target.value })} className={half} />
          </div>
          <div className="flex-1">
            <p className={label}>Port</p>
            <input type="text" placeholder={DEFAULT_PORTS[dbType] || ''} value={net.port} onChange={(e) => setNet({ ...net, port: e.target.value })} className={half} />
          </div>
        </div>
      )}

      {/* Network: database/keyspace */}
      {isNetwork && dbType !== 'elasticsearch' && (
        <div>
          <p className={label}>{dbType === 'cassandra' ? 'Keyspace' : 'Database'}</p>
          <input type="text" placeholder={dbType === 'cassandra' ? 'keyspace' : 'database'} value={net.database} onChange={(e) => setNet({ ...net, database: e.target.value })} className={input} />
        </div>
      )}

      {/* Auth type selector */}
      {hasAuthSelector && (
        <div>
          <p className={label}>Authentication</p>
          <select value={authType} onChange={(e) => setAuthType(e.target.value)} className={input}>
            {authOptions.map(({ value, label: lbl }) => (
              <option key={value} value={value}>{lbl}</option>
            ))}
          </select>
        </div>
      )}

      {/* Username / Password */}
      {isNetwork && (!authOptions || authType === 'basic') && (
        <div className="flex gap-2">
          <div className="flex-1">
            <p className={label}>Username</p>
            <input type="text" placeholder="user" value={net.username} onChange={(e) => setNet({ ...net, username: e.target.value })} className={half} />
          </div>
          <div className="flex-1">
            <p className={label}>Password</p>
            <input type="password" placeholder="password" value={net.password} onChange={(e) => setNet({ ...net, password: e.target.value })} className={half} />
          </div>
        </div>
      )}

      {/* Elasticsearch: API Key */}
      {isNetwork && dbType === 'elasticsearch' && authType === 'api_key' && (
        <div>
          <p className={label}>API Key</p>
          <input type="password" placeholder="base64-encoded API key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={input} />
        </div>
      )}

      {/* Elasticsearch: Bearer Token */}
      {isNetwork && dbType === 'elasticsearch' && authType === 'bearer' && (
        <div>
          <p className={label}>Bearer Token</p>
          <input type="password" placeholder="token" value={bearerToken} onChange={(e) => setBearerToken(e.target.value)} className={input} />
        </div>
      )}

      {/* DynamoDB */}
      {isCloud && dbType === 'dynamodb' && (
        <>
          <div>
            <p className={label}>Authentication</p>
            <select value={authType} onChange={(e) => setAuthType(e.target.value)} className={input}>
              {DB_AUTH_OPTIONS.dynamodb.map(({ value, label: lbl }) => (
                <option key={value} value={value}>{lbl}</option>
              ))}
            </select>
          </div>
          {authType === 'credentials' && (
            <>
              <div className="flex gap-2">
                <div className="flex-1">
                  <p className={label}>Access Key ID</p>
                  <input type="text" placeholder="AKIAIOSFODNN7EXAMPLE" value={accessKeyId} onChange={(e) => setAccessKeyId(e.target.value)} className={half} />
                </div>
                <div className="flex-1">
                  <p className={label}>Secret Access Key</p>
                  <input type="password" placeholder="secret" value={secretAccessKey} onChange={(e) => setSecretAccessKey(e.target.value)} className={half} />
                </div>
              </div>
              <div>
                <p className={label}>Region</p>
                <input type="text" placeholder="us-east-1" value={region} onChange={(e) => setRegion(e.target.value)} className={input} />
              </div>
            </>
          )}
          {authType === 'env' && (
            <p className={hint}>Credentials read from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars or instance IAM role.</p>
          )}
          <div>
            <p className={label}>Endpoint URL (optional, for local DynamoDB)</p>
            <input type="text" placeholder="http://localhost:8000" value={endpoint} onChange={(e) => setEndpoint(e.target.value)} className={input} />
          </div>
        </>
      )}

      {/* CosmosDB */}
      {isCloud && dbType === 'cosmosdb' && (
        <>
          <div>
            <p className={label}>Endpoint</p>
            <input type="text" placeholder="https://account.documents.azure.com:443/" value={endpoint} onChange={(e) => setEndpoint(e.target.value)} className={input} />
          </div>
          <div>
            <p className={label}>Database</p>
            <input type="text" placeholder="mydb" value={net.database} onChange={(e) => setNet({ ...net, database: e.target.value })} className={input} />
          </div>
          <div>
            <p className={label}>Authentication</p>
            <select value={authType} onChange={(e) => setAuthType(e.target.value)} className={input}>
              {DB_AUTH_OPTIONS.cosmosdb.map(({ value, label: lbl }) => (
                <option key={value} value={value}>{lbl}</option>
              ))}
            </select>
          </div>
          {authType === 'key' && (
            <div>
              <p className={label}>Account Key</p>
              <input type="password" placeholder="primary or secondary account key" value={accountKey} onChange={(e) => setAccountKey(e.target.value)} className={input} />
            </div>
          )}
          {authType === 'env' && (
            <p className={hint}>Account key read from AZURE_COSMOS_KEY environment variable.</p>
          )}
        </>
      )}

      {/* Firestore */}
      {isCloud && dbType === 'firestore' && (
        <>
          <div>
            <p className={label}>GCP Project ID</p>
            <input type="text" placeholder="my-gcp-project" value={project} onChange={(e) => setProject(e.target.value)} className={input} />
          </div>
          <div>
            <p className={label}>Collection (optional)</p>
            <input type="text" placeholder="users" value={net.database} onChange={(e) => setNet({ ...net, database: e.target.value })} className={input} />
          </div>
          <div>
            <p className={label}>Authentication</p>
            <select value={authType} onChange={(e) => setAuthType(e.target.value)} className={input}>
              {DB_AUTH_OPTIONS.firestore.map(({ value, label: lbl }) => (
                <option key={value} value={value}>{lbl}</option>
              ))}
            </select>
          </div>
          {authType === 'service_account' && (
            <div>
              <p className={label}>Credentials JSON Path</p>
              <input type="text" placeholder="/path/to/credentials.json" value={credentialsPath} onChange={(e) => setCredentialsPath(e.target.value)} className={input} />
            </div>
          )}
          {authType === 'adc' && (
            <p className={hint}>Credentials read from GOOGLE_APPLICATION_CREDENTIALS environment variable.</p>
          )}
        </>
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
