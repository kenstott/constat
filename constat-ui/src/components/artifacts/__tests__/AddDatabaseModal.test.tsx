// Copyright (c) 2025 Kenneth Stott
//
// Tests for AddDatabaseModal: per-type field rendering and auth type selectors.

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { AddDatabaseModal } from '../AddDatabaseModal'

function renderModal(onAdd = vi.fn(), onCancel = vi.fn()) {
  render(<AddDatabaseModal onAdd={onAdd} onCancel={onCancel} uploading={false} />)
}

function selectType(type: string) {
  fireEvent.change(screen.getByRole('combobox'), { target: { value: type } })
}

describe('AddDatabaseModal', () => {
  let onAdd: ReturnType<typeof vi.fn>
  let onCancel: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onAdd = vi.fn()
    onCancel = vi.fn()
  })

  // ---------------------------------------------------------------------------
  // Initial state
  // ---------------------------------------------------------------------------

  it('renders name input and type selector', () => {
    renderModal(onAdd, onCancel)
    expect(screen.getByPlaceholderText('Name')).toBeInTheDocument()
    expect(screen.getByRole('combobox')).toBeInTheDocument()
  })

  it('Add button disabled when no type selected', () => {
    renderModal(onAdd, onCancel)
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
  })

  it('Cancel calls onCancel', () => {
    renderModal(onAdd, onCancel)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledOnce()
  })

  // ---------------------------------------------------------------------------
  // File types
  // ---------------------------------------------------------------------------

  it('DuckDB shows file path field', () => {
    renderModal(onAdd, onCancel)
    selectType('duckdb')
    expect(screen.getByPlaceholderText(/\.duckdb/)).toBeInTheDocument()
    expect(screen.queryByPlaceholderText('localhost')).not.toBeInTheDocument()
  })

  it('SQLite shows file path field', () => {
    renderModal(onAdd, onCancel)
    selectType('sqlite')
    expect(screen.getByPlaceholderText(/\.db/)).toBeInTheDocument()
  })

  it('DuckDB Add button enabled after name + path filled', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'mydb' } })
    selectType('duckdb')
    fireEvent.change(screen.getByPlaceholderText(/\.duckdb/), { target: { value: '/data/test.duckdb' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('DuckDB calls onAdd with correct args', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'duck1' } })
    selectType('duckdb')
    fireEvent.change(screen.getByPlaceholderText(/\.duckdb/), { target: { value: '/data/duck.db' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledWith('duck1', 'duckdb:////data/duck.db', 'sqlalchemy', undefined)  // mapped to sqlalchemy; URI contains duckdb:/// for dialect detection
  })

  // ---------------------------------------------------------------------------
  // Custom URI
  // ---------------------------------------------------------------------------

  it('Custom shows URI field', () => {
    renderModal(onAdd, onCancel)
    selectType('custom')
    expect(screen.getByPlaceholderText(/dialect\+driver/)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // SQL network types (PostgreSQL / MySQL / MSSQL)
  // ---------------------------------------------------------------------------

  it('PostgreSQL shows host/port/database/username/password', () => {
    renderModal(onAdd, onCancel)
    selectType('postgresql')
    expect(screen.getByPlaceholderText('localhost')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('5432')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('database')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('user')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('password')).toBeInTheDocument()
  })

  it('MSSQL defaults to port 1433', () => {
    renderModal(onAdd, onCancel)
    selectType('mssql')
    expect(screen.getByPlaceholderText('1433')).toBeInTheDocument()
  })

  it('MySQL defaults to port 3306', () => {
    renderModal(onAdd, onCancel)
    selectType('mysql')
    expect(screen.getByPlaceholderText('3306')).toBeInTheDocument()
  })

  it('PostgreSQL builds correct URI via onAdd', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'pg1' } })
    selectType('postgresql')
    fireEvent.change(screen.getByPlaceholderText('localhost'), { target: { value: 'dbhost' } })
    fireEvent.change(screen.getByPlaceholderText('database'), { target: { value: 'mydb' } })
    fireEvent.change(screen.getByPlaceholderText('user'), { target: { value: 'admin' } })
    fireEvent.change(screen.getByPlaceholderText('password'), { target: { value: 'secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledOnce()
    const [, uri, type] = onAdd.mock.calls[0]
    expect(type).toBe('postgresql')
    expect(uri).toContain('postgresql+psycopg2://')
    expect(uri).toContain('admin')
    expect(uri).toContain('dbhost')
    expect(uri).toContain('mydb')
  })

  it('SQL Add button disabled until host filled', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'pg1' } })
    selectType('postgresql')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText('localhost'), { target: { value: 'myhost' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  // ---------------------------------------------------------------------------
  // NoSQL: MongoDB
  // ---------------------------------------------------------------------------

  it('MongoDB shows host/port/database and auth selector', () => {
    renderModal(onAdd, onCancel)
    selectType('mongodb')
    expect(screen.getByPlaceholderText('localhost')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('27017')).toBeInTheDocument()
    // Auth selector should be present (MongoDB has auth options)
    const selects = screen.getAllByRole('combobox')
    expect(selects.length).toBeGreaterThanOrEqual(2)
  })

  // ---------------------------------------------------------------------------
  // Elasticsearch auth types
  // ---------------------------------------------------------------------------

  it('Elasticsearch shows auth selector with 4 options', () => {
    renderModal(onAdd, onCancel)
    selectType('elasticsearch')
    const selects = screen.getAllByRole('combobox')
    // Find the auth selector (not the type selector)
    const authSelect = selects[selects.length - 1]
    const options = authSelect.querySelectorAll('option')
    expect(options.length).toBe(4) // none, basic, api_key, bearer
  })

  it('Elasticsearch + basic shows username/password', () => {
    renderModal(onAdd, onCancel)
    selectType('elasticsearch')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'basic' } })
    expect(screen.getByPlaceholderText('user')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('password')).toBeInTheDocument()
  })

  it('Elasticsearch + api_key shows API key field, hides user/pass', () => {
    renderModal(onAdd, onCancel)
    selectType('elasticsearch')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'api_key' } })
    expect(screen.getByPlaceholderText(/base64/i)).toBeInTheDocument()
    expect(screen.queryByPlaceholderText('user')).not.toBeInTheDocument()
  })

  it('Elasticsearch + bearer shows token field', () => {
    renderModal(onAdd, onCancel)
    selectType('elasticsearch')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'bearer' } })
    expect(screen.getByPlaceholderText('token')).toBeInTheDocument()
  })

  it('Elasticsearch + none hides credentials', () => {
    renderModal(onAdd, onCancel)
    selectType('elasticsearch')
    // default is 'none'
    expect(screen.queryByPlaceholderText('user')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText(/api key/i)).not.toBeInTheDocument()
  })

  it('Elasticsearch api_key passes extraConfig to onAdd', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'elastic' } })
    selectType('elasticsearch')
    fireEvent.change(screen.getByPlaceholderText('localhost'), { target: { value: 'es-host' } })
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'api_key' } })
    fireEvent.change(screen.getByPlaceholderText(/base64/i), { target: { value: 'myapikey' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledOnce()
    const [, , , extraConfig] = onAdd.mock.calls[0]
    expect(extraConfig).toEqual({ api_key: 'myapikey' })
  })

  // ---------------------------------------------------------------------------
  // DynamoDB
  // ---------------------------------------------------------------------------

  it('DynamoDB shows auth selector with env and credentials options', () => {
    renderModal(onAdd, onCancel)
    selectType('dynamodb')
    const selects = screen.getAllByRole('combobox')
    const authSelect = selects[selects.length - 1]
    const values = Array.from(authSelect.querySelectorAll('option')).map((o) => (o as HTMLOptionElement).value)
    expect(values).toContain('env')
    expect(values).toContain('credentials')
  })

  it('DynamoDB env shows IAM hint text', () => {
    renderModal(onAdd, onCancel)
    selectType('dynamodb')
    // The hint paragraph contains 'instance IAM role' — distinct from the option label
    expect(screen.getByText(/instance IAM role/i)).toBeInTheDocument()
  })

  it('DynamoDB credentials shows access key + secret + region fields', () => {
    renderModal(onAdd, onCancel)
    selectType('dynamodb')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'credentials' } })
    expect(screen.getByPlaceholderText(/AKIA/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText('secret')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('us-east-1')).toBeInTheDocument()
  })

  it('DynamoDB credentials passes extraConfig with aws fields', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'dynamo' } })
    selectType('dynamodb')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'credentials' } })
    fireEvent.change(screen.getByPlaceholderText(/AKIA/), { target: { value: 'AKIATEST' } })
    fireEvent.change(screen.getByPlaceholderText('secret'), { target: { value: 'mysecret' } })
    fireEvent.change(screen.getByPlaceholderText('us-east-1'), { target: { value: 'eu-west-1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledOnce()
    const [, , , extraConfig] = onAdd.mock.calls[0]
    expect(extraConfig?.aws_access_key_id).toBe('AKIATEST')
    expect(extraConfig?.aws_secret_access_key).toBe('mysecret')
    expect(extraConfig?.region).toBe('eu-west-1')
  })

  // ---------------------------------------------------------------------------
  // CosmosDB
  // ---------------------------------------------------------------------------

  it('CosmosDB shows endpoint and auth selector', () => {
    renderModal(onAdd, onCancel)
    selectType('cosmosdb')
    expect(screen.getByPlaceholderText(/documents\.azure\.com/)).toBeInTheDocument()
    const selects = screen.getAllByRole('combobox')
    const authSelect = selects[selects.length - 1]
    const values = Array.from(authSelect.querySelectorAll('option')).map((o) => (o as HTMLOptionElement).value)
    expect(values).toContain('env')
    expect(values).toContain('key')
  })

  it('CosmosDB explicit key shows account key field', () => {
    renderModal(onAdd, onCancel)
    selectType('cosmosdb')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'key' } })
    expect(screen.getByPlaceholderText(/primary or secondary/i)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // Firestore
  // ---------------------------------------------------------------------------

  it('Firestore shows project field and auth selector', () => {
    renderModal(onAdd, onCancel)
    selectType('firestore')
    expect(screen.getByPlaceholderText('my-gcp-project')).toBeInTheDocument()
    const selects = screen.getAllByRole('combobox')
    const authSelect = selects[selects.length - 1]
    const values = Array.from(authSelect.querySelectorAll('option')).map((o) => (o as HTMLOptionElement).value)
    expect(values).toContain('adc')
    expect(values).toContain('service_account')
  })

  it('Firestore service_account shows credentials path field', () => {
    renderModal(onAdd, onCancel)
    selectType('firestore')
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[selects.length - 1], { target: { value: 'service_account' } })
    expect(screen.getByPlaceholderText(/credentials\.json/)).toBeInTheDocument()
  })

  it('Firestore ADC shows env hint text', () => {
    renderModal(onAdd, onCancel)
    selectType('firestore')
    expect(screen.getByText(/GOOGLE_APPLICATION_CREDENTIALS/)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // Generic JDBC
  // ---------------------------------------------------------------------------

  it('JDBC shows driver class, URL, JAR path, and optional username/password fields', () => {
    renderModal(onAdd, onCancel)
    selectType('jdbc')
    expect(screen.getByPlaceholderText(/com\.example\.jdbc\.Driver/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/jdbc:vendor/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/\.jar/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText('user')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('password')).toBeInTheDocument()
  })

  it('JDBC shows JVM install hint', () => {
    renderModal(onAdd, onCancel)
    selectType('jdbc')
    expect(screen.getByText(/constat\[jdbc\]/)).toBeInTheDocument()
  })

  it('JDBC Add button disabled until driver + URL filled', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'myjdbc' } })
    selectType('jdbc')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText(/com\.example\.jdbc\.Driver/), { target: { value: 'com.sap.db.jdbc.Driver' } })
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText(/jdbc:vendor/), { target: { value: 'jdbc:sap://host:30015/' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('JDBC calls onAdd with correct type and extraConfig', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'sap' } })
    selectType('jdbc')
    fireEvent.change(screen.getByPlaceholderText(/com\.example\.jdbc\.Driver/), { target: { value: 'com.sap.db.jdbc.Driver' } })
    fireEvent.change(screen.getByPlaceholderText(/jdbc:vendor/), { target: { value: 'jdbc:sap://host:30015/' } })
    fireEvent.change(screen.getByPlaceholderText(/\.jar/), { target: { value: '/opt/ngdbc.jar' } })
    fireEvent.change(screen.getByPlaceholderText('user'), { target: { value: 'admin' } })
    fireEvent.change(screen.getByPlaceholderText('password'), { target: { value: 'pass' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledOnce()
    const [name, uri, type, extraConfig] = onAdd.mock.calls[0]
    expect(name).toBe('sap')
    expect(type).toBe('jdbc')
    expect(uri).toBe('jdbc:sap://host:30015/')
    expect(extraConfig?.jdbc_driver).toBe('com.sap.db.jdbc.Driver')
    expect(extraConfig?.jdbc_url).toBe('jdbc:sap://host:30015/')
    expect(extraConfig?.jar_path).toBe('/opt/ngdbc.jar')
    expect(extraConfig?.username).toBe('admin')
    expect(extraConfig?.password).toBe('pass')
  })

  it('JDBC does not show host/port/database network fields', () => {
    renderModal(onAdd, onCancel)
    selectType('jdbc')
    expect(screen.queryByPlaceholderText('localhost')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText('database')).not.toBeInTheDocument()
  })
})
