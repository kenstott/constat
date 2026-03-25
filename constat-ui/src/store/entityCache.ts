import { openDB, type DBSchema, type IDBPDatabase } from 'idb'
import type { CompactState } from './entityCacheKeys'

const DB_NAME = 'constat-entity-cache'
const DB_VERSION = 1
const STORE_NAME = 'sessions' as const

export interface CachedEntry {
  state: CompactState
  version: number
}

interface EntityCacheDB extends DBSchema {
  [STORE_NAME]: {
    key: string
    value: CachedEntry
  }
}

let dbPromise: Promise<IDBPDatabase<EntityCacheDB>> | null = null

function getDB(): Promise<IDBPDatabase<EntityCacheDB>> {
  if (!dbPromise) {
    dbPromise = openDB<EntityCacheDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        db.createObjectStore(STORE_NAME)
      },
    })
  }
  return dbPromise
}

export async function getCachedEntry(sessionId: string): Promise<CachedEntry | null> {
  const db = await getDB()
  const entry = await db.get(STORE_NAME, sessionId)
  return entry ?? null
}

export async function setCachedEntry(sessionId: string, state: CompactState, version: number): Promise<void> {
  const db = await getDB()
  await db.put(STORE_NAME, { state, version }, sessionId)
}

export async function clearCachedState(sessionId: string): Promise<void> {
  const db = await getDB()
  await db.delete(STORE_NAME, sessionId)
}
