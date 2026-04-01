// Copyright (c) 2025 Kenneth Stott
// Canary: 6d764026-5656-41f8-8283-3f40126b1997
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { ApolloClient, InMemoryCache, split } from '@apollo/client'
import createUploadLink from 'apollo-upload-client/createUploadLink.mjs'
import { GraphQLWsLink } from '@apollo/client/link/subscriptions'
import { getMainDefinition } from '@apollo/client/utilities'
import { createClient } from 'graphql-ws'
import { CachePersistor } from 'apollo3-cache-persist'
import type { PersistentStorage } from 'apollo3-cache-persist/lib/types'
import { getToken } from '@/config/auth-helpers'
import { typePolicies } from './cache-policies'

const DB_NAME = 'constat-apollo'
const STORE = 'cache'

class IDBStorage implements PersistentStorage<string> {
  private dbReady: Promise<IDBDatabase>

  constructor() {
    this.dbReady = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1)
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE)
      }
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
  }

  async getItem(key: string): Promise<string | null> {
    const db = await this.dbReady
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly')
      const req = tx.objectStore(STORE).get(key)
      req.onsuccess = () => resolve(req.result ?? null)
      req.onerror = () => reject(req.error)
    })
  }

  async setItem(key: string, value: string): Promise<void> {
    const db = await this.dbReady
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).put(value, key)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  }

  async removeItem(key: string): Promise<void> {
    const db = await this.dbReady
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).delete(key)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  }
}

const httpLink = createUploadLink({
  uri: '/api/graphql',
  fetch: async (uri: RequestInfo | URL, options?: RequestInit) => {
    const token = await getToken()
    const headers = new Headers(options?.headers)
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
    return fetch(uri, { ...options, headers })
  },
}) as unknown as ReturnType<typeof createUploadLink>

const wsLink = new GraphQLWsLink(
  createClient({
    url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/graphql`,
    connectionParams: async () => {
      const token = await getToken()
      return token ? { authorization: `Bearer ${token}` } : {}
    },
  })
)

const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query)
    return definition.kind === 'OperationDefinition' && definition.operation === 'subscription'
  },
  wsLink,
  httpLink,
)

export const cache = new InMemoryCache({ typePolicies })

export const cachePersistor = new CachePersistor({
  cache,
  storage: new IDBStorage(),
  maxSize: false,
  trigger: 'write',
})

export const apolloClient = new ApolloClient({
  link: splitLink,
  cache,
  devtools: { enabled: import.meta.env.DEV },
})
