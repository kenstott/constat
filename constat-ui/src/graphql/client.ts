import { ApolloClient, InMemoryCache, HttpLink, split } from '@apollo/client'
import { GraphQLWsLink } from '@apollo/client/link/subscriptions'
import { getMainDefinition } from '@apollo/client/utilities'
import { createClient } from 'graphql-ws'
import { CachePersistor } from 'apollo3-cache-persist'
import type { PersistentStorage } from 'apollo3-cache-persist/lib/types'
import { openDB } from 'idb'
import { useAuthStore } from '@/store/authStore'
import { typePolicies } from './cache-policies'

const DB_NAME = 'constat-apollo'
const STORE = 'cache'
const KEY = 'apollo-cache'

class IDBStorage implements PersistentStorage<string> {
  private dbPromise = openDB(DB_NAME, 1, {
    upgrade(db) { db.createObjectStore(STORE) },
  })

  async getItem(key: string): Promise<string | null> {
    return (await this.dbPromise).get(STORE, key) ?? null
  }

  async setItem(key: string, value: string): Promise<void> {
    await (await this.dbPromise).put(STORE, value, key)
  }

  async removeItem(key: string): Promise<void> {
    await (await this.dbPromise).delete(STORE, key)
  }
}

const httpLink = new HttpLink({
  uri: '/api/graphql',
  fetch: async (uri, options) => {
    const token = await useAuthStore.getState().getToken()
    const headers = new Headers((options as RequestInit)?.headers)
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
    return fetch(uri, { ...options, headers })
  },
})

const wsLink = new GraphQLWsLink(
  createClient({
    url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/graphql`,
    connectionParams: async () => {
      const token = await useAuthStore.getState().getToken()
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
  key: KEY,
  maxSize: false,
  trigger: 'write',
  debug: import.meta.env.DEV,
})

export const apolloClient = new ApolloClient({
  link: splitLink,
  cache,
})
