// Copyright (c) 2025 Kenneth Stott
// Canary: ade7d46c-5035-48a0-9a47-38fab8b662c6
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ApolloProvider } from '@apollo/client'
import { apolloClient, cachePersistor } from '@/graphql/client'
import { AuthProvider } from '@/contexts/AuthContext'
import { SessionProvider } from '@/contexts/SessionContext'
import { ArtifactProvider } from '@/contexts/ArtifactContext'
import './index.css'
import App from './App'

// Restore cache from IndexedDB, then render (with timeout fallback)
const renderApp = () => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <BrowserRouter>
        <ApolloProvider client={apolloClient}>
          <AuthProvider>
            <SessionProvider>
              <ArtifactProvider>
                <App />
              </ArtifactProvider>
            </SessionProvider>
          </AuthProvider>
        </ApolloProvider>
      </BrowserRouter>
    </StrictMode>,
  )
}

// Race: restore cache or timeout after 2s
Promise.race([
  cachePersistor.restore(),
  new Promise(resolve => setTimeout(resolve, 2000)),
]).catch((err) => {
  console.warn('[apollo-cache] Failed to restore from IndexedDB:', err)
}).then(renderApp)
