// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { PublicClientApplication, type Configuration } from '@azure/msal-browser'

let msalInstance: PublicClientApplication | null = null

export async function initMsal(clientId: string, tenantId: string): Promise<PublicClientApplication> {
  if (msalInstance) return msalInstance
  const config: Configuration = {
    auth: {
      clientId,
      authority: `https://login.microsoftonline.com/${tenantId}`,
      redirectUri: window.location.origin,
    },
  }
  msalInstance = new PublicClientApplication(config)
  await msalInstance.initialize()
  return msalInstance
}

export async function signInWithMicrosoft(clientId: string, tenantId: string): Promise<{ code: string; redirectUri: string }> {
  const msal = await initMsal(clientId, tenantId)
  const result = await msal.loginPopup({
    scopes: ['openid', 'email', 'profile'],
  })
  return {
    code: result.idToken,
    redirectUri: window.location.origin,
  }
}
