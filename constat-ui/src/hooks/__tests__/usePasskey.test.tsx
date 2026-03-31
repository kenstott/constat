// Copyright (c) 2025 Kenneth Stott
// Canary: e520e045-c165-4157-917a-f1217e71df5c
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { MockedProvider, MockedResponse } from '@apollo/client/testing'
import { usePasskey } from '../usePasskey'
import {
  PASSKEY_REGISTER_BEGIN,
  PASSKEY_REGISTER_COMPLETE,
  PASSKEY_AUTH_BEGIN,
  PASSKEY_AUTH_COMPLETE,
} from '@/graphql/operations/auth'
import type { ReactNode } from 'react'

// Mock WebAuthn browser APIs
const mockCredential = {
  id: 'cred-id-123',
  rawId: new ArrayBuffer(8),
  type: 'public-key',
  response: {
    attestationObject: new ArrayBuffer(8),
    clientDataJSON: new ArrayBuffer(8),
    authenticatorData: new ArrayBuffer(8),
    signature: new ArrayBuffer(8),
    userHandle: new ArrayBuffer(8),
  },
  getClientExtensionResults: () => ({}),
}

Object.defineProperty(global, 'navigator', {
  value: {
    credentials: {
      create: vi.fn().mockResolvedValue(mockCredential),
      get: vi.fn().mockResolvedValue(mockCredential),
    },
  },
  writable: true,
})

function wrapper(mocks: MockedResponse[]) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MockedProvider mocks={mocks}>
        {children}
      </MockedProvider>
    )
  }
}

describe('usePasskey', () => {
  it('registerPasskey calls GraphQL register begin + complete mutations', async () => {
    const mocks: MockedResponse[] = [
      {
        request: {
          query: PASSKEY_REGISTER_BEGIN,
        },
        variableMatcher: (vars) => vars.userId === 'user1',
        result: {
          data: {
            passkeyRegisterBegin: {
              optionsJson: {
                rp: { name: 'test', id: 'localhost' },
                user: { id: 'dXNlcjE', name: 'user1', displayName: 'user1' },
                challenge: 'Y2hhbGxlbmdl',
                pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
              },
            },
          },
        },
      },
      {
        request: {
          query: PASSKEY_REGISTER_COMPLETE,
        },
        variableMatcher: (vars) =>
          vars.userId === 'user1' && typeof vars.credential === 'object',
        result: {
          data: { passkeyRegisterComplete: true },
        },
      },
    ]

    const { result } = renderHook(() => usePasskey({ userId: 'user1' }), {
      wrapper: wrapper(mocks),
    })

    await act(async () => {
      await result.current.registerPasskey()
    })

    expect(navigator.credentials.create).toHaveBeenCalled()
    expect(result.current.error).toBeNull()
  })

  it('authenticatePasskey calls GraphQL auth begin + complete mutations', async () => {
    const mocks: MockedResponse[] = [
      {
        request: {
          query: PASSKEY_AUTH_BEGIN,
        },
        variableMatcher: (vars) => vars.userId === 'user1',
        result: {
          data: {
            passkeyAuthBegin: {
              optionsJson: {
                challenge: 'Y2hhbGxlbmdl',
                rpId: 'localhost',
                allowCredentials: [],
              },
            },
          },
        },
      },
      {
        request: {
          query: PASSKEY_AUTH_COMPLETE,
        },
        variableMatcher: (vars) =>
          vars.userId === 'user1' && typeof vars.credential === 'object',
        result: {
          data: {
            passkeyAuthComplete: {
              token: 'tok-456',
              userId: 'user1',
              email: 'user1',
              vaultUnlocked: false,
            },
          },
        },
      },
    ]

    const { result } = renderHook(() => usePasskey({ userId: 'user1' }), {
      wrapper: wrapper(mocks),
    })

    let authResult: { vault_unlocked?: boolean } | undefined
    await act(async () => {
      authResult = await result.current.authenticatePasskey()
    })

    expect(navigator.credentials.get).toHaveBeenCalled()
    expect(authResult?.vault_unlocked).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('authenticatePasskey returns vault_unlocked true from AuthPayload', async () => {
    const mocks: MockedResponse[] = [
      {
        request: {
          query: PASSKEY_AUTH_BEGIN,
        },
        variableMatcher: (vars) => vars.userId === 'user1',
        result: {
          data: {
            passkeyAuthBegin: {
              optionsJson: {
                challenge: 'Y2hhbGxlbmdl',
                rpId: 'localhost',
                allowCredentials: [],
              },
            },
          },
        },
      },
      {
        request: {
          query: PASSKEY_AUTH_COMPLETE,
        },
        variableMatcher: (vars) =>
          vars.userId === 'user1' && typeof vars.credential === 'object',
        result: {
          data: {
            passkeyAuthComplete: {
              token: 'tok-789',
              userId: 'user1',
              email: 'user1',
              vaultUnlocked: true,
            },
          },
        },
      },
    ]

    const { result } = renderHook(() => usePasskey({ userId: 'user1' }), {
      wrapper: wrapper(mocks),
    })

    let authResult: { vault_unlocked?: boolean } | undefined
    await act(async () => {
      authResult = await result.current.authenticatePasskey()
    })

    expect(authResult?.vault_unlocked).toBe(true)
  })

  it('hasPasskey returns true when credentials exist', async () => {
    const mocks: MockedResponse[] = [
      {
        request: {
          query: PASSKEY_AUTH_BEGIN,
        },
        variableMatcher: (vars) => vars.userId === 'user1',
        result: {
          data: {
            passkeyAuthBegin: {
              optionsJson: { challenge: 'Y2hhbGxlbmdl', rpId: 'localhost', allowCredentials: [{ type: 'public-key', id: 'abc' }] },
            },
          },
        },
      },
    ]

    const { result } = renderHook(() => usePasskey({ userId: 'user1' }), {
      wrapper: wrapper(mocks),
    })

    let has: boolean | undefined
    await act(async () => {
      has = await result.current.hasPasskey()
    })

    expect(has).toBe(true)
  })

  it('hasPasskey returns false on error', async () => {
    const mocks: MockedResponse[] = [
      {
        request: {
          query: PASSKEY_AUTH_BEGIN,
        },
        variableMatcher: (vars) => vars.userId === 'user1',
        error: new Error('No credentials registered'),
      },
    ]

    const { result } = renderHook(() => usePasskey({ userId: 'user1' }), {
      wrapper: wrapper(mocks),
    })

    let has: boolean | undefined
    await act(async () => {
      has = await result.current.hasPasskey()
    })

    expect(has).toBe(false)
  })
})
