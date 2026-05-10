// Copyright (c) 2025 Kenneth Stott
// Canary: aa5104bd-acbf-4249-9bfc-8e56fd7a51e1
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// WebAuthn passkey hook with PRF extension support

import { useState, useCallback } from 'react'
import { useMutation } from '@apollo/client'
import {
  PASSKEY_REGISTER_BEGIN,
  PASSKEY_REGISTER_COMPLETE,
  PASSKEY_AUTH_BEGIN,
  PASSKEY_AUTH_COMPLETE,
} from '@/graphql/operations/auth'

// PRF extension types (not yet in standard TypeScript DOM lib)
interface PRFValues {
  first: BufferSource
  second?: BufferSource
}

interface PRFExtensionInput {
  eval?: PRFValues
  evalByCredential?: Record<string, PRFValues>
}

interface PRFExtensionOutput {
  results?: { first?: ArrayBuffer; second?: ArrayBuffer }
}

interface ExtendedExtensionsInput extends AuthenticationExtensionsClientInputs {
  prf?: PRFExtensionInput
}

interface ExtendedExtensionsOutput extends AuthenticationExtensionsClientOutputs {
  prf?: PRFExtensionOutput
}

interface PasskeyOptions {
  userId: string
}

interface UsePasskeyReturn {
  loading: boolean
  error: string | null
  registerPasskey: () => Promise<void>
  authenticatePasskey: () => Promise<{ vault_unlocked?: boolean }>
  hasPasskey: () => Promise<boolean>
}

// PRF extension salt — fixed per application
const PRF_SALT = new TextEncoder().encode('constat-vault-prf-v1')

export function usePasskey({ userId }: PasskeyOptions): UsePasskeyReturn {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [registerBeginMutation] = useMutation(PASSKEY_REGISTER_BEGIN)
  const [registerCompleteMutation] = useMutation(PASSKEY_REGISTER_COMPLETE)
  const [authBeginMutation] = useMutation(PASSKEY_AUTH_BEGIN)
  const [authCompleteMutation] = useMutation(PASSKEY_AUTH_COMPLETE)

  const registerPasskey = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // 1. Get registration options from server
      const { data: beginData } = await registerBeginMutation({ variables: { userId } })
      const options = beginData.passkeyRegisterBegin.optionsJson as PublicKeyCredentialCreationOptionsJSON

      // 2. Create credential via WebAuthn
      const credential = await navigator.credentials.create({
        publicKey: parseCreationOptions(options),
      }) as PublicKeyCredential | null

      if (!credential) {
        throw new Error('Passkey registration was cancelled')
      }

      // 3. Send credential to server
      const response = credential.response as AuthenticatorAttestationResponse
      await registerCompleteMutation({
        variables: {
          userId,
          credential: serializeAttestationResponse(credential, response),
        },
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Passkey registration failed'
      setError(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [userId, registerBeginMutation, registerCompleteMutation])

  const authenticatePasskey = useCallback(async (): Promise<{ vault_unlocked?: boolean }> => {
    setLoading(true)
    setError(null)
    try {
      // 1. Get authentication options from server
      const { data: beginData } = await authBeginMutation({ variables: { userId } })
      const options = beginData.passkeyAuthBegin.optionsJson as PublicKeyCredentialRequestOptionsJSON

      // 2. Authenticate via WebAuthn with PRF extension
      const requestOptions = parseRequestOptions(options)
      const publicKeyWithPrf = {
        ...requestOptions,
        extensions: {
          ...requestOptions.extensions,
          prf: { eval: { first: PRF_SALT } },
        } as ExtendedExtensionsInput,
      }
      const credential = await navigator.credentials.get({
        publicKey: publicKeyWithPrf,
      }) as PublicKeyCredential | null

      if (!credential) {
        throw new Error('Passkey authentication was cancelled')
      }

      // 3. Extract PRF output if available
      const extensions = credential.getClientExtensionResults() as ExtendedExtensionsOutput
      const prfOutput = extensions.prf?.results?.first
      const prfB64 = prfOutput ? arrayBufferToBase64url(prfOutput) : undefined

      // 4. Send credential to server
      const response = credential.response as AuthenticatorAssertionResponse
      const { data: completeData } = await authCompleteMutation({
        variables: {
          userId,
          credential: serializeAssertionResponse(credential, response),
          prfOutput: prfB64,
        },
      })

      return { vault_unlocked: completeData.passkeyAuthComplete.vaultUnlocked }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Passkey authentication failed'
      setError(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [userId, authBeginMutation, authCompleteMutation])

  const hasPasskey = useCallback(async (): Promise<boolean> => {
    try {
      await authBeginMutation({ variables: { userId } })
      return true
    } catch {
      return false
    }
  }, [userId, authBeginMutation])

  return { loading, error, registerPasskey, authenticatePasskey, hasPasskey }
}

// --- JSON type interfaces matching WebAuthn API ---

interface PublicKeyCredentialCreationOptionsJSON {
  rp: { name: string; id?: string }
  user: { id: string; name: string; displayName: string }
  challenge: string
  pubKeyCredParams: Array<{ type: string; alg: number }>
  timeout?: number
  excludeCredentials?: Array<{ type: string; id: string; transports?: string[] }>
  authenticatorSelection?: {
    authenticatorAttachment?: string
    residentKey?: string
    requireResidentKey?: boolean
    userVerification?: string
  }
  attestation?: string
}

interface PublicKeyCredentialRequestOptionsJSON {
  challenge: string
  timeout?: number
  rpId?: string
  allowCredentials?: Array<{ type: string; id: string; transports?: string[] }>
  userVerification?: string
}

// --- Conversion helpers ---

function base64urlToArrayBuffer(b64url: string): ArrayBuffer {
  const b64 = b64url.replace(/-/g, '+').replace(/_/g, '/')
  const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4))
  const binary = atob(b64 + pad)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes.buffer
}

function arrayBufferToBase64url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function parseCreationOptions(json: PublicKeyCredentialCreationOptionsJSON): PublicKeyCredentialCreationOptions {
  return {
    rp: json.rp,
    user: {
      id: base64urlToArrayBuffer(json.user.id),
      name: json.user.name,
      displayName: json.user.displayName,
    },
    challenge: base64urlToArrayBuffer(json.challenge),
    pubKeyCredParams: json.pubKeyCredParams.map(p => ({
      type: p.type as PublicKeyCredentialType,
      alg: p.alg,
    })),
    timeout: json.timeout,
    excludeCredentials: json.excludeCredentials?.map(c => ({
      type: c.type as PublicKeyCredentialType,
      id: base64urlToArrayBuffer(c.id),
      transports: c.transports as AuthenticatorTransport[] | undefined,
    })),
    authenticatorSelection: json.authenticatorSelection as AuthenticatorSelectionCriteria | undefined,
    attestation: json.attestation as AttestationConveyancePreference | undefined,
  }
}

function parseRequestOptions(json: PublicKeyCredentialRequestOptionsJSON): PublicKeyCredentialRequestOptions {
  return {
    challenge: base64urlToArrayBuffer(json.challenge),
    timeout: json.timeout,
    rpId: json.rpId,
    allowCredentials: json.allowCredentials?.map(c => ({
      type: c.type as PublicKeyCredentialType,
      id: base64urlToArrayBuffer(c.id),
      transports: c.transports as AuthenticatorTransport[] | undefined,
    })),
    userVerification: json.userVerification as UserVerificationRequirement | undefined,
  }
}

function serializeAttestationResponse(
  credential: PublicKeyCredential,
  response: AuthenticatorAttestationResponse
): Record<string, unknown> {
  return {
    id: credential.id,
    rawId: arrayBufferToBase64url(credential.rawId),
    type: credential.type,
    response: {
      attestationObject: arrayBufferToBase64url(response.attestationObject),
      clientDataJSON: arrayBufferToBase64url(response.clientDataJSON),
    },
  }
}

function serializeAssertionResponse(
  credential: PublicKeyCredential,
  response: AuthenticatorAssertionResponse
): Record<string, unknown> {
  return {
    id: credential.id,
    rawId: arrayBufferToBase64url(credential.rawId),
    type: credential.type,
    response: {
      authenticatorData: arrayBufferToBase64url(response.authenticatorData),
      clientDataJSON: arrayBufferToBase64url(response.clientDataJSON),
      signature: arrayBufferToBase64url(response.signature),
      userHandle: response.userHandle ? arrayBufferToBase64url(response.userHandle) : null,
    },
  }
}
