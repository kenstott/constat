// Copyright (c) 2025 Kenneth Stott
// Canary: 8d6bb15b-26f6-419a-9871-1f70a0ec2f1e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

export const URL_SCHEMES = ['http://', 'https://', 'ftp://', 'sftp://', 's3://', 's3a://', 'file://']

export function detectScheme(uri: string): string | null {
  const lower = uri.toLowerCase()
  return URL_SCHEMES.find(s => lower.startsWith(s)) ?? null
}

export function isValidUri(uri: string): boolean {
  const trimmed = uri.trim()
  if (!trimmed) return false
  const scheme = detectScheme(trimmed)
  if (scheme === 'http://' || scheme === 'https://') {
    try { new URL(trimmed); return true } catch { return false }
  }
  if (scheme === 's3://' || scheme === 's3a://') {
    return trimmed.replace(/^s3a?:\/\//i, '').split('/')[0].length > 0
  }
  if (scheme === 'ftp://' || scheme === 'sftp://') {
    try { new URL(trimmed); return true } catch { return false }
  }
  if (scheme === 'file://') {
    return trimmed.length > 'file://'.length
  }
  return trimmed.length > 0
}
