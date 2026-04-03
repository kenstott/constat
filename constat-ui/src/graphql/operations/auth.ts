// Copyright (c) 2025 Kenneth Stott
// Canary: 780c5a2d-ac70-4326-a7c0-127d31aec4d1
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

export const LOGIN_MUTATION = gql`
  mutation Login($email: String!, $password: String!) {
    login(email: $email, password: $password) {
      token
      userId
      email
    }
  }
`

export const REGISTER_MUTATION = gql`
  mutation Register($username: String!, $password: String!, $email: String) {
    register(username: $username, password: $password, email: $email) {
      token
      userId
      email
    }
  }
`

export const LOGOUT_MUTATION = gql`
  mutation Logout {
    logout
  }
`

export const CONFIG_QUERY = gql`
  query Config {
    config {
      databases
      apis
      documents
      llmProvider
      llmModel
      executionTimeout
      taskRouting
    }
  }
`

export const MY_PERMISSIONS_QUERY = gql`
  query MyPermissions {
    myPermissions {
      userId
      email
      admin
      persona
      domains
      databases
      documents
      apis
      visibility
      writes
      feedback
    }
  }
`

export const PASSKEY_REGISTER_BEGIN = gql`
  mutation PasskeyRegisterBegin($userId: String!) {
    passkeyRegisterBegin(userId: $userId) {
      optionsJson
    }
  }
`

export const PASSKEY_REGISTER_COMPLETE = gql`
  mutation PasskeyRegisterComplete($userId: String!, $credential: JSON!) {
    passkeyRegisterComplete(userId: $userId, credential: $credential)
  }
`

export const PASSKEY_AUTH_BEGIN = gql`
  mutation PasskeyAuthBegin($userId: String!) {
    passkeyAuthBegin(userId: $userId) {
      optionsJson
    }
  }
`

export const PASSKEY_AUTH_COMPLETE = gql`
  mutation PasskeyAuthComplete($userId: String!, $credential: JSON!, $prfOutput: String) {
    passkeyAuthComplete(userId: $userId, credential: $credential, prfOutput: $prfOutput) {
      token
      userId
      email
      vaultUnlocked
    }
  }
`

export const LOGIN_WITH_MICROSOFT_MUTATION = gql`
  mutation LoginWithMicrosoft($idToken: String!) {
    loginWithMicrosoft(idToken: $idToken) {
      token
      userId
      email
    }
  }
`

export const EMAIL_OAUTH_PROVIDERS_QUERY = gql`
  query EmailOAuthProviders {
    emailOAuthProviders {
      google
      microsoft
    }
  }
`
