// Copyright (c) 2025 Kenneth Stott
// Canary: 8244f92f-2d7c-40a8-bc0b-5f29c62e0e3b
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const FINE_TUNE_JOBS_QUERY = gql`
  query FineTuneJobs($status: String, $domain: String) {
    fineTuneJobs(status: $status, domain: $domain) {
      id
      name
      provider
      baseModel
      fineTunedModelId
      taskTypes
      domain
      status
      created
      exemplarCount
      metrics
      trainingDataPath
    }
  }
`

export const FINE_TUNE_JOB_QUERY = gql`
  query FineTuneJob($modelId: String!) {
    fineTuneJob(modelId: $modelId) {
      id
      name
      provider
      baseModel
      fineTunedModelId
      taskTypes
      domain
      status
      created
      exemplarCount
      metrics
      trainingDataPath
    }
  }
`

export const FINE_TUNE_PROVIDERS_QUERY = gql`
  query FineTuneProviders {
    fineTuneProviders {
      name
      models
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const START_FINE_TUNE_JOB = gql`
  mutation StartFineTuneJob($input: StartFineTuneInput!) {
    startFineTuneJob(input: $input) {
      id
      name
      provider
      baseModel
      fineTunedModelId
      taskTypes
      domain
      status
      created
      exemplarCount
      metrics
      trainingDataPath
    }
  }
`

export const CANCEL_FINE_TUNE_JOB = gql`
  mutation CancelFineTuneJob($modelId: String!) {
    cancelFineTuneJob(modelId: $modelId) {
      id
      name
      provider
      baseModel
      status
      created
      exemplarCount
    }
  }
`

export const DELETE_FINE_TUNE_JOB = gql`
  mutation DeleteFineTuneJob($modelId: String!) {
    deleteFineTuneJob(modelId: $modelId) {
      status
      id
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toFineTuneJob(gql: any) {
  return {
    id: gql.id,
    name: gql.name,
    provider: gql.provider,
    base_model: gql.baseModel,
    fine_tuned_model_id: gql.fineTunedModelId ?? null,
    task_types: gql.taskTypes ?? [],
    domain: gql.domain ?? null,
    status: gql.status,
    created: gql.created,
    exemplar_count: gql.exemplarCount ?? 0,
    metrics: gql.metrics ?? null,
    training_data_path: gql.trainingDataPath ?? null,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toFineTuneProvider(gql: any) {
  return {
    name: gql.name,
    models: gql.models ?? [],
  }
}
