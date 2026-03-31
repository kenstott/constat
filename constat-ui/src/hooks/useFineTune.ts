// Copyright (c) 2025 Kenneth Stott
// Canary: 616029d1-3abf-456f-9b35-f5dbb9a0dafa
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery, useMutation } from '@apollo/client'
import {
  FINE_TUNE_JOBS_QUERY,
  FINE_TUNE_JOB_QUERY,
  FINE_TUNE_PROVIDERS_QUERY,
  START_FINE_TUNE_JOB,
  CANCEL_FINE_TUNE_JOB,
  DELETE_FINE_TUNE_JOB,
  toFineTuneJob,
  toFineTuneProvider,
} from '@/graphql/operations/fine-tune'

export function useFineTuneJobs(status?: string, domain?: string) {
  const { data, loading, error, refetch } = useQuery(FINE_TUNE_JOBS_QUERY, {
    variables: { status, domain },
  })
  const jobs = (data?.fineTuneJobs ?? []).map(toFineTuneJob)
  return { jobs, loading, error, refetch }
}

export function useFineTuneJob(modelId: string) {
  const { data, loading, error, refetch } = useQuery(FINE_TUNE_JOB_QUERY, {
    variables: { modelId },
    skip: !modelId,
  })
  const job = data?.fineTuneJob ? toFineTuneJob(data.fineTuneJob) : null
  return { job, loading, error, refetch }
}

export function useFineTuneProviders() {
  const { data, loading, error } = useQuery(FINE_TUNE_PROVIDERS_QUERY)
  const providers = (data?.fineTuneProviders ?? []).map(toFineTuneProvider)
  return { providers, loading, error }
}

export function useFineTuneMutations() {
  const [startMut] = useMutation(START_FINE_TUNE_JOB, {
    refetchQueries: ['FineTuneJobs'],
  })
  const [cancelMut] = useMutation(CANCEL_FINE_TUNE_JOB, {
    refetchQueries: ['FineTuneJobs'],
  })
  const [deleteMut] = useMutation(DELETE_FINE_TUNE_JOB, {
    refetchQueries: ['FineTuneJobs'],
  })

  return {
    startFineTuneJob: (input: {
      name: string
      provider: string
      baseModel: string
      taskTypes: string[]
      domain?: string
    }) => startMut({ variables: { input } }),
    cancelFineTuneJob: (modelId: string) =>
      cancelMut({ variables: { modelId } }),
    deleteFineTuneJob: (modelId: string) =>
      deleteMut({ variables: { modelId } }),
  }
}
