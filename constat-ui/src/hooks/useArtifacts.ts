// Copyright (c) 2025 Kenneth Stott
// Canary: 0223b8c4-5482-4bce-8aa9-24f7ad0428a3
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery, useMutation } from '@apollo/client'
import { useSessionContext } from '@/contexts/SessionContext'
import {
  ARTIFACTS_QUERY,
  ARTIFACT_QUERY,
  ARTIFACT_VERSIONS_QUERY,
  DELETE_ARTIFACT,
  TOGGLE_ARTIFACT_STAR,
  toArtifact,
  toArtifactContent,
  toArtifactVersions,
} from '@/graphql/operations/data'

export function useArtifacts() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(ARTIFACTS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const artifacts = (data?.artifacts?.artifacts ?? []).map(toArtifact)
  return { artifacts, total: data?.artifacts?.total ?? 0, loading, error }
}

export function useArtifact(id: number) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(ARTIFACT_QUERY, {
    variables: { sessionId: sessionId!, artifactId: id },
    skip: !sessionId || !id,
  })
  return {
    artifact: data?.artifact ? toArtifactContent(data.artifact) : null,
    loading,
    error,
  }
}

export function useArtifactVersions(id: number) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(ARTIFACT_VERSIONS_QUERY, {
    variables: { sessionId: sessionId!, artifactId: id },
    skip: !sessionId || !id,
  })
  return {
    versions: data?.artifactVersions ? toArtifactVersions(data.artifactVersions) : null,
    loading,
    error,
  }
}

export function useArtifactMutations() {
  const { sessionId } = useSessionContext()
  const [deleteArtifactMut] = useMutation(DELETE_ARTIFACT, {
    refetchQueries: ['Artifacts'],
  })
  const [toggleStarMut] = useMutation(TOGGLE_ARTIFACT_STAR)

  return {
    deleteArtifact: (id: number) =>
      deleteArtifactMut({ variables: { sessionId: sessionId!, artifactId: id } }),
    toggleStar: (id: number) =>
      toggleStarMut({ variables: { sessionId: sessionId!, artifactId: id } }),
  }
}
