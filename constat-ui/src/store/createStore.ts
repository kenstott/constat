// Copyright (c) 2025 Kenneth Stott
// Canary: 13a0c2e3-4054-4b89-8d16-2a42f40d9887
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useSyncExternalStore } from 'react'

type SetState<T> = (partial: Partial<T> | ((state: T) => Partial<T>)) => void
type GetState<T> = () => T

type StoreApi<T> = {
  getState: GetState<T>
  setState: SetState<T>
  subscribe: (listener: () => void) => () => void
}

type UseStore<T> = {
  (): T
  <S>(selector: (state: T) => S): S
  getState: GetState<T>
  setState: SetState<T>
  subscribe: StoreApi<T>['subscribe']
}

export function create<T>(createFn: (set: SetState<T>, get: GetState<T>) => T): UseStore<T> {
  let state: T
  const listeners = new Set<() => void>()

  const getState: GetState<T> = () => state

  const setState: SetState<T> = (partial) => {
    const p = typeof partial === 'function' ? (partial as (s: T) => Partial<T>)(state) : partial
    state = { ...state, ...p }
    listeners.forEach((l) => l())
  }

  const subscribe = (listener: () => void) => {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  }

  state = createFn(setState, getState)

  function useStore(): T
  function useStore<S>(selector: (state: T) => S): S
  function useStore<S>(selector?: (state: T) => S): T | S {
    const snap = selector
      ? () => selector(getState())
      : (getState as () => T | S)
    return useSyncExternalStore(subscribe, snap)
  }

  useStore.getState = getState
  useStore.setState = setState
  useStore.subscribe = subscribe

  return useStore as unknown as UseStore<T>
}
