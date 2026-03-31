// Copyright (c) 2025 Kenneth Stott
// Canary: 74ffcc56-36c5-49b8-a897-b1f03231a6ed
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

/** Shimmer skeleton lines for loading states in panels. */
export function SkeletonLoader({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2 py-1">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="skeleton-line h-3"
          style={{ width: `${70 + Math.sin(i * 1.5) * 20}%` }}
        />
      ))}
    </div>
  )
}
