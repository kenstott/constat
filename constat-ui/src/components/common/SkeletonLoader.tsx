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
