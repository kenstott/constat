// Vera logo — stylized V with nodes at endpoints

export function VeraIcon({ className = 'w-5 h-5' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* V shape — optically centered in 24x24 viewBox */}
      <path
        d="M4 6L12 20L20 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Nodes at endpoints */}
      <circle cx="4" cy="6" r="2.5" fill="currentColor" />
      <circle cx="20" cy="6" r="2.5" fill="currentColor" />
      <circle cx="12" cy="20" r="2.5" fill="currentColor" />
    </svg>
  )
}
