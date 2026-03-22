import { useToastStore } from '@/store/toastStore'
import { CheckCircleIcon, ExclamationCircleIcon, InformationCircleIcon, XMarkIcon } from '@heroicons/react/24/outline'

const icons = {
  success: CheckCircleIcon,
  error: ExclamationCircleIcon,
  info: InformationCircleIcon,
}

const colors = {
  success: 'bg-green-50 dark:bg-green-900/30 text-green-800 dark:text-green-200 border-green-200 dark:border-green-700',
  error: 'bg-red-50 dark:bg-red-900/30 text-red-800 dark:text-red-200 border-red-200 dark:border-red-700',
  info: 'bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-700',
}

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts)
  const removeToast = useToastStore((s) => s.removeToast)

  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => {
        const Icon = icons[toast.type]
        return (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-center gap-2 px-4 py-2.5 rounded-lg border shadow-lg text-sm font-medium animate-slide-in ${colors[toast.type]}`}
          >
            <Icon className="h-5 w-5 flex-shrink-0" />
            <span>{toast.message}</span>
            <button onClick={() => removeToast(toast.id)} className="ml-2 opacity-60 hover:opacity-100">
              <XMarkIcon className="h-4 w-4" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
