// Hamburger Menu (drawer) component

import { Fragment } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import {
  TableCellsIcon,
  CodeBracketIcon,
  CircleStackIcon,
  DocumentTextIcon,
  ClockIcon,
  BookOpenIcon,
  ArrowUpTrayIcon,
  LinkIcon,
  ServerIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'

interface MenuItem {
  name: string
  icon: React.ComponentType<{ className?: string }>
  command: string
  description: string
}

const menuItems: MenuItem[] = [
  {
    name: 'Tables',
    icon: TableCellsIcon,
    command: '/tables',
    description: 'View session tables',
  },
  {
    name: 'Code',
    icon: CodeBracketIcon,
    command: '/code',
    description: 'Show generated code',
  },
  {
    name: 'Query',
    icon: CircleStackIcon,
    command: '/query',
    description: 'Run SQL query',
  },
  {
    name: 'Facts',
    icon: DocumentTextIcon,
    command: '/facts',
    description: 'View resolved facts',
  },
  {
    name: 'History',
    icon: ClockIcon,
    command: '/history',
    description: 'Session history',
  },
  {
    name: 'Learnings',
    icon: BookOpenIcon,
    command: '/learnings',
    description: 'View learnings',
  },
  {
    name: 'Upload File',
    icon: ArrowUpTrayIcon,
    command: '/add',
    description: 'Upload a file',
  },
  {
    name: 'Add File Ref',
    icon: LinkIcon,
    command: '/file',
    description: 'Add file reference',
  },
  {
    name: 'Database',
    icon: ServerIcon,
    command: '/database',
    description: 'Manage databases',
  },
]

interface HamburgerMenuProps {
  onCommand?: (command: string) => void
}

export function HamburgerMenu({ onCommand }: HamburgerMenuProps) {
  const { menuOpen, setMenuOpen, theme, setTheme } = useUIStore()

  const handleCommand = (command: string) => {
    onCommand?.(command)
    setMenuOpen(false)
  }

  return (
    <Transition.Root show={menuOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={setMenuOpen}>
        <Transition.Child
          as={Fragment}
          enter="ease-in-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in-out duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500/75 dark:bg-gray-900/75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 left-0 flex max-w-full pr-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-300"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-xs">
                  <div className="flex h-full flex-col overflow-y-auto bg-white dark:bg-gray-800 shadow-xl">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 dark:border-gray-700">
                      <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                        Commands
                      </Dialog.Title>
                      <button
                        onClick={() => setMenuOpen(false)}
                        className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                      >
                        <XMarkIcon className="w-5 h-5 text-gray-500" />
                      </button>
                    </div>

                    {/* Menu items */}
                    <nav className="flex-1 px-2 py-4 space-y-1">
                      {menuItems.map((item) => (
                        <button
                          key={item.command}
                          onClick={() => handleCommand(item.command)}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        >
                          <item.icon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                              {item.name}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                              {item.description}
                            </p>
                          </div>
                          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono">
                            {item.command}
                          </span>
                        </button>
                      ))}
                    </nav>

                    {/* Settings */}
                    <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Cog6ToothIcon className="w-5 h-5 text-gray-500" />
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            Theme
                          </span>
                        </div>
                        <select
                          value={theme}
                          onChange={(e) => setTheme(e.target.value as 'light' | 'dark' | 'system')}
                          className="text-sm bg-gray-100 dark:bg-gray-700 border-0 rounded-md px-2 py-1"
                        >
                          <option value="light">Light</option>
                          <option value="dark">Dark</option>
                          <option value="system">System</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  )
}