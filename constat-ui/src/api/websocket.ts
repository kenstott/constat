// WebSocket Manager with reconnection logic

import type { WSMessage, WSEvent, CompletionItem } from '@/types/api'
import { getCachedEntry } from '@/store/entityCache'

export type WSEventHandler = (event: WSEvent) => void
export type WSStatusHandler = (connected: boolean) => void
export type AutocompleteCallback = (items: CompletionItem[]) => void

interface WSManagerOptions {
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

export class WebSocketManager {
  private ws: WebSocket | null = null
  private sessionId: string | null = null
  private eventHandlers: Set<WSEventHandler> = new Set()
  private statusHandlers: Set<WSStatusHandler> = new Set()
  private autocompleteCallbacks: Map<string, AutocompleteCallback> = new Map()
  private autocompleteRequestId = 0
  private reconnectAttempts = 0
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null
  private lastHeartbeatTime: string | null = null
  private options: Required<WSManagerOptions>

  constructor(options: WSManagerOptions = {}) {
    this.options = {
      reconnectInterval: options.reconnectInterval ?? 3000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
    }
  }

  connect(sessionId: string): void {
    if (this.ws && this.sessionId === sessionId) {
      return // Already connected to this session
    }

    this.disconnect()
    this.sessionId = sessionId

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/api/sessions/${sessionId}/ws`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this.reconnectAttempts = 0
      this.notifyStatus(true)
      getCachedEntry(sessionId).then((entry) => {
        this.send('entity_seed', { version: entry?.version ?? null })
      })
      this.startHeartbeat()
    }

    this.ws.onclose = () => {
      this.notifyStatus(false)
      this.scheduleReconnect()
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    this.ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)
        if (message.type === 'event') {
          const wsEvent = message.payload as unknown as WSEvent
          // Handle autocomplete responses
          if (wsEvent.event_type === 'autocomplete_response') {
            const data = wsEvent.data as { request_id?: string; items?: CompletionItem[] }
            const requestId = data.request_id
            const items = data.items || []
            if (requestId && this.autocompleteCallbacks.has(requestId)) {
              const callback = this.autocompleteCallbacks.get(requestId)!
              this.autocompleteCallbacks.delete(requestId)
              callback(items)
            }
          } else {
            this.notifyEvent(wsEvent)
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }
  }

  disconnect(): void {
    this.stopHeartbeat()

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    this.sessionId = null
    this.reconnectAttempts = 0
    this.lastHeartbeatTime = null
    // Clear all handlers to prevent duplicates on reconnect
    this.eventHandlers.clear()
    this.statusHandlers.clear()
    this.autocompleteCallbacks.clear()
  }

  private scheduleReconnect(): void {
    if (!this.sessionId) return
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.warn('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.options.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1)

    this.reconnectTimeout = setTimeout(() => {
      if (this.sessionId) {
        this.connect(this.sessionId)
      }
    }, delay)
  }

  send(action: string, data?: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected')
      return
    }

    this.ws.send(JSON.stringify({ action, data }))
  }

  approve(): void {
    this.send('approve')
  }

  reject(feedback: string): void {
    this.send('reject', { feedback })
  }

  cancel(): void {
    this.send('cancel')
  }

  replanFrom(stepNumber: number, mode: 'edit' | 'delete' | 'redo', editedGoal?: string): void {
    this.send('replan_from', { step_number: stepNumber, mode, edited_goal: editedGoal })
  }

  editObjective(objectiveIndex: number, newText: string): void {
    this.send('edit_objective', { objective_index: objectiveIndex, new_text: newText })
  }

  deleteObjective(objectiveIndex: number): void {
    this.send('delete_objective', { objective_index: objectiveIndex })
  }

  requestAutocomplete(
    context: 'table' | 'column' | 'entity',
    prefix: string,
    callback: AutocompleteCallback,
    parent?: string
  ): void {
    if (!this.isConnected) {
      callback([])
      return
    }

    const requestId = `ac_${++this.autocompleteRequestId}`
    this.autocompleteCallbacks.set(requestId, callback)

    this.send('autocomplete', {
      context,
      prefix,
      parent,
      request_id: requestId,
    })

    // Timeout after 5 seconds
    setTimeout(() => {
      if (this.autocompleteCallbacks.has(requestId)) {
        this.autocompleteCallbacks.delete(requestId)
        callback([])
      }
    }, 5000)
  }

  private startHeartbeat(): void {
    this.stopHeartbeat()
    // Immediate first heartbeat with since=null
    this.send('heartbeat', { since: this.lastHeartbeatTime })
    this.heartbeatInterval = setInterval(() => {
      this.send('heartbeat', { since: this.lastHeartbeatTime })
    }, 30_000)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  setLastHeartbeatTime(time: string): void {
    this.lastHeartbeatTime = time
  }

  onEvent(handler: WSEventHandler): () => void {
    this.eventHandlers.add(handler)
    return () => this.eventHandlers.delete(handler)
  }

  onStatus(handler: WSStatusHandler): () => void {
    this.statusHandlers.add(handler)
    return () => this.statusHandlers.delete(handler)
  }

  private notifyEvent(event: WSEvent): void {
    this.eventHandlers.forEach((handler) => {
      try {
        handler(event)
      } catch (error) {
        console.error('Event handler error:', error)
      }
    })
  }

  private notifyStatus(connected: boolean): void {
    this.statusHandlers.forEach((handler) => {
      try {
        handler(connected)
      } catch (error) {
        console.error('Status handler error:', error)
      }
    })
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Singleton instance
export const wsManager = new WebSocketManager()