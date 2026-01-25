// Static command definitions for client-side autocomplete
// Derived from HELP_COMMANDS in constat/commands/help.py

export interface CommandDef {
  command: string
  description: string
  category: string
  argType?: 'table' | 'entity' | 'scope' | 'text'
}

export const COMMANDS: CommandDef[] = [
  // Data Exploration
  { command: '/help', description: 'Show help message', category: 'Data Exploration' },
  { command: '/h', description: 'Show help message', category: 'Data Exploration' },
  { command: '/tables', description: 'List available tables', category: 'Data Exploration' },
  { command: '/show', description: 'Show table contents', category: 'Data Exploration', argType: 'table' },
  { command: '/export', description: 'Export table to CSV or XLSX', category: 'Data Exploration', argType: 'table' },
  { command: '/query', description: 'Run SQL query on datastore', category: 'Data Exploration', argType: 'table' },
  { command: '/code', description: 'Show generated code (all or specific step)', category: 'Data Exploration' },
  { command: '/artifacts', description: 'Show artifacts (use "all" for intermediate)', category: 'Data Exploration' },

  // Session Management
  { command: '/state', description: 'Show session state', category: 'Session' },
  { command: '/reset', description: 'Clear session state and start fresh', category: 'Session' },
  { command: '/redo', description: 'Retry last query (with optional modifications)', category: 'Session' },
  { command: '/update', description: 'Refresh metadata and rebuild cache', category: 'Session' },
  { command: '/refresh', description: 'Refresh metadata and rebuild cache', category: 'Session' },
  { command: '/context', description: 'Show context size and token usage', category: 'Session' },
  { command: '/compact', description: 'Compact context to reduce token usage', category: 'Session' },

  // Facts & Memory
  { command: '/facts', description: 'Show cached facts from this session', category: 'Facts' },
  { command: '/remember', description: 'Persist a session fact', category: 'Facts', argType: 'text' },
  { command: '/forget', description: 'Forget a remembered fact', category: 'Facts', argType: 'text' },

  // Plans & History
  { command: '/save', description: 'Save current plan for replay', category: 'Plans', argType: 'text' },
  { command: '/share', description: 'Save plan as shared (all users)', category: 'Plans', argType: 'text' },
  { command: '/plans', description: 'List saved plans', category: 'Plans' },
  { command: '/replay', description: 'Replay a saved plan', category: 'Plans', argType: 'text' },
  { command: '/history', description: 'List recent sessions', category: 'Plans' },
  { command: '/sessions', description: 'List recent sessions', category: 'Plans' },
  { command: '/resume', description: 'Resume a previous session', category: 'Plans', argType: 'text' },

  // Data Sources
  { command: '/databases', description: 'List configured databases', category: 'Data Sources' },
  { command: '/database', description: 'Add a database to this session', category: 'Data Sources', argType: 'text' },
  { command: '/apis', description: 'List configured APIs', category: 'Data Sources' },
  { command: '/api', description: 'Add an API to this session', category: 'Data Sources', argType: 'text' },
  { command: '/documents', description: 'List all documents', category: 'Data Sources' },
  { command: '/docs', description: 'List all documents', category: 'Data Sources' },
  { command: '/files', description: 'List all data files', category: 'Data Sources' },
  { command: '/doc', description: 'Add a document to this session', category: 'Data Sources', argType: 'text' },

  // Preferences
  { command: '/verbose', description: 'Toggle verbose mode', category: 'Preferences' },
  { command: '/raw', description: 'Toggle raw output display', category: 'Preferences' },
  { command: '/insights', description: 'Toggle insight synthesis', category: 'Preferences' },
  { command: '/preferences', description: 'Show current preferences', category: 'Preferences' },
  { command: '/user', description: 'Show or set current user', category: 'Preferences', argType: 'text' },

  // Analysis
  { command: '/discover', description: 'Search data sources', category: 'Analysis', argType: 'scope' },
  { command: '/summarize', description: 'Summarize plan|session|facts|<table>', category: 'Analysis', argType: 'entity' },
  { command: '/prove', description: 'Verify conversation claims with auditable proof', category: 'Analysis' },
  { command: '/correct', description: 'Record a correction for future reference', category: 'Analysis', argType: 'text' },
  { command: '/learnings', description: 'Show learnings and rules', category: 'Analysis' },
  { command: '/compact-learnings', description: 'Promote similar learnings into rules', category: 'Analysis' },

  // Exit
  { command: '/quit', description: 'Exit the session', category: 'Exit' },
  { command: '/q', description: 'Exit the session', category: 'Exit' },
]

// Discovery scopes for /discover command
export const DISCOVER_SCOPES = ['database', 'api', 'document']

// Summarize targets
export const SUMMARIZE_TARGETS = ['plan', 'session', 'facts']

export function getCommandByName(name: string): CommandDef | undefined {
  return COMMANDS.find((cmd) => cmd.command === name)
}

export function filterCommands(prefix: string): CommandDef[] {
  const lowerPrefix = prefix.toLowerCase()
  return COMMANDS.filter((cmd) => cmd.command.toLowerCase().startsWith(lowerPrefix))
}
