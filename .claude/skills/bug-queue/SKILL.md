---
name: bug-queue
description: Bug queue database operations and analytics. Auto-triggers when investigating test failures, triaging bugs, or analyzing failure patterns.
---

# Bug Queue

Persistent DuckDB-based bug tracker at `.constat/bugs.duckdb`. Populated automatically by the `--bug-queue` pytest plugin and managed via SQL.

## Running Tests with Bug Queue

```bash
python -m pytest tests/ -x -q --bug-queue
```

This captures failures into the `bug_queue` table with error details, stack traces, and file locations.

## Querying Bugs

### Open bugs by priority
```sql
SELECT id, test_name, error_type, error_message, file_path, priority, created_at
FROM bug_queue
WHERE status = 'open'
ORDER BY priority, created_at;
```

### Bugs for a specific module
```sql
SELECT id, test_name, error_message
FROM bug_queue
WHERE file_path LIKE '%/session/%' AND status = 'open';
```

### Recent failures
```sql
SELECT id, test_name, error_type, created_at
FROM bug_queue
WHERE status = 'open'
ORDER BY created_at DESC
LIMIT 10;
```

## Claiming and Resolving

### Claim a bug
```sql
UPDATE bug_queue
SET status = 'in_progress', assigned_to = '<agent-name>'
WHERE id = '<bug_id>';
```

### Resolve a bug
```sql
UPDATE bug_queue
SET status = 'resolved',
    resolution = '<description of fix>',
    resolution_commit = '<commit hash>'
WHERE id = '<bug_id>';
```

### Reopen a bug
```sql
UPDATE bug_queue
SET status = 'open', assigned_to = NULL, resolution = NULL, resolution_commit = NULL
WHERE id = '<bug_id>';
```

## Analytics

### Failure hotspots — files with most bugs
```sql
SELECT file_path, COUNT(*) AS bug_count
FROM bug_queue
WHERE status = 'open'
GROUP BY file_path
ORDER BY bug_count DESC
LIMIT 10;
```

### Flaky tests — tests that have been resolved and reopened
```sql
SELECT test_name, COUNT(*) AS occurrence_count
FROM bug_queue
GROUP BY test_name
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;
```

### Error type distribution
```sql
SELECT error_type, COUNT(*) AS count
FROM bug_queue
WHERE status = 'open'
GROUP BY error_type
ORDER BY count DESC;
```

### Agent effectiveness — resolved bugs per agent
```sql
SELECT assigned_to, COUNT(*) AS resolved_count
FROM bug_queue
WHERE status = 'resolved'
GROUP BY assigned_to
ORDER BY resolved_count DESC;
```

### Resolution rate
```sql
SELECT
    status,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM bug_queue
GROUP BY status;
```

## Database Location

- Path: `.constat/bugs.duckdb`
- Created automatically on first `--bug-queue` test run
- Class: `constat.testing.bug_queue.BugQueue`
- Plugin: `constat.testing.bug_queue_plugin`
