# Calendar Service Document Source

## Goal

Add calendar services (Google Calendar, Microsoft Outlook/Exchange) as a document data source. Each calendar event becomes a document. Recurring events are expanded to individual instances within the sync window. Event attachments are extracted as child documents and routed through the existing vectorization pipeline.

## MCP-First Strategy

**Default approach**: Use existing MCP servers for Google Calendar and Microsoft Outlook rather than building custom integrations. The [MCP ecosystem](https://modelcontextprotocol.io/) includes calendar connectors that expose events as MCP resources and calendar operations as MCP tools.

### MCP config (preferred)

```yaml
documents:
  team-calendar:
    type: mcp
    url: https://mcp-gcal.example.com/sse      # Google Calendar MCP server
    auth:
      method: bearer
      token_ref: gcal-mcp-token
    description: "Team shared calendar"

  outlook-calendar:
    type: mcp
    url: https://mcp-outlook.example.com/sse    # Outlook MCP server
    auth:
      method: bearer
      token_ref: outlook-mcp-token
    description: "Outlook work calendar"
```

MCP resources flow through the standard pipeline: `resources/read` → event content → chunk → embed → DuckDB vector store. MCP tools (e.g., `search_events`, `create_event`) are registered as API operations for the query engine. See `mcp.md` for full MCP client architecture.

### When to fall back to custom `CalendarFetcher`

Build the custom implementation below only if MCP servers cannot handle:

| Gap | Why MCP may fall short | Custom solution |
|---|---|---|
| Recurring event expansion | MCP server may return master events only, not instances | `singleEvents=true` (Google) / `calendarView` (MS Graph) |
| Event attachment extraction | MCP resources may not include attachments as separate items | `_fetch_ms_attachments()` / Google attachment download |
| Incremental sync | MCP has no `updatedMin` equivalent for delta queries | `updatedMin` (Google) / `$filter=lastModifiedDateTime ge` (MS) |
| Structured event metadata | MCP text content may lose attendees, location, recurrence info | `_render_event()` with full field extraction |
| Multi-calendar support | MCP server may only expose one calendar per connection | `calendars: [primary, holidays@group...]` config |

**Decision process**: Try MCP first. If the MCP server expands recurring events, includes attachments, and preserves structured metadata, skip everything below. If not, implement only the missing pieces.

---

The rest of this document specifies the custom fallback implementation.

## Addressing Scheme

```
<data_source>:<event_id>                           # event body
<data_source>:<event_id>:<attachment_filename>      # event attachment
```

Examples:
```
team-calendar:evt_20260315_standup_abc              # daily standup event
team-calendar:evt_20260320_review_def:agenda.pdf    # attachment on event
team-calendar:evt_20260320_review_def:deck.pptx     # second attachment
```

Recurring events use the instance date in the ID to distinguish occurrences:
```
team-calendar:evt_20260315_standup_abc              # Monday instance
team-calendar:evt_20260322_standup_abc              # following Monday
```

## Configuration

### In domain YAML or root config:

```yaml
documents:
  team-calendar:
    type: calendar
    provider: google                     # "google" | "microsoft"
    calendar_id: primary                 # Google: calendar ID; Microsoft: calendar ID or "me"
    description: "Team shared calendar"

    # OAuth2 credentials (reuses existing oauth2_* fields from DocumentConfig)
    oauth2_client_id: ${GOOGLE_CLIENT_ID}
    oauth2_client_secret: ${GOOGLE_CLIENT_SECRET}
    oauth2_scopes:
      - "https://www.googleapis.com/auth/calendar.readonly"

    # Calendar-specific options
    since: "2026-01-01"                  # only events after this date (default: 90 days ago)
    until: "2026-12-31"                  # only events before this date (default: 90 days ahead)
    max_events: 1000                     # cap per sync
    expand_recurring: true               # expand recurring events to instances (default: true)
    include_declined: false              # include events the user declined (default: false)
    include_cancelled: false             # include cancelled events (default: false)
    extract_attachments: true            # extract event attachments (default: true)
    extract_body: true                   # index event description/notes (default: true)
    calendars:                           # multiple calendars from same account (optional)
      - primary
      - company-holidays@group.calendar.google.com

  outlook-calendar:
    type: calendar
    provider: microsoft
    calendar_id: me                      # "me" = default calendar
    description: "Outlook work calendar"

    oauth2_client_id: ${AZURE_CLIENT_ID}
    oauth2_client_secret: ${AZURE_CLIENT_SECRET}
    oauth2_tenant_id: ${AZURE_TENANT_ID}
    oauth2_scopes:
      - "https://graph.microsoft.com/Calendars.Read"
```

### `DocumentConfig` additions

```python
class DocumentConfig(BaseModel):
    ...
    # Calendar fields
    provider: Optional[str] = None            # "google" | "microsoft"
    calendar_id: str = "primary"              # calendar identifier
    until: Optional[str] = None               # events before this date
    max_events: int = 1000
    expand_recurring: bool = True
    include_declined: bool = False
    include_cancelled: bool = False
    extract_body: bool = True
    calendars: list[str] = Field(default_factory=list)  # multi-calendar
```

Reuses existing fields: `since`, `extract_attachments`, `oauth2_*`, `description`, `tags`.

## Pipeline Overview

```
OAuth2 authentication
  → fetch event list (paginated)
  → for each event:
      ├─ extract metadata (title, start, end, location, organizer, attendees, status)
      ├─ extract body (description/notes → markdown)
      ├─ render as document text
      ├─ _chunk_document → encode → add_chunks    (existing pipeline)
      │
      └─ for each attachment:
            ├─ download attachment bytes
            ├─ detect type via _mime.detect_type_from_source()
            ├─ if document → _extract_content() → _chunk_document → encode → add_chunks
            ├─ if image → _extract_image() → image pipeline
            └─ address: <source>:<event_id>:<filename>
```

## Implementation

### New file: `constat/discovery/doc_tools/_calendar.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class CalendarEvent:
    event_id: str
    title: str
    start: datetime
    end: datetime
    all_day: bool
    location: str | None
    organizer: str
    attendees: list[str]
    status: str                          # confirmed, tentative, cancelled
    description: str | None              # HTML or plain text body
    recurrence_id: str | None            # parent recurring event ID
    attachments: list[EventAttachment]
    html_link: str | None                # link back to calendar UI

@dataclass
class EventAttachment:
    filename: str
    mime_type: str
    url: str                             # download URL (Google) or content bytes
    size: int | None

class CalendarFetcher:
    """Fetches events from Google Calendar or Microsoft Graph API."""

    def __init__(self, config: DocumentConfig, config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    def fetch_events(self) -> list[CalendarEvent]:
        if self._config.provider == "google":
            return self._fetch_google()
        elif self._config.provider == "microsoft":
            return self._fetch_microsoft()
        raise ValueError(f"Unknown calendar provider: {self._config.provider}")

    def _get_time_range(self) -> tuple[datetime, datetime]:
        """Parse since/until or default to +/- 90 days."""
        now = datetime.now(timezone.utc)
        since = datetime.fromisoformat(self._config.since) if self._config.since else now - timedelta(days=90)
        until = datetime.fromisoformat(self._config.until) if self._config.until else now + timedelta(days=90)
        return since, until
```

### Google Calendar (REST API)

```python
    def _fetch_google(self) -> list[CalendarEvent]:
        token = self._get_access_token()
        since, until = self._get_time_range()
        calendar_ids = self._config.calendars or [self._config.calendar_id]
        all_events = []

        for cal_id in calendar_ids:
            url = f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events"
            params = {
                "timeMin": since.isoformat(),
                "timeMax": until.isoformat(),
                "singleEvents": str(self._config.expand_recurring).lower(),
                "orderBy": "startTime" if self._config.expand_recurring else "updated",
                "maxResults": min(self._config.max_events, 2500),
            }
            headers = {"Authorization": f"Bearer {token}"}

            events = []
            while True:
                resp = httpx.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                events.extend(data.get("items", []))
                page_token = data.get("nextPageToken")
                if not page_token or len(events) >= self._config.max_events:
                    break
                params["pageToken"] = page_token

            all_events.extend(self._parse_google_events(events[:self._config.max_events], cal_id))
        return all_events

    def _parse_google_events(self, items: list[dict], cal_id: str) -> list[CalendarEvent]:
        events = []
        for item in items:
            if not self._config.include_cancelled and item.get("status") == "cancelled":
                continue
            if not self._config.include_declined and self._is_declined(item):
                continue

            start = item.get("start", {})
            end = item.get("end", {})
            all_day = "date" in start
            start_dt = datetime.fromisoformat(start.get("dateTime") or start.get("date"))
            end_dt = datetime.fromisoformat(end.get("dateTime") or end.get("date"))

            attachments = []
            for att in item.get("attachments", []):
                attachments.append(EventAttachment(
                    filename=att.get("title", "attachment"),
                    mime_type=att.get("mimeType", "application/octet-stream"),
                    url=att.get("fileUrl", ""),
                    size=att.get("fileSize"),
                ))

            event_id = self._make_event_id(item, start_dt)
            events.append(CalendarEvent(
                event_id=event_id,
                title=item.get("summary", "(No title)"),
                start=start_dt, end=end_dt, all_day=all_day,
                location=item.get("location"),
                organizer=item.get("organizer", {}).get("email", ""),
                attendees=[a.get("email", "") for a in item.get("attendees", [])],
                status=item.get("status", "confirmed"),
                description=item.get("description"),
                recurrence_id=item.get("recurringEventId"),
                attachments=attachments,
                html_link=item.get("htmlLink"),
            ))
        return events
```

### Microsoft Graph API

```python
    def _fetch_microsoft(self) -> list[CalendarEvent]:
        token = self._get_access_token()
        since, until = self._get_time_range()
        cal_id = self._config.calendar_id

        if cal_id == "me":
            url = "https://graph.microsoft.com/v1.0/me/calendarView"
        else:
            url = f"https://graph.microsoft.com/v1.0/me/calendars/{cal_id}/calendarView"

        params = {
            "startDateTime": since.isoformat(),
            "endDateTime": until.isoformat(),
            "$top": min(self._config.max_events, 1000),
            "$select": "id,subject,start,end,isAllDay,location,organizer,attendees,body,webLink,hasAttachments",
        }
        headers = {"Authorization": f"Bearer {token}", "Prefer": 'outlook.timezone="UTC"'}

        events = []
        while True:
            resp = httpx.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            events.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            if not next_link or len(events) >= self._config.max_events:
                break
            url = next_link
            params = {}

        return self._parse_microsoft_events(events[:self._config.max_events])

    def _parse_microsoft_events(self, items: list[dict]) -> list[CalendarEvent]:
        events = []
        for item in items:
            start_dt = datetime.fromisoformat(item["start"]["dateTime"])
            end_dt = datetime.fromisoformat(item["end"]["dateTime"])

            attachments = []
            if item.get("hasAttachments"):
                attachments = self._fetch_ms_attachments(item["id"])

            event_id = self._make_event_id(item, start_dt)
            events.append(CalendarEvent(
                event_id=event_id,
                title=item.get("subject", "(No title)"),
                start=start_dt, end=end_dt,
                all_day=item.get("isAllDay", False),
                location=item.get("location", {}).get("displayName"),
                organizer=item.get("organizer", {}).get("emailAddress", {}).get("address", ""),
                attendees=[a["emailAddress"]["address"] for a in item.get("attendees", [])],
                status=item.get("responseStatus", {}).get("response", "none"),
                description=item.get("body", {}).get("content"),
                recurrence_id=None,  # calendarView already expands
                attachments=attachments,
                html_link=item.get("webLink"),
            ))
        return events

    def _fetch_ms_attachments(self, event_id: str) -> list[EventAttachment]:
        """Fetch attachments for a single Microsoft event."""
        token = self._get_access_token()
        url = f"https://graph.microsoft.com/v1.0/me/events/{event_id}/attachments"
        resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        attachments = []
        for att in resp.json().get("value", []):
            if att.get("@odata.type") == "#microsoft.graph.fileAttachment":
                import base64
                attachments.append(EventAttachment(
                    filename=att.get("name", "attachment"),
                    mime_type=att.get("contentType", "application/octet-stream"),
                    url="",  # data inline for MS Graph
                    size=att.get("size"),
                ))
        return attachments
```

### Event ID Generation

```python
    def _make_event_id(self, item: dict, start_dt: datetime) -> str:
        """Deterministic short ID from event data + date."""
        raw_id = item.get("id", "")
        short_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:8]
        date_str = start_dt.strftime("%Y%m%d")
        title_slug = re.sub(r"[^a-z0-9]", "", (item.get("summary") or item.get("subject") or "event").lower())[:20]
        return f"evt_{date_str}_{title_slug}_{short_hash}"
```

### Rendering Event as Document Text

```python
def _render_event(event: CalendarEvent, source_name: str) -> str:
    parts = [f"# {event.title}"]
    if event.all_day:
        parts.append(f"Date: {event.start.strftime('%Y-%m-%d')} (all day)")
    else:
        parts.append(f"Start: {event.start.isoformat()}")
        parts.append(f"End: {event.end.isoformat()}")
    if event.location:
        parts.append(f"Location: {event.location}")
    parts.append(f"Organizer: {event.organizer}")
    if event.attendees:
        parts.append(f"Attendees: {', '.join(event.attendees)}")
    parts.append(f"Status: {event.status}")
    parts.append("")
    if event.description:
        # Convert HTML to markdown if needed
        parts.append(event.description)
    if event.attachments:
        parts.append("\n## Attachments")
        for att in event.attachments:
            parts.append(f"- {att.filename} ({att.mime_type})")
    return "\n".join(parts)
```

### Integration with `_core.py`

In `_load_document(name)`:

```python
if doc_config.type == "calendar":
    fetcher = CalendarFetcher(doc_config, config_dir=self._config_dir)
    events = fetcher.fetch_events()
    for event in events:
        event_name = f"{name}:{event.event_id}"

        # Index event body
        if doc_config.extract_body:
            body_text = _render_event(event, name)
            self._loaded_documents[event_name] = LoadedDocument(
                name=event_name, content=body_text, doc_type="markdown",
            )

        # Index attachments
        if doc_config.extract_attachments:
            for att in event.attachments:
                att_name = f"{event_name}:{att.filename}"
                att_bytes = fetcher.download_attachment(att)
                att_type = detect_type_from_source(att.mime_type, att.filename)
                if att_type.startswith("image/"):
                    image_result = _extract_image(path=None, data=att_bytes)
                    if image_result.category == "image-primary":
                        desc = await self._describe_image(self._provider, att_bytes, att_type)
                        image_result.description = desc.description
                        image_result.subcategory = desc.subcategory
                        image_result.labels = desc.labels
                    content = _render_image_result(image_result, att.filename)
                else:
                    content = _extract_content_from_bytes(att_bytes, att_type)
                self._loaded_documents[att_name] = LoadedDocument(
                    name=att_name, content=content, doc_type=att_type,
                )
```

### Transport Layer (`_transport.py`)

Add `calendar` transport recognition in `infer_transport()`:

```python
if config.type == "calendar":
    return "calendar"
```

Calendar fetching is handled entirely by `CalendarFetcher`, not the generic `fetch_document()` path, because calendar APIs require multi-event iteration and provider-specific pagination. The transport inference is used only for type detection/validation.

### OAuth2 Authentication

Reuses the existing OAuth2 infrastructure from the IMAP implementation (`email.md`):

```python
def _get_access_token(self) -> str:
    if self._config.provider == "microsoft":
        provider = AzureOAuth2Provider(self._config)
    elif self._config.provider == "google":
        provider = GoogleOAuth2Provider(self._config)
    return provider.get_access_token()
```

Google Calendar requires `https://www.googleapis.com/auth/calendar.readonly` scope. Microsoft Graph requires `Calendars.Read` or `Calendars.ReadWrite`.

## Incremental Sync

Track which events have been indexed to avoid re-processing on subsequent loads.

```python
# In vector store: check if event already has chunks
def _is_event_indexed(self, event_name: str) -> bool:
    return self._vector_store.has_document(event_name)
```

Strategy:
1. First sync: fetch all events in the time window, index everything.
2. Subsequent syncs: use `updatedMin` (Google) or `$filter=lastModifiedDateTime ge ...` (Microsoft) to fetch only changed events.
3. For changed events: delete existing chunks, re-index.
4. Track `last_synced` timestamp in session metadata.

```python
# Google: incremental query
params["updatedMin"] = last_synced.isoformat()

# Microsoft: filter by modification time
params["$filter"] = f"lastModifiedDateTime ge {last_synced.isoformat()}"
```

The existing `source_refresher.py` handles the refresh scheduling. Calendar sources are eligible when `auto_refresh: true` (default).

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `httpx` | HTTP client for REST APIs | Already installed |
| `msal` | Microsoft OAuth2 (if using Microsoft) | Optional, `pip install msal` |
| `google-auth-oauthlib` | Google OAuth2 (if using Google) | Optional, `pip install google-auth-oauthlib` |

OAuth2 packages are optional — same as IMAP implementation. Import-guarded with helpful error messages.

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_calendar.py` | **New**: `CalendarFetcher`, `CalendarEvent`, `EventAttachment`, `_render_event()` |
| `constat/discovery/doc_tools/_core.py` | Add calendar branch in `_load_document()` |
| `constat/discovery/doc_tools/_transport.py` | Recognize `calendar` type |
| `constat/core/config.py` | Add calendar fields to `DocumentConfig` |
| `tests/test_calendar_ingestion.py` | Unit tests with mocked API responses |

## Testing Strategy

1. **Unit tests** (`test_calendar_ingestion.py`):
   - Mock `httpx.get` with canned Google Calendar API responses
   - Mock `httpx.get` with canned Microsoft Graph API responses
   - `_parse_google_events()` with single, recurring, all-day, cancelled events
   - `_parse_microsoft_events()` with attachments, HTML body, attendees
   - `_render_event()` output format for both all-day and timed events
   - Event ID generation: verify deterministic and unique per instance
   - Time range parsing: default +/- 90 days, custom since/until

2. **Integration tests**:
   - Full pipeline mock: fetch events → parse → index body + attachments
   - Verify chunks in vector store with correct `document_name` addresses
   - Incremental sync: second run with `updatedMin` skips unchanged events
   - Multi-calendar: events from two calendars indexed with correct source prefix

3. **Fixtures**: Construct test responses as JSON dicts matching API schemas:
   - Google: single event, recurring daily (3 instances), all-day event, event with attachment
   - Microsoft: event with HTML body, event with file attachment, declined event

## Edge Cases

| Case | Handling |
|------|----------|
| All-day events | Use `date` field instead of `dateTime`; render as date range without time |
| Recurring events (unexpanded) | If `expand_recurring: false`, index the master event once with recurrence rule in body |
| Cancelled occurrences | Skip unless `include_cancelled: true` |
| Events with no title | Use `"(No title)"` placeholder |
| HTML event descriptions | Convert to markdown via existing `html_to_markdown()` |
| Very long attendee lists (100+) | Truncate to first 50 attendees with "and N more" |
| Google Drive attachments | Attachment `fileUrl` points to Drive — download via Drive API with same OAuth token |
| Microsoft inline attachments | Decode base64 `contentBytes` from Graph API response |
| Timezone handling | Normalize all timestamps to UTC; store original timezone in metadata |
| OAuth2 token expiry mid-sync | Token refresh is handled by the OAuth2 provider layer (cached tokens with auto-refresh) |
| Rate limiting | Google Calendar API: 1M queries/day; Microsoft Graph: 10K/10min. Respect `Retry-After` headers. |

## Non-Goals (v1)

- Creating, updating, or deleting calendar events (read-only)
- Real-time push notifications (webhook/push subscriptions) — sync is pull-based
- Free/busy lookups or scheduling
- CalDAV protocol support (use provider REST APIs instead)
- Calendar-to-calendar sync or mirroring
- Meeting notes or recording transcripts (separate source type)
- ICS file import (could be added as a simple file-type document)
