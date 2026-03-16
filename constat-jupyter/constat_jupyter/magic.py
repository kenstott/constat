"""IPython magics for zero-code Constat interaction.

Usage::

    %load_ext constat_jupyter
    %constat connect sales-analytics,hr-reporting
    %%constat
    What are the top 10 products by revenue?

Cell magic (%%constat) is handled via an input transformer that rewrites
the cell into ``await _constat_run(...)``, so Jupyter's native async
execution blocks properly until the query completes.
"""
from __future__ import annotations

import html as _html
import shlex
from pathlib import Path
from textwrap import dedent

from IPython.core.magic import Magics, magics_class, line_magic, no_var_expand
from IPython.display import display, HTML

from .client import ConstatClient, Session


_TOKEN_CACHE_DIR = Path.home() / ".constat"


def _cache_token(server_url: str, token: str) -> None:
    """Persist token to ~/.constat/tokens.json keyed by server URL."""
    import json
    _TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _TOKEN_CACHE_DIR / "tokens.json"
    tokens = {}
    if cache_file.exists():
        try:
            tokens = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    tokens[server_url] = token
    cache_file.write_text(json.dumps(tokens))
    cache_file.chmod(0o600)


def _load_cached_token(server_url: str) -> str | None:
    """Load cached token for a server URL, or None."""
    import json
    cache_file = _TOKEN_CACHE_DIR / "tokens.json"
    if not cache_file.exists():
        return None
    try:
        tokens = json.loads(cache_file.read_text())
        return tokens.get(server_url)
    except (json.JSONDecodeError, OSError):
        return None


def _esc(text: str) -> str:
    """Escape text for safe HTML interpolation."""
    return _html.escape(str(text))


def _error_html(msg: str) -> HTML:
    """Render error box. Caller must escape dynamic values with _esc()."""
    return HTML(
        f'<div style="color:red;padding:8px;border:1px solid red;border-radius:4px">'
        f'<b>Error:</b> {msg}</div>'
    )


def _success_html(msg: str) -> HTML:
    """Render success box. Caller must escape dynamic values with _esc()."""
    return HTML(
        f'<div style="color:green;padding:8px;border:1px solid green;border-radius:4px">'
        f'{msg}</div>'
    )


def _info_html(msg: str) -> HTML:
    return HTML(
        f'<div style="padding:8px;border:1px solid #ccc;border-radius:4px">'
        f'{msg}</div>'
    )


def _table_html(headers: list[str], rows: list[list[str]], escape: bool = True) -> HTML:
    hdr = "".join(f"<th style='text-align:left;padding:4px 8px'>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td style='padding:4px 8px'>{_esc(c) if escape else c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    return HTML(
        f"<table style='border-collapse:collapse;margin:8px 0'>"
        f"<tr style='border-bottom:2px solid #ccc'>{hdr}</tr>{body}</table>"
    )


_USAGE = dedent("""\
    <b>Constat Magics</b><br><br>
    <code>%constat connect [domains]</code> — Connect to server, optionally set domains<br>
    <code>%constat login</code> — Authenticate (auto-detects local or Firebase)<br>
    <code>%constat login local</code> — Authenticate with username/password<br>
    <code>%constat login firebase</code> — Authenticate with Firebase email/password<br>
    <code>%constat status</code> — Show session info<br>
    <code>%constat domains</code> — List available domains<br>
    <code>%constat domains active</code> — Show active domains<br>
    <code>%constat domains add &lt;name&gt;</code> — Add a domain<br>
    <code>%constat domains drop &lt;name&gt;</code> — Remove a domain<br>
    <code>%constat sources</code> — Show data sources by domain<br>
    <code>%constat add database &lt;uri&gt; [name]</code> — Add a database<br>
    <code>%constat add api &lt;spec_url&gt; [name]</code> — Add an API<br>
    <code>%constat add document &lt;uri&gt;</code> — Add a document<br>
    <code>%constat tables</code> — List session tables<br>
    <code>%constat table &lt;name&gt;</code> — Display a table<br>
    <code>%constat artifacts</code> — List artifacts<br>
    <code>%constat artifact &lt;id&gt;</code> — Display an artifact<br><br>
    <code>%%constat</code> — Ask a question (cell body)<br>
    <code>%%constat new</code> — New session, then ask<br>
    <code>%%constat published</code> — Ask, display starred only<br>
    <code>%%constat code</code> — Ask, also display code artifacts<br>
    <code>%%constat output</code> — Ask, also display step output artifacts<br>
    <code>%%constat verbose</code> — Ask, display all artifacts (code + output)<br>
    <code>%%constat approve</code> — Ask, always require plan approval<br>
    <code>%%constat auto</code> — Ask, auto-approve all plans (no widget)<br>
    <code>%%constat include:md,html</code> — Only show these artifact types<br>
    <code>%%constat exclude:table</code> — Hide these artifact types<br>
""")


# ---------------------------------------------------------------------------
# Input transformer: rewrites %%constat cells into await _constat_run(...)
# ---------------------------------------------------------------------------

def _constat_cell_transform(lines: list[str]) -> list[str]:
    """Rewrite ``%%constat`` cells into ``await _constat_run(...)``."""
    if not lines:
        return lines
    first = lines[0].strip()
    if not first.startswith('%%constat'):
        return lines

    after_magic = first[len('%%constat'):].strip()
    tokens = after_magic.split() if after_magic else []
    question = ''.join(lines[1:]).strip()

    if not question:
        return lines  # Let it error naturally

    # Parse flags and type filters from tokens
    flags: list[str] = []
    include_types: list[str] = []
    exclude_types: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low.startswith("include:"):
            include_types.extend(low[len("include:"):].split(","))
        elif low.startswith("exclude:"):
            exclude_types.extend(low[len("exclude:"):].split(","))
        else:
            flags.append(low)

    published = 'published' in flags
    new_session = 'new' in flags
    show_code = 'code' in flags
    show_output = 'output' in flags
    if 'verbose' in flags:
        show_code = show_output = True

    # Approval modes:
    #   (default)     — conditional: server auto-approves simple plans, widget for complex
    #   "approve"     — always ask: force approval widget for every plan
    #   "auto"        — never ask: client auto-approves everything
    if 'auto' in flags:
        auto_approve = True
        require_approval = "None"
    elif 'approve' in flags:
        auto_approve = False
        require_approval = "True"
    else:
        # Conditional: let server decide, show widget when server asks
        auto_approve = False
        require_approval = "None"

    inc = repr(set(include_types)) if include_types else "None"
    exc = repr(set(exclude_types)) if exclude_types else "None"

    return [f"await _constat_run({repr(question)}, published={published}, new_session={new_session}, show_code={show_code}, show_output={show_output}, include_types={inc}, exclude_types={exc}, auto_approve={auto_approve}, require_approval={require_approval})\n"]


@magics_class
class ConstatMagic(Magics):

    def __init__(self, shell):
        super().__init__(shell)
        self.client: ConstatClient | None = None
        self.session: Session | None = None
        self.domains: list[str] = []
        self._has_asked: bool = False

    def _ensure_connected(self) -> bool:
        if self.session is None:
            display(_error_html("Not connected. Run <code>%constat connect</code> first."))
            return False
        return True

    # ---- Line magic ----

    @no_var_expand
    @line_magic
    def constat(self, line: str):
        """Constat line magic: dispatches subcommands."""
        line = line.strip()
        if not line:
            display(_info_html(_USAGE))
            return

        try:
            parts = shlex.split(line)
        except ValueError as e:
            display(_error_html(_esc(str(e))))
            return
        cmd = parts[0].lower()
        args = parts[1:]

        dispatch = {
            "connect": self._connect_cmd,
            "login": self._login_cmd,
            "status": self._status_cmd,
            "domains": self._domains_cmd,
            "sources": self._sources_cmd,
            "add": self._add_cmd,
            "tables": self._tables_cmd,
            "table": self._table_cmd,
            "artifacts": self._artifacts_cmd,
            "artifact": self._artifact_cmd,
        }

        handler = dispatch.get(cmd)
        if handler is None:
            display(_error_html(f"Unknown subcommand: <code>{_esc(cmd)}</code>"))
            display(_info_html(_USAGE))
            return
        handler(args)

    # ---- Subcommand handlers ----

    def _connect_cmd(self, args: list[str]):
        import os
        # Auto-restore cached token if not already set
        if not os.environ.get("CONSTAT_AUTH_TOKEN"):
            server_url = os.environ.get("CONSTAT_SERVER_URL", "http://localhost:8000").rstrip("/")
            cached = _load_cached_token(server_url)
            if cached:
                os.environ["CONSTAT_AUTH_TOKEN"] = cached

        try:
            self.client = ConstatClient()
        except Exception as e:
            display(_error_html(_esc(str(e))))
            return

        try:
            self.session = self.client.create_session()
        except Exception as e:
            display(_error_html(_esc(str(e))))
            return

        if args:
            self.domains = [d.strip() for d in args[0].split(",") if d.strip()]
            try:
                self.session.set_domains(self.domains)
            except Exception as e:
                display(_error_html(f"Connected but failed to set domains: {_esc(str(e))}"))
                self._inject_globals()
                return

        self._has_asked = False
        self._inject_globals()

        msg = f"Connected. Session: <code>{_esc(self.session.session_id)}</code>"
        if self.domains:
            msg += f"<br>Domains: {_esc(', '.join(self.domains))}"
        display(_success_html(msg))

    def _login_cmd(self, args: list[str]):
        import os
        import getpass
        import httpx

        server_url = os.environ.get("CONSTAT_SERVER_URL", "http://localhost:8000").rstrip("/")

        # Check for cached token first (skip prompt if valid)
        if not os.environ.get("CONSTAT_AUTH_TOKEN"):
            cached = _load_cached_token(server_url)
            if cached:
                os.environ["CONSTAT_AUTH_TOKEN"] = cached

        if os.environ.get("CONSTAT_AUTH_TOKEN"):
            # Verify the cached token still works
            try:
                resp = httpx.get(
                    f"{server_url}/api/sessions",
                    headers={"Authorization": f"Bearer {os.environ['CONSTAT_AUTH_TOKEN']}"},
                    timeout=10,
                )
                if resp.status_code != 401:
                    display(_success_html("Already logged in (cached). Run <code>%constat connect</code> to start a session."))
                    return
            except Exception:
                pass
            # Token invalid — clear and fall through to interactive login
            os.environ.pop("CONSTAT_AUTH_TOKEN", None)

        # Determine auth method: explicit arg, or auto-detect from /health
        method = args[0].lower() if args else None
        if method not in ("local", "firebase", None):
            display(_error_html(f"Unknown auth method: <code>{_esc(method)}</code>. Use <code>local</code> or <code>firebase</code>."))
            return

        if method is None:
            try:
                resp = httpx.get(f"{server_url}/health", timeout=10)
                resp.raise_for_status()
                health = resp.json()
                auth_info = health.get("auth", {})
                if auth_info.get("auth_disabled"):
                    display(_info_html("Auth is disabled on this server. No login needed."))
                    return
                methods = auth_info.get("auth_methods", [])
                if not methods:
                    display(_error_html("Server reports no auth methods configured."))
                    return
                if "local" in methods:
                    method = "local"
                elif len(methods) == 1:
                    method = methods[0]
                else:
                    display(_info_html(
                        f"Multiple auth methods available: {_esc(', '.join(methods))}.<br>"
                        f"Use <code>%constat login local</code> or <code>%constat login firebase</code>."
                    ))
                    return
            except Exception as e:
                display(_error_html(f"Failed to reach server: {_esc(str(e))}"))
                return

        if method == "local":
            username = input("Username: ")
            password = getpass.getpass("Password: ")
            endpoint = f"{server_url}/api/auth/login"
            payload = {"username": username, "password": password}
        else:
            username = input("Email: ")
            password = getpass.getpass("Password: ")
            endpoint = f"{server_url}/api/auth/firebase-login"
            payload = {"email": username, "password": password}

        try:
            resp = httpx.post(endpoint, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            os.environ["CONSTAT_AUTH_TOKEN"] = data["token"]
            _cache_token(server_url, data["token"])
            display(_success_html(
                f"Logged in as <code>{_esc(data.get('user_id') or data.get('email', ''))}</code>. "
                f"Run <code>%constat connect</code> to start a session."
            ))
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            display(_error_html(f"Login failed: {_esc(detail or str(e))}"))
        except Exception as e:
            display(_error_html(f"Login failed: {_esc(str(e))}"))

    def _status_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        try:
            status = self.session.status
            rows = [
                ["Session ID", f"<code>{_esc(self.session.session_id)}</code>"],
                ["Active Domains", ", ".join(status.get("active_domains", self.domains)) or "<i>none</i>"],
                ["Status", status.get("status", "unknown")],
            ]
            display(_table_html(["Key", "Value"], rows, escape=False))
        except Exception as e:
            display(_error_html(_esc(str(e))))

    def _domains_cmd(self, args: list[str]):
        if not args:
            # List all available domains on server
            if self.client is None:
                display(_error_html("Not connected. Run <code>%constat connect</code> first."))
                return
            try:
                domains = self.client.domains()
                if not domains:
                    display(_info_html("No domains configured on server."))
                    return
                rows = [[d.get("name", ""), d.get("description", "")] for d in domains]
                display(_table_html(["Name", "Description"], rows))
            except Exception as e:
                display(_error_html(_esc(str(e))))
            return

        subcmd = args[0].lower()

        if subcmd == "active":
            if not self._ensure_connected():
                return
            try:
                active = self.session.status.get("active_domains", self.domains)
                if active:
                    display(_info_html("Active domains: " + ", ".join(active)))
                else:
                    display(_info_html("No active domains."))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        elif subcmd == "add":
            if not self._ensure_connected():
                return
            if len(args) < 2:
                display(_error_html("Usage: <code>%constat domains add &lt;name&gt;</code>"))
                return
            name = args[1]
            try:
                active = list(self.session.status.get("active_domains", self.domains))
                if name not in active:
                    active.append(name)
                self.session.set_domains(active)
                self.domains = active
                display(_success_html(f"Added domain: {_esc(name)}"))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        elif subcmd == "drop":
            if not self._ensure_connected():
                return
            if len(args) < 2:
                display(_error_html("Usage: <code>%constat domains drop &lt;name&gt;</code>"))
                return
            name = args[1]
            try:
                active = list(self.session.status.get("active_domains", self.domains))
                active = [d for d in active if d != name]
                self.session.set_domains(active)
                self.domains = active
                display(_success_html(f"Dropped domain: {_esc(name)}"))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        else:
            display(_error_html(f"Unknown domains subcommand: <code>{_esc(subcmd)}</code>"))

    def _sources_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        try:
            sources = self.session.sources()
            parts = []
            for domain_name, domain_sources in sources.items():
                if domain_name.startswith("_"):
                    continue
                parts.append(f"<b>{_esc(domain_name)}</b>")
                dbs = domain_sources if isinstance(domain_sources, list) else domain_sources.get("databases", [])
                apis = [] if isinstance(domain_sources, list) else domain_sources.get("apis", [])
                docs = [] if isinstance(domain_sources, list) else domain_sources.get("documents", [])
                if isinstance(domain_sources, dict):
                    for db in dbs:
                        parts.append(f"&nbsp;&nbsp;DB: {_esc(db.get('name', db))}")
                    for api in apis:
                        parts.append(f"&nbsp;&nbsp;API: {_esc(api.get('name', api))}")
                    for doc in docs:
                        parts.append(f"&nbsp;&nbsp;Doc: {_esc(doc.get('name', doc))}")
                else:
                    for src in domain_sources:
                        name = src.get("name", "") if isinstance(src, dict) else str(src)
                        parts.append(f"&nbsp;&nbsp;{_esc(name)}")
            if parts:
                display(_info_html("<br>".join(parts)))
            else:
                display(_info_html("No sources found."))
        except Exception as e:
            display(_error_html(_esc(str(e))))

    def _add_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        if not args:
            display(_error_html("Usage: <code>%constat add database|api|document &lt;uri&gt; [name]</code>"))
            return

        source_type = args[0].lower()

        if source_type == "database":
            if len(args) < 2:
                display(_error_html("Usage: <code>%constat add database &lt;uri&gt; [name]</code>"))
                return
            uri = args[1]
            name = args[2] if len(args) > 2 else None
            try:
                self.session.add_database(uri, name)
                display(_success_html(f"Added database: {_esc(name or uri)}"))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        elif source_type == "api":
            if len(args) < 2:
                display(_error_html("Usage: <code>%constat add api &lt;spec_url&gt; [name]</code>"))
                return
            spec_url = args[1]
            name = args[2] if len(args) > 2 else None
            try:
                self.session.add_api(spec_url, name)
                display(_success_html(f"Added API: {_esc(name or spec_url)}"))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        elif source_type == "document":
            if len(args) < 2:
                display(_error_html("Usage: <code>%constat add document &lt;uri&gt;</code>"))
                return
            uri = args[1]
            try:
                self.session.add_document(uri)
                display(_success_html(f"Added document: {_esc(uri)}"))
            except Exception as e:
                display(_error_html(_esc(str(e))))

        else:
            display(_error_html(f"Unknown source type: <code>{_esc(source_type)}</code>. Use database, api, or document."))

    def _tables_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        try:
            tables = self.session.tables()
            if not tables:
                display(_info_html("No tables in session."))
                return
            rows = []
            for t in tables:
                name = t.get("name", "")
                rows_count = t.get("row_count", "")
                cols = t.get("column_count", "")
                starred = "Y" if t.get("is_starred") else ""
                rows.append([name, str(rows_count), str(cols), starred])
            display(_table_html(["Name", "Rows", "Columns", "Starred"], rows))
        except Exception as e:
            display(_error_html(_esc(str(e))))

    def _table_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        if not args:
            display(_error_html("Usage: <code>%constat table &lt;name&gt;</code>"))
            return
        name = args[0]
        try:
            df = self.session.table(name)
            try:
                import itables
                itables.show(df.to_pandas() if hasattr(df, "to_pandas") else df,
                             buttons=["csvHtml5", "excelHtml5"])
            except ImportError:
                display(df)
        except Exception as e:
            display(_error_html(_esc(str(e))))

    def _artifacts_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        try:
            artifacts = self.session.artifacts()
            if not artifacts:
                display(_info_html("No artifacts in session."))
                return
            rows = []
            for a in artifacts:
                starred = "Y" if a.is_starred else ""
                rows.append([str(a.id), a.name, a.artifact_type, starred])
            display(_table_html(["ID", "Name", "Type", "Starred"], rows))
        except Exception as e:
            display(_error_html(_esc(str(e))))

    def _artifact_cmd(self, args: list[str]):
        if not self._ensure_connected():
            return
        if not args:
            display(_error_html("Usage: <code>%constat artifact &lt;id&gt;</code>"))
            return
        try:
            artifact_id = int(args[0])
        except ValueError:
            display(_error_html("Artifact ID must be an integer."))
            return
        try:
            artifact = self.session.artifact(artifact_id)
            artifact.display()
        except Exception as e:
            display(_error_html(_esc(str(e))))

    # ---- Async run function (injected into user namespace) ----

    def _make_run_fn(self):
        """Create the async _constat_run function bound to this magic instance."""
        magic = self

        async def _constat_run(
            question: str, *, published: bool = False, new_session: bool = False,
            show_code: bool = False, show_output: bool = False,
            include_types: set[str] | None = None, exclude_types: set[str] | None = None,
            auto_approve: bool = False, require_approval: bool | None = None,
        ):
            if magic.session is None:
                display(_error_html("Not connected. Run <code>%constat connect</code> first."))
                return None

            try:
                session = magic.session
                if new_session:
                    session = magic.client.create_session()
                    if magic.domains:
                        session.set_domains(magic.domains)
                    magic.session = session
                    magic.shell.user_ns['_constat_session'] = session
                    magic._has_asked = False

                if magic._has_asked:
                    result = await session.follow_up(question, auto_approve=auto_approve, require_approval=require_approval)
                else:
                    result = await session.solve(question, auto_approve=auto_approve, require_approval=require_approval)

                magic._has_asked = True
                magic.shell.user_ns['_constat_result'] = result

                # Always use display() which renders answer as Markdown
                result.display(
                    published=published, show_code=show_code, show_output=show_output,
                    include_types=include_types, exclude_types=exclude_types,
                )
                return None
            except Exception as e:
                display(_error_html(_esc(str(e))))
                return None

        return _constat_run

    # ---- Helpers ----

    def _inject_globals(self):
        """Inject client/session into notebook namespace for power users."""
        self.shell.user_ns["_constat_client"] = self.client
        self.shell.user_ns["_constat_session"] = self.session
        self.shell.user_ns["_constat_run"] = self._make_run_fn()
