# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Download code endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from constat.server.auth import CurrentUserId
from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{session_id}/download-code")
async def download_code(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Download all step codes as a standalone Python script.

    Generates a self-contained Python script that can be run independently
    to reproduce the analysis. Includes all step functions, imports, and
    helper utilities. Facts are loaded from _facts.parquet and passed as
    explicit arguments to run_analysis().

    Args:
        session_id: Session ID
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Python script file download

    Raises:
        404: Session not found or no code available
    """
    from fastapi.responses import Response

    # Try to get the session from memory first
    # noinspection DuplicatedCode
    managed = session_manager.get_session_or_none(session_id)

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
        logger.debug(f"[download-code] Found managed session. Server: {session_id}, History: {history_session_id}")
    else:
        # Session not in memory - try reverse lookup from disk
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=user_id)
        history_session_id = history.find_session_by_server_id(session_id)
        logger.debug(f"[download-code] Session not in memory. Reverse lookup found: {history_session_id}")

    try:
        if not history_session_id:
            logger.warning(f"[download-code] No history session ID for server session {session_id}")
            raise HTTPException(
                status_code=404,
                detail="No code available for this session. Run a query first to generate step code."
            )

        # Check if the steps directory exists
        steps_dir = history._steps_dir(history_session_id)
        logger.debug(f"[download-code] Steps directory: {steps_dir}, exists: {steps_dir.exists() if steps_dir else 'N/A'}")

        steps = history.list_step_codes(history_session_id)
        logger.debug(f"[download-code] Found {len(steps)} steps for history session {history_session_id}")

        if not steps:
            # Provide more context for debugging
            detail = "No code available for this session."
            if history_session_id:
                detail += f" History session {history_session_id} has no steps."
                if steps_dir.exists():
                    # Check what's in the directory
                    try:
                        contents = list(steps_dir.iterdir())
                        detail += f" Steps dir exists with {len(contents)} files."
                    except OSError:
                        pass
            detail += " Run a query first to generate step code."
            raise HTTPException(status_code=404, detail=detail)

        # Get facts from the _facts table (if session is in memory and has datastore)
        facts_list = []
        if managed and managed.session.datastore:
            try:
                facts_df = managed.session.datastore.load_dataframe("_facts")
                for _, row in facts_df.iterrows():
                    facts_list.append({
                        "name": row.get("name", ""),
                        "value": row.get("value", ""),
                        "description": row.get("description", ""),
                    })
            except (KeyError, ValueError, OSError):
                # No facts table - that's okay
                pass

        # Get data sources from config (if session is in memory)
        databases = []
        apis = []
        files = []
        llm_config = None
        email_config = None

        if managed and managed.session.config:
            config = managed.session.config
            if config.databases:
                for name, db_config in config.databases.items():
                    if db_config.is_file_source():
                        files.append({
                            "name": name,
                            "path": db_config.path,
                            "description": db_config.description or "",
                        })
                    else:
                        databases.append({
                            "name": name,
                            "type": db_config.type or "sql",
                            "uri": db_config.uri or "",
                            "description": db_config.description or "",
                        })

            if config.apis:
                for name, api_config in config.apis.items():
                    apis.append({
                        "name": name,
                        "type": api_config.type,
                        "url": api_config.url or "",
                        "description": api_config.description or "",
                    })

            # Extract LLM config
            if config.llm:
                llm_config = {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "api_key": config.llm.api_key,
                    "base_url": config.llm.base_url,
                }

            # Extract email config
            if config.email:
                email_config = {
                    "smtp_host": config.email.smtp_host,
                    "smtp_port": config.email.smtp_port,
                    "smtp_user": config.email.smtp_user,
                    "smtp_password": config.email.smtp_password,
                    "from_address": config.email.from_address,
                    "from_name": config.email.from_name,
                    "tls": config.email.tls,
                }

        # Build standalone Python script
        script_lines = [
            '#!/usr/bin/env python3',
            '"""',
            f'Constat Analysis Script - Session {session_id[:8]}',
            '',
            'This script was automatically generated from a Constat analysis session.',
            'It contains all the step code that was executed during the analysis.',
            '',
            'Usage:',
            '  1. Edit _facts.parquet with your context values, then run:',
            '     python script.py',
            '',
            '  2. Or call run_analysis() directly with your values:',
            '     from script import run_analysis',
        ]

        # Add example call with explicit args
        if facts_list:
            args_example = ", ".join(f'{f["name"]}="..."' for f in facts_list)
            script_lines.append(f'     run_analysis({args_example})')
        script_lines.extend([
            '"""',
            '',
            'import os',
            'import pandas as pd',
            'import numpy as np',
            'import duckdb',
            'from pathlib import Path',
        ])

        # Add SQLAlchemy import if there are SQL databases
        if any(db['type'] in ('sql', 'postgresql', 'mysql', 'sqlite') for db in databases):
            script_lines.append('from sqlalchemy import create_engine')

        # Helper to format multi-line descriptions as comments
        def format_description_comment(description: str, prefix: str = "#   ") -> list[str]:
            """Format a description as properly wrapped comment lines."""
            if not description:
                return []
            # Split on newlines and wrap each line
            comment_lines = []
            for sentence in description.replace('\n', ' ').split('. '):
                sentence = sentence.strip()
                if sentence:
                    if not sentence.endswith('.'):
                        sentence += '.'
                    comment_lines.append(f"{prefix}{sentence}")
            return comment_lines

        # Add data sources section if there are any
        if databases or apis or files:
            script_lines.extend([
                '',
                '# ============================================================================',
                '# Data Sources (from Constat config)',
                '# ============================================================================',
                '# Configure these for your environment. Values containing secrets should',
                '# use environment variables: os.environ["VAR_NAME"]',
                '',
            ])

            if databases:
                script_lines.append('# Databases')
                for db in databases:
                    uri = db['uri']
                    # Mask passwords in URIs for safety, suggest env var
                    if '@' in uri and ':' in uri.split('@')[0]:
                        # Has embedded credentials - suggest env var
                        script_lines.append(f"# db_{db['name']}: {db['type']} - credentials detected, use env var")
                        script_lines.append(f"# db_{db['name']} = create_engine(os.environ['DB_{db['name'].upper()}_URI'])")
                        # Also show masked version
                        masked = uri.split('://')[0] + '://***:***@' + uri.split('@')[-1] if '://' in uri else uri
                        script_lines.append(f"# Original (masked): {masked}")
                    else:
                        script_lines.append(f"# db_{db['name']}: {db['type']}")
                        # Add description as wrapped comment lines
                        if db['description']:
                            script_lines.extend(format_description_comment(db['description']))
                        if 'duckdb' in uri.lower():
                            # DuckDB uses its own connect() method
                            script_lines.append(f"db_{db['name']} = duckdb.connect('{uri.replace('duckdb:///', '')}')")
                        else:
                            # SQLite, PostgreSQL, MySQL, etc. use SQLAlchemy
                            script_lines.append(f"db_{db['name']} = create_engine('{uri}')")
                    script_lines.append('')

            if apis:
                script_lines.append('# APIs')
                for api in apis:
                    script_lines.append(f"# api_{api['name']}: {api['type']} - {api['url']}")
                    # Add description as wrapped comment lines
                    if api['description']:
                        script_lines.extend(format_description_comment(api['description']))
                    # Add config variable for the API base URL
                    script_lines.append(f"API_{api['name'].upper()}_URL = '{api['url']}'")
                    script_lines.append('')

            if files:
                script_lines.append('# Files')
                for f in files:
                    script_lines.append(f"# file_{f['name']}")
                    # Add description as wrapped comment lines
                    if f['description']:
                        script_lines.extend(format_description_comment(f['description']))
                    script_lines.append(f"file_{f['name']} = Path('{f['path']}')")
                script_lines.append('')

        # Add LLM configuration section
        script_lines.extend([
            '',
            '# ============================================================================',
            '# LLM Configuration',
            '# ============================================================================',
            '',
        ])
        if llm_config and llm_config['api_key']:
            script_lines.append(f'LLM_PROVIDER = "{llm_config["provider"]}"')
            script_lines.append(f'LLM_MODEL = "{llm_config["model"]}"')
            script_lines.append(f'LLM_API_KEY = "{llm_config["api_key"]}"')
            script_lines.append(f'LLM_BASE_URL = "{llm_config["base_url"]}"' if llm_config['base_url'] else 'LLM_BASE_URL = None')
        else:
            script_lines.append('LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")')
            script_lines.append('LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")')
            script_lines.append('LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY")')
            script_lines.append('LLM_BASE_URL = os.environ.get("LLM_BASE_URL")')
        script_lines.append('')

        # Add Email configuration section
        script_lines.extend([
            '',
            '# ============================================================================',
            '# Email Configuration',
            '# ============================================================================',
            '',
        ])
        if email_config and email_config['smtp_host']:
            script_lines.append(f'SMTP_HOST = "{email_config["smtp_host"]}"')
            script_lines.append(f'SMTP_PORT = {email_config["smtp_port"]}')
            script_lines.append(f'SMTP_USER = "{email_config["smtp_user"]}"' if email_config['smtp_user'] else 'SMTP_USER = os.environ.get("SMTP_USER")')
            script_lines.append(f'SMTP_PASSWORD = "{email_config["smtp_password"]}"' if email_config['smtp_password'] else 'SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")')
            script_lines.append(f'SMTP_FROM = "{email_config["from_address"]}"')
            script_lines.append(f'SMTP_FROM_NAME = "{email_config["from_name"]}"')
            script_lines.append(f'SMTP_USE_TLS = {email_config["tls"]}')
        else:
            script_lines.append('SMTP_HOST = os.environ.get("SMTP_HOST")')
            script_lines.append('SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))')
            script_lines.append('SMTP_USER = os.environ.get("SMTP_USER")')
            script_lines.append('SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")')
            script_lines.append('SMTP_FROM = os.environ.get("SMTP_FROM")')
            script_lines.append('SMTP_FROM_NAME = "Constat"')
            script_lines.append('SMTP_USE_TLS = True')
        script_lines.append('')

        script_lines.extend([
            '',
            '# ============================================================================',
            '# Store Class (Data persistence between steps)',
            '# ============================================================================',
            '',
            'class _DataStore:',
            '    """Simple datastore for sharing data between steps."""',
            '',
            '    def __init__(self):',
            '        self._tables: dict[str, pd.DataFrame] = {}',
            '        self._state: dict[str, any] = {}',
            '        self._conn = duckdb.connect()',
            '',
            '    def save_dataframe(self, name: str, df: pd.DataFrame, step_number: int = 0, description: str = "") -> None:',
            '        """Save a DataFrame to the store."""',
            '        self._tables[name] = df',
            '        self._conn.register(name, df)',
            '        print(f"Saved table: {name} ({len(df)} rows)")',
            '',
            '    def load_dataframe(self, name: str) -> pd.DataFrame:',
            '        """Load a DataFrame from the store."""',
            '        if name not in self._tables:',
            '            raise ValueError(f"Table not found: {name}. Available: {list(self._tables.keys())}")',
            '        return self._tables[name]',
            '',
            '    def query(self, sql: str) -> pd.DataFrame:',
            '        """Execute SQL query against stored DataFrames."""',
            '        return self._conn.execute(sql).fetchdf()',
            '',
            '    def set_state(self, key: str, value: any, step_number: int = 0) -> None:',
            '        """Save a state variable."""',
            '        self._state[key] = value',
            '',
            '    def get_state(self, key: str) -> any:',
            '        """Get a state variable (returns None if not found)."""',
            '        return self._state.get(key)',
            '',
            '    def list_tables(self) -> list[str]:',
            '        """List all stored tables."""',
            '        return list(self._tables.keys())',
            '',
            '    def table_exists(self, name: str) -> bool:',
            '        """Check if a table exists."""',
            '        return name in self._tables',
            '',
            '',
            '# ============================================================================',
            '# Visualization Helper (Save charts, files, and outputs)',
            '# ============================================================================',
            '',
            'class _VizHelper:',
            '    """Helper for saving visualizations and files."""',
            '',
            '    def __init__(self, output_dir: Path = None):',
            '        self.output_dir = output_dir or Path("./outputs")',
            '        self.output_dir.mkdir(parents=True, exist_ok=True)',
            '',
            '    def _save_and_print(self, filepath: Path, description: str) -> Path:',
            '        """Print file URI and return path."""',
            '        print(f"{description}: {filepath.resolve().as_uri()}")',
            '        return filepath',
            '',
            '    def save_file(self, name: str, content: str, ext: str = "txt", title: str = None, description: str = None) -> Path:',
            '        """Save a text file."""',
            '        filepath = self.output_dir / f"{name}.{ext}"',
            '        filepath.write_text(content)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_html(self, name: str, html_content: str, title: str = None, description: str = None) -> Path:',
            '        """Save HTML content."""',
            '        return self.save_file(name, html_content, ext="html", title=title)',
            '',
            '    def save_chart(self, name: str, figure: any, title: str = None, description: str = None, chart_type: str = "plotly") -> Path:',
            '        """Save a Plotly or matplotlib chart."""',
            '        filepath = self.output_dir / f"{name}.html"',
            '        if hasattr(figure, "write_html"):  # Plotly',
            '            figure.write_html(str(filepath))',
            '        elif hasattr(figure, "savefig"):  # Matplotlib',
            '            filepath = self.output_dir / f"{name}.png"',
            '            figure.savefig(str(filepath))',
            '        else:',
            '            raise ValueError(f"Unknown figure type: {type(figure)}")',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_image(self, name: str, figure: any, fmt: str = "png", title: str = None, description: str = None) -> Path:',
            '        """Save a matplotlib figure as image."""',
            '        filepath = self.output_dir / f"{name}.{fmt}"',
            '        figure.savefig(str(filepath), format=fmt)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_map(self, name: str, folium_map: any, title: str = None, description: str = None) -> Path:',
            '        """Save a folium map."""',
            '        filepath = self.output_dir / f"{name}.html"',
            '        folium_map.save(str(filepath))',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_binary(self, name: str, content: bytes, ext: str = "xlsx", title: str = None, description: str = None) -> Path:',
            '        """Save binary content (Excel, images, etc.)."""',
            '        filepath = self.output_dir / f"{name}.{ext}"',
            '        filepath.write_bytes(content)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_excel(self, name: str, df: pd.DataFrame, title: str = None, sheet_name: str = "Sheet1") -> Path:',
            '        """Save DataFrame to Excel file (.xlsx). Requires openpyxl."""',
            '        try:',
            '            filepath = self.output_dir / f"{name}.xlsx"',
            '            df.to_excel(str(filepath), sheet_name=sheet_name, index=False)',
            '            return self._save_and_print(filepath, title or name)',
            '        except ModuleNotFoundError:',
            '            raise ImportError("openpyxl required: pip install openpyxl")',
            '',
            '    def save_csv(self, name: str, df: pd.DataFrame, title: str = None) -> Path:',
            '        """Save DataFrame to CSV file."""',
            '        filepath = self.output_dir / f"{name}.csv"',
            '        df.to_csv(str(filepath), index=False)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_word(self, name: str, content: str, title: str = None) -> Path:',
            '        """Save content to Word document (.docx). Requires python-docx."""',
            '        try:',
            '            from docx import Document',
            '            doc = Document()',
            '            for para in content.split("\\n\\n"):',
            '                doc.add_paragraph(para)',
            '            filepath = self.output_dir / f"{name}.docx"',
            '            doc.save(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("python-docx required: pip install python-docx")',
            '',
            '    def save_pdf(self, name: str, content: str, title: str = None) -> Path:',
            '        """Save content to PDF. Requires fpdf2."""',
            '        try:',
            '            from fpdf import FPDF',
            '            pdf = FPDF()',
            '            pdf.add_page()',
            '            pdf.set_font("Helvetica", size=11)',
            '            pdf.multi_cell(0, 5, content)',
            '            filepath = self.output_dir / f"{name}.pdf"',
            '            pdf.output(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("fpdf2 required: pip install fpdf2")',
            '',
            '    def save_powerpoint(self, name: str, slides: list[dict], title: str = None) -> Path:',
            '        """Save to PowerPoint (.pptx). Requires python-pptx.',
            '        ',
            '        Args:',
            '            slides: List of dicts with "title" and "content" keys',
            '        """',
            '        try:',
            '            from pptx import Presentation',
            '            from pptx.util import Inches, Pt',
            '            prs = Presentation()',
            '            for slide_data in slides:',
            '                slide = prs.slides.add_slide(prs.slide_layouts[1])',
            '                slide.shapes.title.text = slide_data.get("title", "")',
            '                slide.placeholders[1].text = slide_data.get("content", "")',
            '            filepath = self.output_dir / f"{name}.pptx"',
            '            prs.save(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("python-pptx required: pip install python-pptx")',
            '',
            '',
            '# Create global instances',
            'store = _DataStore()',
            'viz = _VizHelper()',
            '',
            '# Legacy helper functions (for backwards compatibility)',
            'save_dataframe = store.save_dataframe',
            'load_dataframe = store.load_dataframe',
            'query = store.query',
            '',
            '',
            '# ============================================================================',
            '# LLM Helper Function',
            '# ============================================================================',
            '',
            'def llm_ask(question: str) -> str:',
            '    """Query an LLM for general knowledge.',
            '    ',
            '    Args:',
            '        question: The question to ask',
            '    ',
            '    Returns:',
            '        The LLM response text',
            '    """',
            '    if not LLM_API_KEY:',
            '        raise ValueError("llm_ask() requires LLM_API_KEY")',
            '    ',
            '    if LLM_PROVIDER == "anthropic":',
            '        try:',
            '            import anthropic',
            '            client = anthropic.Anthropic(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL else anthropic.Anthropic(api_key=LLM_API_KEY)',
            '            response = client.messages.create(',
            '                model=LLM_MODEL,',
            '                max_tokens=2048,',
            '                messages=[{"role": "user", "content": question}]',
            '            )',
            '            return response.content[0].text',
            '        except ImportError:',
            '            raise ImportError("anthropic package required: pip install anthropic")',
            '    elif LLM_PROVIDER == "openai":',
            '        try:',
            '            import openai',
            '            client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL else openai.OpenAI(api_key=LLM_API_KEY)',
            '            response = client.chat.completions.create(',
            '                model=LLM_MODEL,',
            '                messages=[{"role": "user", "content": question}]',
            '            )',
            '            return response.choices[0].message.content',
            '        except ImportError:',
            '            raise ImportError("openai package required: pip install openai")',
            '    else:',
            '        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")',
            '',
            '',
            '# ============================================================================',
            '# Email Helper Function',
            '# ============================================================================',
            '',
            '# Basic email-safe CSS for rendered Markdown',
            '_EMAIL_CSS = """',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #333; }',
            'h1, h2, h3 { color: #2c3e50; margin-top: 1em; margin-bottom: 0.5em; }',
            'p { margin: 0.5em 0; }',
            'ul, ol { margin: 0.5em 0; padding-left: 1.5em; }',
            'table { border-collapse: collapse; margin: 1em 0; }',
            'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            'th { background-color: #f5f5f5; }',
            'code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }',
            '</style>',
            '"""',
            '',
            'def _markdown_to_html(text: str) -> str:',
            '    """Convert Markdown text to HTML with email-safe styling."""',
            '    try:',
            '        import markdown',
            '        html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])',
            '        return f"<!DOCTYPE html><html><head>{_EMAIL_CSS}</head><body>{html_body}</body></html>"',
            '    except ImportError:',
            '        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")',
            '        html_body = f"<pre>{escaped}</pre>"',
            '        return f"<!DOCTYPE html><html><head>{_EMAIL_CSS}</head><body>{html_body}</body></html>"',
            '',
            '',
            'def send_email(',
            '    to: str,',
            '    subject: str,',
            '    body: str,',
            '    format: str = "plain",',
            '    df: pd.DataFrame = None,',
            '    attachment_name: str = "data.csv",',
            ') -> bool:',
            '    """Send an email with optional DataFrame attachment.',
            '    ',
            '    Args:',
            '        to: Recipient email address (or comma-separated list)',
            '        subject: Email subject',
            '        body: Email body (plain text, Markdown, or HTML)',
            '        format: Body format - "plain" (default), "markdown", or "html"',
            '        df: Optional DataFrame to attach',
            '        attachment_name: Filename for the DataFrame attachment',
            '    ',
            '    Returns:',
            '        True if sent successfully',
            '    """',
            '    import smtplib',
            '    from email.mime.text import MIMEText',
            '    from email.mime.multipart import MIMEMultipart',
            '    from email.mime.base import MIMEBase',
            '    from email import encoders',
            '    ',
            '    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):',
            '        raise ValueError("send_email() requires SMTP_HOST, SMTP_USER, SMTP_PASSWORD")',
            '    ',
            '    # Handle format conversion',
            '    send_as_html = False',
            '    final_body = body',
            '    if format == "markdown":',
            '        final_body = _markdown_to_html(body)',
            '        send_as_html = True',
            '    elif format == "html":',
            '        send_as_html = True',
            '    ',
            '    # Create message',
            '    msg = MIMEMultipart()',
            '    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"',
            '    msg["To"] = to',
            '    msg["Subject"] = subject',
            '    msg.attach(MIMEText(final_body, "html" if send_as_html else "plain"))',
            '    ',
            '    # Attach DataFrame if provided',
            '    if df is not None:',
            '        attachment_content = df.to_csv(index=False)',
            '        part = MIMEBase("text", "csv")',
            '        part.set_payload(attachment_content.encode("utf-8"))',
            '        encoders.encode_base64(part)',
            '        part.add_header("Content-Disposition", f"attachment; filename={attachment_name}")',
            '        msg.attach(part)',
            '    ',
            '    # Send email',
            '    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:',
            '        if SMTP_USE_TLS:',
            '            server.starttls()',
            '        server.login(SMTP_USER, SMTP_PASSWORD)',
            '        recipients = [addr.strip() for addr in to.split(",")]',
            '        server.sendmail(SMTP_FROM, recipients, msg.as_string())',
            '    ',
            '    print(f"Email sent to {to}")',
            '    return True',
            '',
        ])

        # Add API helper functions if there are APIs configured
        if apis:
            script_lines.extend([
                '',
                '# ============================================================================',
                '# API Helper Functions',
                '# ============================================================================',
                '',
                'import requests',
                '',
            ])

            for api in apis:
                api_name = api['name']
                api_type = api['type']
                api_url_var = f"API_{api_name.upper()}_URL"

                if api_type == 'graphql':
                    # GraphQL API helper
                    script_lines.extend([
                        f'def api_{api_name}(query: str, variables: dict = None) -> dict:',
                        f'    """Execute GraphQL query against {api_name} API.',
                        '    ',
                        '    Args:',
                        '        query: GraphQL query string',
                        '        variables: Optional dict of query variables',
                        '    ',
                        '    Returns:',
                        '        Response JSON data',
                        '    """',
                        f'    response = requests.post(',
                        f'        {api_url_var},',
                        '        json={"query": query, "variables": variables or {}},',
                        '        headers={"Content-Type": "application/json"}',
                        '    )',
                        '    response.raise_for_status()',
                        '    result = response.json()',
                        '    if "errors" in result:',
                        '        raise ValueError(f"GraphQL error: {result[\'errors\']}")',
                        '    return result.get("data", result)',
                        '',
                        '',
                    ])
                else:
                    # REST/OpenAPI helper
                    script_lines.extend([
                        f'def api_{api_name}(method_path: str, params: dict = None, body: dict = None) -> dict:',
                        f'    """Make REST API call to {api_name}.',
                        '    ',
                        '    Args:',
                        '        method_path: HTTP method and path, e.g., "GET /breeds" or "POST /users"',
                        '        params: Optional query parameters dict',
                        '        body: Optional request body dict (for POST/PUT/PATCH)',
                        '    ',
                        '    Returns:',
                        '        Response JSON data',
                        '    """',
                        '    parts = method_path.split(" ", 1)',
                        '    method = parts[0].upper() if len(parts) > 0 else "GET"',
                        '    path = parts[1] if len(parts) > 1 else "/"',
                        '    ',
                        f'    base_url = {api_url_var}.rstrip("/")',
                        '    if not path.startswith("/"):',
                        '        path = "/" + path',
                        '    url = base_url + path',
                        '    ',
                        '    response = requests.request(',
                        '        method=method,',
                        '        url=url,',
                        '        params=params,',
                        '        json=body if body else None,',
                        '        headers={"Accept": "application/json"}',
                        '    )',
                        '    response.raise_for_status()',
                        '    return response.json()',
                        '',
                        '',
                    ])

        script_lines.extend([
            '',
            '# ============================================================================',
            '# Publish Function (no-op in standalone mode)',
            '# ============================================================================',
            '',
            'def publish(tables: list[str] = None, artifacts: list[str] = None) -> None:',
            '    """Publish tables/artifacts (no-op in standalone mode)."""',
            '    if tables:',
            '        print(f"Would publish tables: {tables}")',
            '    if artifacts:',
            '        print(f"Would publish artifacts: {artifacts}")',
            '',
            '# ============================================================================',
            '# Step Functions',
            '# ============================================================================',
            '',
        ])

        # Add step functions
        for step in steps:
            step_num = step.get("step_number", 0)
            goal = step.get("goal", "Unknown goal")
            code = step.get("code", "pass")

            script_lines.append(f'def step_{step_num}(facts: dict):')
            script_lines.append(f'    """Step {step_num}: {goal}"""')

            # Indent the code properly
            for line in code.split('\n'):
                if line.strip():
                    script_lines.append(f'    {line}')
                else:
                    script_lines.append('')

            script_lines.append('')

        # Build run_analysis function with explicit fact arguments
        script_lines.append('# ============================================================================')
        script_lines.append('# Main Analysis Function')
        script_lines.append('# ============================================================================')
        script_lines.append('')

        # Build function signature with explicit args
        if facts_list:
            args_with_types = ", ".join(f'{f["name"]}: str' for f in facts_list)
            script_lines.append(f'def run_analysis({args_with_types}):')
            script_lines.append('    """')
            script_lines.append('    Run the complete analysis with the given facts.')
            script_lines.append('')
            script_lines.append('    Args:')
            for fact in facts_list:
                desc = fact["description"] or "No description"
                script_lines.append(f'        {fact["name"]}: {desc}')
            script_lines.append('    """')
            # Build facts dict from explicit args
            script_lines.append('    facts = {')
            for fact in facts_list:
                script_lines.append(f'        "{fact["name"]}": {fact["name"]},')
            script_lines.append('    }')
        else:
            script_lines.append('def run_analysis():')
            script_lines.append('    """Run the complete analysis."""')
            script_lines.append('    facts = {}')

        script_lines.append('')
        for step in steps:
            step_num = step.get("step_number", 0)
            goal = step.get("goal", "Unknown goal")
            script_lines.append(f'    print("\\n=== Step {step_num}: {goal} ===")')
            script_lines.append(f'    step_{step_num}(facts)')
        script_lines.append('')

        # Add main function that loads facts and calls run_analysis
        script_lines.append('')
        script_lines.append('# ============================================================================')
        script_lines.append('# Main Entry Point')
        script_lines.append('# ============================================================================')
        script_lines.append('')
        script_lines.append('def main():')
        script_lines.append('    """Load facts from _facts.parquet and run analysis."""')

        if facts_list:
            # Generate the facts table schema comment
            script_lines.append('    # Expected _facts.parquet schema:')
            script_lines.append('    #   name (str)         | value (str)')
            script_lines.append('    #   -------------------+' + '-' * 40)
            for fact in facts_list:
                desc = fact["description"][:35] + "..." if len(fact.get("description", "")) > 38 else fact.get("description", "")
                script_lines.append(f'    #   {fact["name"]:<18} | {desc}')
            script_lines.append('    #')
            script_lines.append('    facts_df = pd.read_parquet("_facts.parquet")')
            script_lines.append('    facts = dict(zip(facts_df["name"], facts_df["value"]))')
            script_lines.append('')
            # Call run_analysis with explicit args from facts dict
            args_from_dict = ", ".join(f'{f["name"]}=facts["{f["name"]}"]' for f in facts_list)
            script_lines.append(f'    run_analysis({args_from_dict})')
        else:
            script_lines.append('    run_analysis()')

        script_lines.extend([
            '',
            '',
            'if __name__ == "__main__":',
            '    main()',
            '',
        ])

        script_content = '\n'.join(script_lines)

        return Response(
            content=script_content,
            media_type="text/x-python",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_code.py"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading code: {e}")
        raise HTTPException(status_code=500, detail=str(e))
