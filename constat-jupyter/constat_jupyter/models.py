from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import itables
    HAS_ITABLES = True
    # Configure iTables defaults: left-aligned, compact buttons
    itables.options.style = "table-layout:auto;width:auto;margin-left:0;caption-side:bottom"
    itables.options.layout = {
        "topStart": "buttons",
        "topEnd": "search",
        "bottomStart": "info",
        "bottomEnd": "paging",
    }
    itables.options.classes = "display nowrap compact"
except ImportError:
    HAS_ITABLES = False


class ConstatError(Exception):
    """Error from the Constat server."""


@dataclass
class StepInfo:
    number: int
    goal: str
    status: str
    duration_ms: int | None = None
    error: str | None = None


@dataclass
class Artifact:
    id: int
    name: str
    artifact_type: str
    content: str | None = None
    mime_type: str | None = None
    is_starred: bool = False

    @staticmethod
    def _highlight_code(code: str, language: str = "") -> str:
        """Syntax-highlight code, falling back to plain <pre> if Pygments unavailable."""
        try:
            from pygments import highlight
            from pygments.formatters import HtmlFormatter
            from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
            try:
                lexer = get_lexer_by_name(language) if language else guess_lexer(code)
            except Exception:
                lexer = TextLexer()
            formatter = HtmlFormatter(nowrap=False, noclasses=True, style="default")
            return highlight(code, lexer, formatter)
        except ImportError:
            import html as _html
            return f"<pre><code>{_html.escape(code)}</code></pre>"

    def display(self) -> None:
        """Render artifact inline using IPython display system."""
        from IPython.display import display, HTML, Markdown, Image, SVG

        if self.content is None:
            raise ConstatError(f"Artifact '{self.name}' has no content. Fetch with session.artifact({self.id}).")

        atype = self.artifact_type.lower()

        if atype == "plotly":
            try:
                import plotly.io as pio
                fig = pio.from_json(self.content)
                fig.show()
            except ImportError:
                display(HTML(self.content))
        elif atype in ("png", "jpeg", "gif", "webp"):
            import base64
            display(Image(data=base64.b64decode(self.content)))
        elif atype == "svg":
            display(SVG(data=self.content))
        elif atype == "html":
            display(HTML(self.content))
        elif atype in ("markdown", "md"):
            display(Markdown(self.content))
        elif atype == "table":
            import json as _json
            try:
                data = _json.loads(self.content)
                import pandas as pd
                df = pd.DataFrame(data)
                if HAS_ITABLES:
                    itables.show(df, buttons=["csvHtml5", "excelHtml5"])
                else:
                    display(HTML(df.to_html()))
            except (ValueError, TypeError):
                display(HTML(f"<pre>{self.content[:500]}</pre>"))
        elif atype == "csv":
            import io as _io
            try:
                import pandas as pd
                df = pd.read_csv(_io.StringIO(self.content))
                if HAS_ITABLES:
                    itables.show(df, buttons=["csvHtml5", "excelHtml5"])
                else:
                    display(HTML(df.to_html()))
            except Exception:
                display(HTML(f"<pre>{self.content}</pre>"))
        elif atype == "json":
            display(HTML(self._highlight_code(self.content, "json")))
        elif atype == "sql":
            display(HTML(self._highlight_code(self.content, "sql")))
        elif atype == "python":
            display(HTML(self._highlight_code(self.content, "python")))
        elif atype == "code":
            display(HTML(self._highlight_code(self.content)))
        elif atype in ("output", "text"):
            display(HTML(f"<pre>{self.content}</pre>"))
        elif atype == "error":
            display(HTML(f"<pre style='color:red'>{self.content}</pre>"))
        else:
            display(HTML(f"<pre>{self.content}</pre>"))

    def _repr_html_(self) -> str:
        atype = self.artifact_type.lower()
        if self.content and atype == "html":
            return self.content
        if self.content and atype in ("markdown", "md"):
            return f"<pre>{self.content}</pre>"
        return f"<p>Artifact: {self.name} ({self.artifact_type})</p>"


@dataclass
class SolveResult:
    success: bool
    answer: str
    tables: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    steps: list[StepInfo] = field(default_factory=list)
    error: str | None = None
    raw_output: str | None = None
    _session: Any = field(default=None, repr=False)

    @staticmethod
    def _strip_trailing_json(text: str) -> str:
        """Remove JSON arrays/objects from answer text."""
        import json as _json
        lines = text.split('\n')
        kept = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            # Try to detect JSON arrays/objects
            if (stripped.startswith('[{') or stripped.startswith('{"')) and (
                stripped.endswith('}]') or stripped.endswith('}')
            ):
                try:
                    _json.loads(stripped)
                    continue  # Valid JSON — skip it
                except _json.JSONDecodeError:
                    pass
            kept.append(line)
        # Collapse multiple blank lines
        import re
        result = '\n'.join(kept)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    @staticmethod
    def _to_pandas(df: Any):
        """Convert to pandas DataFrame."""
        return df.to_pandas() if hasattr(df, "to_pandas") else df

    @staticmethod
    def _df_to_html(df: Any) -> str:
        """Render a DataFrame as HTML, using iTables when available."""
        if HAS_ITABLES:
            return itables.to_html_datatable(
                SolveResult._to_pandas(df),
                buttons=["csvHtml5", "excelHtml5"],
            )
        return f"<pre>{df}</pre>"

    def _repr_html_(self) -> str:
        """Auto-display answer + tables when result is last expression in cell."""
        parts = []
        answer = self._strip_trailing_json(self.answer)
        if self.error:
            parts.append(f"<p><strong>Error:</strong> {self.error}</p>")
        else:
            parts.append(f"<div>{answer}</div>")
        for name, df in self.tables.items():
            parts.append(f"<h4>{name}</h4>")
            parts.append(self._df_to_html(df))
        return "\n".join(parts)

    def _repr_markdown_(self) -> str:
        if self.error:
            return f"**Error:** {self.error}"
        return self._strip_trailing_json(self.answer)

    def _display_table(self, name: str, df: Any) -> None:
        """Display a single table, using iTables when available."""
        from IPython.display import display, HTML
        if HAS_ITABLES:
            display(HTML(f"<h4>{name}</h4>"))
            itables.show(
                SolveResult._to_pandas(df),
                buttons=["csvHtml5", "excelHtml5"],
            )
        else:
            print(f"\n--- {name} ---")
            print(df)

    _DEFAULT_HIDDEN = frozenset(("code", "sql", "python", "output", "text", "error"))
    # Normalize aliases so users can write "md" or "markdown"
    _TYPE_ALIASES = {"md": "markdown", "markdown": "markdown"}

    @classmethod
    def _normalize_types(cls, types: set[str] | None) -> set[str] | None:
        if types is None:
            return None
        return {cls._TYPE_ALIASES.get(t, t) for t in types}

    def _should_show_artifact(
        self,
        artifact: "Artifact",
        *,
        skip_tables: set[str],
        include_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
    ) -> bool:
        atype = self._TYPE_ALIASES.get(artifact.artifact_type.lower(), artifact.artifact_type.lower())
        # Duplicate table artifacts already rendered as DataFrames
        if atype == "table" and artifact.name in skip_tables:
            return False
        # Explicit whitelist takes priority
        if include_types is not None:
            return atype in include_types
        # Explicit blacklist
        if exclude_types is not None:
            return atype not in exclude_types
        # Default: hide code and step-output types
        return atype not in self._DEFAULT_HIDDEN

    def _should_show_tables(
        self, include_types: set[str] | None, exclude_types: set[str] | None,
    ) -> bool:
        """Whether DataFrame tables should be rendered."""
        if include_types is not None:
            return "table" in include_types
        if exclude_types is not None:
            return "table" not in exclude_types
        return True

    def display(
        self,
        published: bool = False,
        show_code: bool = False,
        show_output: bool = False,
        include_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
    ) -> None:
        """Rich display: answer + new tables + new artifacts.

        Only shows tables/artifacts from the current query (self.tables,
        self.artifacts), not accumulated results from previous queries.

        Args:
            published: If True, only show starred items among the new results.
            show_code: If True, also display code/sql artifacts (hidden by default).
            show_output: If True, also display step output/text/error artifacts (hidden by default).
            include_types: Whitelist — only show these types (applies to both tables
                and artifacts). Use ``{"md"}`` to show only markdown artifacts.
            exclude_types: Blacklist — hide these types.
        """
        from IPython.display import display, Markdown

        include_types = self._normalize_types(include_types)
        exclude_types = self._normalize_types(exclude_types)

        # show_code / show_output modify exclude set when no explicit filter given
        if include_types is None and exclude_types is None:
            if show_code or show_output:
                exclude_types = set(self._DEFAULT_HIDDEN)
                if show_code:
                    exclude_types -= {"code", "sql", "python"}
                if show_output:
                    exclude_types -= {"output", "text", "error"}

        show_tables = self._should_show_tables(include_types, exclude_types)
        filter_kw = dict(include_types=include_types, exclude_types=exclude_types)

        display(Markdown(self._strip_trailing_json(self.answer)))

        if published:
            starred: set[str] = set()
            if self._session is not None:
                try:
                    starred = {t["name"] for t in self._session.tables() if t.get("is_starred")}
                except Exception:
                    pass
            if show_tables:
                for name, df in self.tables.items():
                    if name in starred:
                        self._display_table(name, df)
            for artifact in self.artifacts:
                if not artifact.is_starred:
                    continue
                if self._should_show_artifact(artifact, skip_tables=starred if show_tables else set(), **filter_kw):
                    artifact.display()
        else:
            shown_tables: set[str] = set()
            if show_tables:
                shown_tables = set(self.tables.keys())
                for name, df in self.tables.items():
                    self._display_table(name, df)
            for artifact in self.artifacts:
                if self._should_show_artifact(artifact, skip_tables=shown_tables, **filter_kw):
                    artifact.display()
