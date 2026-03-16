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

    def display(self) -> None:
        """Render artifact inline using IPython display system."""
        from IPython.display import display, HTML, Markdown, Image

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
        elif atype in ("png", "jpeg"):
            import base64
            display(Image(data=base64.b64decode(self.content)))
        elif atype == "html":
            display(HTML(self.content))
        elif atype in ("markdown", "md"):
            display(Markdown(self.content))
        elif atype == "table":
            # Table artifacts contain JSON data — render as DataFrame
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
        elif atype == "json":
            display(HTML(f"<pre>{self.content}</pre>"))
        elif atype == "code":
            display(HTML(f"<pre><code>{self.content}</code></pre>"))
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

    def display(self, published: bool = False) -> None:
        """Rich display: answer + tables + artifacts.

        Args:
            published: If True, only show starred/published tables and artifacts.
        """
        from IPython.display import display, Markdown

        if published:
            if self._session is None:
                raise ConstatError("No session reference.")
            display(Markdown(self._strip_trailing_json(self.answer)))

            # Only starred tables
            table_list = self._session.tables()
            starred = {t["name"] for t in table_list if t.get("is_starred")}
            for name in starred:
                if name in self.tables:
                    self._display_table(name, self.tables[name])

            # Only starred artifacts (skip virtual table artifacts with negative IDs,
            # and skip table artifacts whose data was already rendered above)
            for a in self._session.artifacts():
                if a.id < 0 or not a.is_starred:
                    continue
                if a.artifact_type.lower() == "table" and a.name in starred:
                    continue
                full = self._session.artifact(a.id)
                full.display()
        else:
            display(Markdown(self._strip_trailing_json(self.answer)))
            for name, df in self.tables.items():
                self._display_table(name, df)
            for artifact in self.artifacts:
                artifact.display()
