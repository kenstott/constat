from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        import base64

        if self.content is None:
            raise ConstatError(f"Artifact '{self.name}' has no content. Fetch with session.artifact({self.id}).")

        if self.artifact_type == "PLOTLY":
            try:
                import plotly.io as pio
                fig = pio.from_json(self.content)
                fig.show()
                return
            except ImportError:
                display(HTML(self.content))
        elif self.artifact_type in ("PNG", "JPEG"):
            display(Image(data=base64.b64decode(self.content)))
        elif self.artifact_type == "HTML":
            display(HTML(self.content))
        elif self.artifact_type == "MARKDOWN":
            display(Markdown(self.content))
        elif self.artifact_type == "TABLE":
            import polars
            from io import StringIO
            display(polars.read_csv(StringIO(self.content)))
        else:
            print(self.content)

    def _repr_html_(self) -> str:
        if self.content and self.artifact_type == "HTML":
            return self.content
        if self.content and self.artifact_type == "MARKDOWN":
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
        """Remove trailing JSON array/object from answer text."""
        import re
        # Strip trailing JSON array like [{...}, {...}]
        return re.sub(r'\n\s*\[\s*\{[\s\S]*\}\s*\]\s*$', '', text)

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
            if hasattr(df, '_repr_html_'):
                parts.append(df._repr_html_())
            else:
                parts.append(f"<pre>{df}</pre>")
        return "\n".join(parts)

    def _repr_markdown_(self) -> str:
        if self.error:
            return f"**Error:** {self.error}"
        return self._strip_trailing_json(self.answer)

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
                    print(f"\n--- {name} ---")
                    df = self.tables[name]
                    display(df) if hasattr(df, '_repr_html_') else print(df)

            # Only starred artifacts (skip virtual table artifacts with negative IDs)
            for a in self._session.artifacts():
                if a.id < 0 or not a.is_starred:
                    continue
                full = self._session.artifact(a.id)
                full.display()
        else:
            display(Markdown(self._strip_trailing_json(self.answer)))
            for name, df in self.tables.items():
                print(f"\n--- {name} ---")
                display(df) if hasattr(df, '_repr_html_') else print(df)
            for artifact in self.artifacts:
                artifact.display()
