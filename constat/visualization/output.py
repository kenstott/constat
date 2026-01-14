"""Output helper for saving files and visualizations.

Saves outputs in two ways:
1. To disk for immediate CLI access (file:// URIs users can click)
2. As artifacts in datastore for React UI to display

Usage in generated code:
    # Save a document (markdown, text, etc.)
    viz.save_file('quarterly_report', content, ext='md', title='Q4 Report')

    # Save a folium map
    import folium
    m = folium.Map(location=[50, 10], zoom_start=4)
    viz.save_map('euro_countries', m, title='Countries Using Euro')

    # Save a plotly chart
    import plotly.express as px
    fig = px.bar(df, x='country', y='population')
    viz.save_chart('population_chart', fig, title='Population by Country')

    # Save raw HTML
    viz.save_html('custom_viz', html_string, title='Custom Visualization')
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from constat.storage.datastore import DataStore


def is_repl_mode() -> bool:
    """Check if running in REPL mode."""
    return os.environ.get("CONSTAT_REPL_MODE", "").lower() in ("1", "true", "yes")


# Global to collect outputs during REPL execution
_pending_outputs: list[dict] = []


def clear_pending_outputs() -> None:
    """Clear the pending outputs list."""
    global _pending_outputs
    _pending_outputs = []


def get_pending_outputs() -> list[dict]:
    """Get and clear pending outputs for display."""
    global _pending_outputs
    outputs = _pending_outputs[:]
    _pending_outputs = []
    return outputs


def add_pending_output(file_uri: str, description: str, file_type: str = "") -> None:
    """Add an output to the pending list for later display."""
    _pending_outputs.append({
        "file_uri": file_uri,
        "description": description,
        "type": file_type,
    })


# File extension to artifact type mapping
FILE_EXT_ARTIFACT_TYPES = {
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "csv": "csv",
    "json": "json",
    "xml": "xml",
    "yaml": "yaml",
    "yml": "yaml",
}


class VisualizationHelper:
    """Helper for saving files and visualizations.

    Provides a simple interface for generated code to save files and
    interactive visualizations. Files are saved to an output directory
    and also registered as artifacts in the datastore for the React UI.

    Attributes:
        output_dir: Directory where files are saved
        datastore: DataStore for artifact registration (optional)
        print_file_refs: Whether to print file:// URIs (True for CLI, False for React UI)
        session_id: Session ID for organizing outputs
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        datastore: Optional["DataStore"] = None,
        step_number: int = 0,
        print_file_refs: bool = True,
        session_id: Optional[str] = None,
    ):
        """Initialize the output helper.

        Args:
            output_dir: Directory for saving files. Defaults to ~/.constat/outputs/<session_id>/
            datastore: DataStore for registering artifacts (for React UI)
            step_number: Current step number for artifact metadata
            print_file_refs: If True, print file:// URIs for CLI users.
                           If False, suppress file references (for React UI where
                           artifacts are displayed directly).
            session_id: Session ID for organizing outputs by session
        """
        self.session_id = session_id

        if output_dir is None:
            base_dir = Path.home() / ".constat" / "outputs"
            if session_id:
                # Use session-specific subdirectory
                output_dir = base_dir / session_id[:20]  # Truncate for filesystem
            else:
                output_dir = base_dir

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datastore = datastore
        self.step_number = step_number
        self.print_file_refs = print_file_refs

    def _generate_filename(self, name: str, extension: str) -> Path:
        """Generate a unique filename for the output."""
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self.output_dir / f"{safe_name}.{extension}"

    def _file_uri(self, path: Path) -> str:
        """Convert a path to a file:// URI for clickable terminal links."""
        return path.resolve().as_uri()

    def _print_ref(self, label: str, filepath: Path, description: str = "") -> None:
        """Print file reference or collect for later display.

        In REPL mode, outputs are collected and displayed in an "Outputs:" section
        at the end of execution rather than printed inline.
        """
        if not self.print_file_refs:
            return

        file_uri = self._file_uri(filepath)
        desc = description or label

        if is_repl_mode():
            # Collect for later display in REPL
            add_pending_output(file_uri, desc, filepath.suffix.lstrip("."))
        else:
            # Direct print for non-REPL contexts
            print(f"{label}: {file_uri}")

    def save_file(
        self,
        name: str,
        content: str,
        ext: str = "txt",
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        """Save any text file to disk and artifact store.

        Use this for documents, reports, data exports, etc.

        Args:
            name: Name for the file (used in filename and artifact)
            content: Text content to save
            ext: File extension (md, txt, csv, json, etc.)
            title: Human-readable title for the artifact
            description: Description of the file

        Returns:
            Path to the saved file
        """
        # Normalize extension
        ext = ext.lstrip(".")

        # Save to file
        filepath = self._generate_filename(name, ext)
        filepath.write_text(content, encoding="utf-8")

        # Register as artifact if datastore available
        if self.datastore:
            try:
                artifact_type = FILE_EXT_ARTIFACT_TYPES.get(ext, "text")
                self.datastore.save_rich_artifact(
                    name=name,
                    artifact_type=artifact_type,
                    content=content,
                    step_number=self.step_number,
                    title=title or name,
                    description=description,
                )
            except Exception as e:
                # Don't fail if artifact save fails - file is already saved
                if self.print_file_refs:
                    print(f"Note: Could not register artifact: {e}")

        self._print_ref(f"File saved ({ext})", filepath, description=title or name)
        return filepath

    def save_html(
        self,
        name: str,
        html_content: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        """Save raw HTML content to file and artifact store.

        Args:
            name: Name for the visualization (used in filename and artifact)
            html_content: HTML string to save
            title: Human-readable title for the artifact
            description: Description of the visualization

        Returns:
            Path to the saved HTML file
        """
        # Save to file
        filepath = self._generate_filename(name, "html")
        filepath.write_text(html_content, encoding="utf-8")

        # Register as artifact if datastore available
        if self.datastore:
            try:
                self.datastore.save_html(
                    name=name,
                    html_content=html_content,
                    step_number=self.step_number,
                    title=title or name,
                    description=description,
                )
            except Exception as e:
                # Don't fail if artifact save fails - file is already saved
                if self.print_file_refs:
                    print(f"Note: Could not register artifact: {e}")

        self._print_ref("HTML", filepath, description=title or name)
        return filepath

    def save_map(
        self,
        name: str,
        folium_map: Any,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        """Save a folium map to file and artifact store.

        Args:
            name: Name for the map (used in filename and artifact)
            folium_map: Folium Map object
            title: Human-readable title for the artifact
            description: Description of the map

        Returns:
            Path to the saved HTML file
        """
        # Get HTML from folium map
        html_content = folium_map._repr_html_()

        # Save to file
        filepath = self._generate_filename(name, "html")
        folium_map.save(str(filepath))

        # Register as artifact if datastore available
        if self.datastore:
            try:
                self.datastore.save_html(
                    name=name,
                    html_content=html_content,
                    step_number=self.step_number,
                    title=title or f"Map: {name}",
                    description=description or "Interactive map visualization",
                )
            except Exception as e:
                if self.print_file_refs:
                    print(f"Note: Could not register artifact: {e}")

        self._print_ref("Map", filepath, description=title or f"Map: {name}")
        return filepath

    def save_chart(
        self,
        name: str,
        figure: Any,
        title: Optional[str] = None,
        description: Optional[str] = None,
        chart_type: str = "plotly",
    ) -> Path:
        """Save a Plotly or Altair chart to file and artifact store.

        Args:
            name: Name for the chart (used in filename and artifact)
            figure: Plotly Figure or Altair Chart object
            title: Human-readable title for the artifact
            description: Description of the chart
            chart_type: Type of chart ("plotly" or "altair")

        Returns:
            Path to the saved HTML file
        """
        filepath = self._generate_filename(name, "html")

        # Handle different chart libraries
        if chart_type == "plotly" or hasattr(figure, "write_html"):
            # Plotly figure
            figure.write_html(str(filepath), include_plotlyjs=True, full_html=True)
            html_content = filepath.read_text(encoding="utf-8")

            # Also save chart spec as artifact
            if self.datastore:
                try:
                    spec = json.loads(figure.to_json())
                    self.datastore.save_chart(
                        name=name,
                        spec=spec,
                        step_number=self.step_number,
                        title=title or name,
                        chart_type="plotly",
                    )
                except Exception as e:
                    if self.print_file_refs:
                        print(f"Note: Could not register chart artifact: {e}")

        elif chart_type == "altair" or hasattr(figure, "save"):
            # Altair chart
            figure.save(str(filepath))
            html_content = filepath.read_text(encoding="utf-8")

            # Also save chart spec as artifact
            if self.datastore:
                try:
                    spec = figure.to_dict()
                    self.datastore.save_chart(
                        name=name,
                        spec=spec,
                        step_number=self.step_number,
                        title=title or name,
                        chart_type="vega-lite",
                    )
                except Exception as e:
                    if self.print_file_refs:
                        print(f"Note: Could not register chart artifact: {e}")
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        self._print_ref("Chart", filepath, description=title or name)
        return filepath

    def save_image(
        self,
        name: str,
        figure: Any,
        format: str = "png",
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        """Save a matplotlib figure or image to file.

        Args:
            name: Name for the image
            figure: Matplotlib figure or image data
            format: Image format (png, svg, jpg)
            title: Human-readable title
            description: Description of the image

        Returns:
            Path to the saved image file
        """
        filepath = self._generate_filename(name, format)

        # Handle matplotlib figures
        if hasattr(figure, "savefig"):
            figure.savefig(str(filepath), format=format, bbox_inches="tight", dpi=150)
        else:
            # Assume raw bytes
            filepath.write_bytes(figure)

        # Register as artifact if datastore available
        if self.datastore and format in ("png", "svg", "jpg", "jpeg"):
            try:
                image_data = filepath.read_bytes()
                from constat.core.models import ArtifactType
                artifact_type = {
                    "png": ArtifactType.PNG,
                    "svg": ArtifactType.SVG,
                    "jpg": ArtifactType.JPEG,
                    "jpeg": ArtifactType.JPEG,
                }.get(format, ArtifactType.PNG)

                self.datastore.save_image(
                    name=name,
                    image_data=image_data,
                    image_format=format,
                    step_number=self.step_number,
                    title=title or name,
                )
            except Exception as e:
                if self.print_file_refs:
                    print(f"Note: Could not register image artifact: {e}")

        self._print_ref("Image", filepath, description=title or name)
        return filepath


def create_viz_helper(
    datastore: Optional["DataStore"] = None,
    output_dir: Optional[Path] = None,
    step_number: int = 0,
    print_file_refs: bool = True,
    session_id: Optional[str] = None,
) -> VisualizationHelper:
    """Create a VisualizationHelper instance.

    Factory function for use in execution globals.

    Args:
        datastore: DataStore for artifact registration
        output_dir: Custom output directory
        step_number: Current step number
        print_file_refs: If True, print file:// URIs (CLI mode).
                        If False, suppress (React UI mode).
        session_id: Session ID for organizing outputs by session

    Returns:
        Configured VisualizationHelper instance
    """
    return VisualizationHelper(
        output_dir=output_dir,
        datastore=datastore,
        step_number=step_number,
        print_file_refs=print_file_refs,
        session_id=session_id,
    )
