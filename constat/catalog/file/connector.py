"""File-based data source connector.

Provides schema introspection for file-based data sources (CSV, JSON, Parquet, Arrow)
with the same interface pattern as SQL and NoSQL connectors.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import glob as glob_module


class FileType(Enum):
    """Type of file-based data source."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    ARROW = "arrow"
    FEATHER = "feather"  # Alias for arrow


@dataclass
class ColumnInfo:
    """Information about a column in a file-based data source."""
    name: str
    data_type: str
    nullable: bool = True
    sample_values: list[Any] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class FileMetadata:
    """Metadata for a file-based data source."""
    name: str  # Logical name
    path: str  # File path (local, s3://, https://, etc.)
    file_type: FileType
    columns: list[ColumnInfo]
    row_count: int = 0
    size_bytes: int = 0
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.file_type.value,
            "columns": [
                {
                    "name": c.name,
                    "type": c.data_type,
                    "nullable": c.nullable,
                    "sample_values": c.sample_values[:5],
                }
                for c in self.columns
            ],
            "row_count": self.row_count,
            "description": self.description,
        }

    def to_embedding_text(self) -> str:
        """Generate text for vector embedding (schema discovery)."""
        lines = [
            f"File: {self.name}",
            f"Path: {self.path}",
            f"Format: {self.file_type.value}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.row_count:
            lines.append(f"Rows: {self.row_count:,}")

        if self.columns:
            col_strs = []
            for c in self.columns:
                col_str = f"{c.name} ({c.data_type})"
                if c.sample_values:
                    samples = [str(v) for v in c.sample_values[:3]]
                    col_str += f": {samples}"
                col_strs.append(col_str)
            lines.append(f"Columns: {', '.join(col_strs)}")

        return "\n".join(lines)


class FileConnector:
    """Connector for file-based data sources.

    Provides schema introspection for CSV, JSON, Parquet, and Arrow files.
    Supports local paths and remote URLs (s3://, https://, etc.).
    """

    # Number of rows to sample for schema inference
    SAMPLE_SIZE = 100

    def __init__(
        self,
        name: str,
        path: str,
        file_type: FileType,
        description: str = "",
        sample_size: int = SAMPLE_SIZE,
    ):
        self.name = name
        self.path = path
        self.file_type = file_type
        self.description = description
        self.sample_size = sample_size
        self._metadata: Optional[FileMetadata] = None

    @classmethod
    def from_config(cls, name: str, db_config) -> "FileConnector":
        """Create a FileConnector from a DatabaseConfig."""
        file_type = FileType(db_config.type)
        return cls(
            name=name,
            path=db_config.path or "",
            file_type=file_type,
            description=db_config.description,
            sample_size=db_config.sample_size,
        )

    def get_metadata(self) -> FileMetadata:
        """Get metadata for the file, inferring schema if needed."""
        if self._metadata is None:
            self._metadata = self._infer_schema()
        return self._metadata

    def _infer_schema(self) -> FileMetadata:
        """Infer schema from the file."""
        if self.file_type == FileType.CSV:
            return self._infer_csv_schema()
        elif self.file_type in (FileType.JSON, FileType.JSONL):
            return self._infer_json_schema()
        elif self.file_type in (FileType.PARQUET, FileType.ARROW, FileType.FEATHER):
            return self._infer_parquet_arrow_schema()
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def _infer_csv_schema(self) -> FileMetadata:
        """Infer schema from a CSV file."""
        import pandas as pd

        # Read sample for schema inference
        try:
            df = pd.read_csv(self.path, nrows=self.sample_size)
        except Exception as e:
            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=[],
                description=f"Error reading file: {e}",
            )

        columns = self._infer_columns_from_dataframe(df)

        # Get full row count (more efficient than reading all data)
        try:
            # For local files, count lines
            if not self._is_remote_path():
                with open(self.path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
            else:
                row_count = len(df)  # Use sample size for remote
        except Exception:
            row_count = len(df)

        return FileMetadata(
            name=self.name,
            path=self.path,
            file_type=self.file_type,
            columns=columns,
            row_count=row_count,
            description=self.description,
        )

    def _infer_json_schema(self) -> FileMetadata:
        """Infer schema from a JSON or JSONL file."""
        import pandas as pd

        try:
            if self.file_type == FileType.JSONL:
                df = pd.read_json(self.path, lines=True, nrows=self.sample_size)
            else:
                df = pd.read_json(self.path)
                if len(df) > self.sample_size:
                    df = df.head(self.sample_size)
        except Exception as e:
            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=[],
                description=f"Error reading file: {e}",
            )

        columns = self._infer_columns_from_dataframe(df)

        # Get row count
        try:
            if self.file_type == FileType.JSONL and not self._is_remote_path():
                with open(self.path, 'r') as f:
                    row_count = sum(1 for line in f if line.strip())
            else:
                # For regular JSON, re-read to get full count
                full_df = pd.read_json(
                    self.path,
                    lines=(self.file_type == FileType.JSONL)
                )
                row_count = len(full_df)
        except Exception:
            row_count = len(df)

        return FileMetadata(
            name=self.name,
            path=self.path,
            file_type=self.file_type,
            columns=columns,
            row_count=row_count,
            description=self.description,
        )

    def _infer_parquet_arrow_schema(self) -> FileMetadata:
        """Infer schema from a Parquet or Arrow file."""
        try:
            import pyarrow.parquet as pq
            import pyarrow.feather as feather
            import pyarrow as pa
        except ImportError:
            # Fall back to pandas
            return self._infer_parquet_arrow_schema_pandas()

        try:
            if self.file_type == FileType.PARQUET:
                # Use PyArrow for efficient metadata reading
                parquet_file = pq.ParquetFile(self.path)
                schema = parquet_file.schema_arrow
                row_count = parquet_file.metadata.num_rows

                # Read sample for values
                table = parquet_file.read_row_groups([0])
                if table.num_rows > self.sample_size:
                    table = table.slice(0, self.sample_size)
                df = table.to_pandas()
            else:
                # Arrow/Feather
                table = feather.read_table(self.path)
                schema = table.schema
                row_count = table.num_rows

                if table.num_rows > self.sample_size:
                    table = table.slice(0, self.sample_size)
                df = table.to_pandas()

            columns = []
            for i, field in enumerate(schema):
                col_name = field.name
                col_type = self._arrow_type_to_string(field.type)

                # Get sample values from dataframe
                sample_values = []
                if col_name in df.columns:
                    sample_values = df[col_name].dropna().head(5).tolist()

                columns.append(ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    nullable=field.nullable,
                    sample_values=sample_values,
                ))

            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=columns,
                row_count=row_count,
                description=self.description,
            )

        except Exception as e:
            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=[],
                description=f"Error reading file: {e}",
            )

    def _infer_parquet_arrow_schema_pandas(self) -> FileMetadata:
        """Fallback schema inference using pandas."""
        import pandas as pd

        try:
            if self.file_type == FileType.PARQUET:
                df = pd.read_parquet(self.path)
            else:
                df = pd.read_feather(self.path)

            row_count = len(df)
            if len(df) > self.sample_size:
                df = df.head(self.sample_size)

            columns = self._infer_columns_from_dataframe(df)

            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=columns,
                row_count=row_count,
                description=self.description,
            )
        except Exception as e:
            return FileMetadata(
                name=self.name,
                path=self.path,
                file_type=self.file_type,
                columns=[],
                description=f"Error reading file: {e}",
            )

    def _infer_columns_from_dataframe(self, df) -> list[ColumnInfo]:
        """Infer column info from a pandas DataFrame."""
        columns = []
        for col_name in df.columns:
            col = df[col_name]
            dtype_str = self._pandas_dtype_to_string(col.dtype)

            # Get sample values
            sample_values = col.dropna().head(5).tolist()

            # Check nullability
            nullable = col.isna().any()

            columns.append(ColumnInfo(
                name=str(col_name),
                data_type=dtype_str,
                nullable=nullable,
                sample_values=sample_values,
            ))

        return columns

    def _pandas_dtype_to_string(self, dtype) -> str:
        """Convert pandas dtype to string representation."""
        dtype_str = str(dtype)

        if "int" in dtype_str:
            return "integer"
        elif "float" in dtype_str:
            return "float"
        elif "bool" in dtype_str:
            return "boolean"
        elif "datetime" in dtype_str:
            return "datetime"
        elif "timedelta" in dtype_str:
            return "timedelta"
        elif "object" in dtype_str:
            return "string"
        elif "category" in dtype_str:
            return "category"
        else:
            return dtype_str

    def _arrow_type_to_string(self, arrow_type) -> str:
        """Convert PyArrow type to string representation."""
        import pyarrow as pa

        if pa.types.is_integer(arrow_type):
            return "integer"
        elif pa.types.is_floating(arrow_type):
            return "float"
        elif pa.types.is_boolean(arrow_type):
            return "boolean"
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return "string"
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return "binary"
        elif pa.types.is_timestamp(arrow_type):
            return "timestamp"
        elif pa.types.is_date(arrow_type):
            return "date"
        elif pa.types.is_time(arrow_type):
            return "time"
        elif pa.types.is_list(arrow_type):
            return "array"
        elif pa.types.is_struct(arrow_type):
            return "struct"
        elif pa.types.is_map(arrow_type):
            return "map"
        elif pa.types.is_dictionary(arrow_type):
            return "dictionary"
        else:
            return str(arrow_type)

    def _is_remote_path(self) -> bool:
        """Check if the path is a remote URL."""
        return any(self.path.startswith(p) for p in ["s3://", "gs://", "http://", "https://", "ftp://"])

    def get_overview(self) -> str:
        """Generate token-optimized overview for system prompt."""
        meta = self.get_metadata()
        col_names = [c.name for c in meta.columns[:8]]
        cols_str = ", ".join(col_names)
        if len(meta.columns) > 8:
            cols_str += f" (+{len(meta.columns) - 8} more)"

        lines = [
            f"  {self.name} ({self.file_type.value}): {self.path}",
            f"    Columns: {cols_str}",
            f"    Rows: ~{meta.row_count:,}",
        ]

        return "\n".join(lines)

    def get_read_code(self) -> str:
        """Generate pandas code to read this file."""
        if self.file_type == FileType.CSV:
            return f"pd.read_csv('{self.path}')"
        elif self.file_type == FileType.JSON:
            return f"pd.read_json('{self.path}')"
        elif self.file_type == FileType.JSONL:
            return f"pd.read_json('{self.path}', lines=True)"
        elif self.file_type == FileType.PARQUET:
            return f"pd.read_parquet('{self.path}')"
        elif self.file_type in (FileType.ARROW, FileType.FEATHER):
            return f"pd.read_feather('{self.path}')"
        else:
            return f"# Unknown file type: {self.file_type}"

    def load(self, **kwargs):
        """Load the file data as a pandas DataFrame.

        This is the primary method for accessing file data in generated code.
        Additional kwargs are passed to the underlying pandas read function.

        Returns:
            pandas.DataFrame: The file data

        Example:
            df = db_events.load()
            df = db_events.load(nrows=1000)  # Load only first 1000 rows
        """
        import pandas as pd

        if self.file_type == FileType.CSV:
            return pd.read_csv(self.path, **kwargs)
        elif self.file_type == FileType.JSON:
            return pd.read_json(self.path, **kwargs)
        elif self.file_type == FileType.JSONL:
            return pd.read_json(self.path, lines=True, **kwargs)
        elif self.file_type == FileType.PARQUET:
            return pd.read_parquet(self.path, **kwargs)
        elif self.file_type in (FileType.ARROW, FileType.FEATHER):
            return pd.read_feather(self.path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
