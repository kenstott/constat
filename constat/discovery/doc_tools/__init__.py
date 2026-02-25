# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document discovery tools for reference documents.

These tools allow the LLM to discover and search reference documents
on-demand rather than loading everything into the system prompt.
"""

from ._access import _AccessMixin
from ._core import _CoreMixin
from ._entities import _EntityMixin
from ._schema_inference import (
    _is_glob_pattern,
    _expand_file_paths,
    _infer_csv_schema,
    _infer_json_schema,
    _infer_column_type,
    _infer_structured_schema,
    _infer_jsonl_schema,
)
from ._schemas import DOC_TOOL_SCHEMAS


class DocumentDiscoveryTools(_CoreMixin, _EntityMixin, _AccessMixin):
    """Tools for discovering and searching reference documents on-demand.

    Supports incremental updates - only reloads documents that have changed
    based on file modification times and content hashes.
    """
    pass


__all__ = [
    "DocumentDiscoveryTools",
    "DOC_TOOL_SCHEMAS",
    "_is_glob_pattern",
    "_expand_file_paths",
    "_infer_csv_schema",
    "_infer_json_schema",
    "_infer_column_type",
    "_infer_structured_schema",
    "_infer_jsonl_schema",
]
