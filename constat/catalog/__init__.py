# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema and API discovery with vector search and execution."""

from .schema_manager import (
    ColumnMetadata,
    ForeignKey,
    SchemaManager,
    TableMatch,
    TableMetadata,
)
from .api_catalog import (
    APICatalog,
    ArgumentType,
    OperationArgument,
    OperationField,
    OperationMatch,
    OperationMetadata,
    OperationType,
    create_constat_api_catalog,
    introspect_graphql_endpoint,
)
from .api_executor import (
    APIExecutor,
    APIExecutionError,
)

__all__ = [
    # Schema Manager
    "ColumnMetadata",
    "ForeignKey",
    "SchemaManager",
    "TableMatch",
    "TableMetadata",
    # API Catalog
    "APICatalog",
    "ArgumentType",
    "OperationArgument",
    "OperationField",
    "OperationMatch",
    "OperationMetadata",
    "OperationType",
    "create_constat_api_catalog",
    "introspect_graphql_endpoint",
    # API Executor
    "APIExecutor",
    "APIExecutionError",
]
