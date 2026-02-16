# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data access REST endpoints (tables, artifacts, facts)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _sanitize_value(v: Any) -> Any:
    """Convert a single value to a JSON-safe Python type."""
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_sanitize_value(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(dk): _sanitize_value(dv) for dk, dv in v.items()}
    if isinstance(v, (np.str_, np.bytes_)):
        return str(v)
    if hasattr(v, 'item'):
        return v.item()
    return v


def _sanitize_df_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-safe list of dicts.

    Handles NaN, NaT, numpy types, ndarray columns that break Pydantic JSON serialization.
    """
    # noinspection PyTypeChecker
    df = df.where(df.notna(), None)
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            row[k] = _sanitize_value(v)
    return records


from constat.server.routes.data.tables import router as tables_router
from constat.server.routes.data.artifacts import router as artifacts_router
from constat.server.routes.data.facts import router as facts_router
from constat.server.routes.data.entities import router as entities_router
from constat.server.routes.data.scripts import router as scripts_router
from constat.server.routes.data.inference import router as inference_router
from constat.server.routes.data.inference import generate_inference_script, _gather_source_configs
from constat.server.routes.data.glossary import router as glossary_router

router = APIRouter()
router.include_router(tables_router)
router.include_router(artifacts_router)
router.include_router(facts_router)
router.include_router(entities_router)
router.include_router(scripts_router)
router.include_router(inference_router)
router.include_router(glossary_router)
