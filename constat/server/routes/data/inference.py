# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Step code and inference code endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from constat.server.auth import CurrentUserId
from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Step Code Endpoints
# ============================================================================


@router.get("/{session_id}/steps")
async def list_step_codes(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """List all step codes for a session.

    Returns the code executed for each step in the plan, stored on disk.

    Args:
        session_id: Session ID
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        List of step codes with step_number, goal, and code

    Raises:
        404: Session not found
    """
    # Try to get the session from memory first
    # noinspection DuplicatedCode
    managed = session_manager.get_session_or_none(session_id)

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
        logger.debug(f"[list_step_codes] Found managed session. Server: {session_id}, History: {history_session_id}")
    else:
        # Session not in memory - try reverse lookup from disk
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=user_id)
        history_session_id = history.find_session_by_server_id(session_id)
        logger.debug(f"[list_step_codes] Session not in memory. Reverse lookup found: {history_session_id}")

    try:
        steps = history.list_step_codes(history_session_id) if history_session_id else []
        logger.debug(f"[list_step_codes] Found {len(steps)} steps")

        return {
            "steps": steps,
            "total": len(steps),
            # Include session ID info for debugging
            "session_info": {
                "server_session_id": session_id,
                "history_session_id": history_session_id,
            },
        }

    except Exception as e:
        logger.error(f"Error listing step codes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/inference-codes")
async def list_inference_codes(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """List all inference codes for a session (auditable mode)."""
    managed = session_manager.get_session_or_none(session_id)

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
    else:
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=user_id)
        history_session_id = history.find_session_by_server_id(session_id)

    try:
        inferences = history.list_inference_codes(history_session_id) if history_session_id else []
        return {"inferences": inferences, "total": len(inferences)}
    except Exception as e:
        logger.error(f"Error listing inference codes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _gather_source_configs(managed) -> tuple[list[dict], list[dict]]:
    """Extract api and database configs from a session for script generation."""
    apis = []
    if managed and managed.session.config and managed.session.config.apis:
        for name, api_config in managed.session.config.apis.items():
            apis.append({
                "name": name,
                "type": api_config.type,
                "url": api_config.url or "",
            })

    databases = []
    seen_db_names = set()
    if managed and managed.session.config and managed.session.config.databases:
        for name, db_config in managed.session.config.databases.items():
            if not db_config.is_file_source():
                databases.append({"name": name, "uri": db_config.uri or ""})
                seen_db_names.add(name)

    # Include dynamically added databases (from projects) not in base config
    if managed and hasattr(managed.session, 'schema_manager'):
        from constat.catalog.sql_transpiler import TranspilingConnection
        for name, conn in managed.session.schema_manager.connections.items():
            if name not in seen_db_names:
                if isinstance(conn, TranspilingConnection):
                    uri = str(conn.engine.url)
                else:
                    uri = str(conn.url)
                databases.append({"name": name, "uri": uri})
                seen_db_names.add(name)

    return apis, databases


def generate_inference_script(
    inferences: list[dict],
    premises: list[dict],
    apis: list[dict],
    databases: list[dict],
    session_label: str,
) -> str:
    """Generate a standalone Python script from inference codes.

    Returns the script content as a string.
    """
    import ast as _ast

    lines = [
        '#!/usr/bin/env python3',
        '"""',
        f'Constat Inference Code - Session {session_label}',
        '',
        'Auto-generated from auditable mode execution.',
        'Each inference function derives facts from premises using code.',
        '"""',
        '',
        'import pandas as pd',
        'import numpy as np',
        'import duckdb',
        'import json',
        'import tempfile',
        'from pathlib import Path',
        '',
        '',
        '# ============================================================================',
        '# Store Class',
        '# ============================================================================',
        '',
        'class _DataStore:',
        '    def __init__(self):',
        '        self._conn = duckdb.connect()',
        '        self._output_dir: Path | None = None',
        '        self._files: dict[str, str] = {}',
        '',
        '    def _ensure_output_dir(self) -> Path:',
        '        if self._output_dir is None:',
        '            self._output_dir = Path(tempfile.mkdtemp(prefix="constat_skill_"))',
        '        return self._output_dir',
        '',
        '    def save_dataframe(self, name: str, df: pd.DataFrame, **kwargs) -> None:',
        '        self._conn.register(name, df)',
        '        out = self._ensure_output_dir() / f"{name}.parquet"',
        '        df.to_parquet(out, index=False)',
        '        self._files[name] = str(out)',
        '        print(f"Saved: {name} ({len(df)} rows)")',
        '',
        '    def query(self, sql: str) -> pd.DataFrame:',
        '        return self._conn.execute(sql).fetchdf()',
        '',
        '    def load_dataframe(self, name: str) -> pd.DataFrame:',
        '        if name not in self._files:',
        '            raise ValueError(f"Table not found: {name}")',
        '        return pd.read_parquet(self._files[name])',
        '',
        '',
        'store = _DataStore()',
        '',
    ]

    # Add module-level defaults for constant premises (overridable via run_proof kwargs)
    constant_premises_early = [p for p in premises if p.get("source") in ("embedded", "llm_knowledge")]
    if constant_premises_early:
        lines.append('# Default parameters (overridable via run_proof kwargs)')
        for p in constant_premises_early:
            pname = p["name"].lower().replace(" ", "_").replace("-", "_")
            value = p["value"]
            try:
                literal = _ast.literal_eval(value)
                lines.append(f'_{pname} = {repr(literal)}')
            except (ValueError, SyntaxError):
                lines.append(f'_{pname} = {repr(value)}')
        lines.append('')

    # Add API helpers
    if apis:
        lines.extend(['import requests', ''])
        for api in apis:
            if api['type'] == 'graphql':
                lines.extend([
                    f"API_{api['name'].upper()}_URL = '{api['url']}'",
                    '',
                    f'def api_{api["name"]}(query: str, variables: dict = None) -> dict:',
                    f'    """GraphQL query against {api["name"]}."""',
                    f'    resp = requests.post(API_{api["name"].upper()}_URL, json={{"query": query, "variables": variables or {{}}}})',
                    '    resp.raise_for_status()',
                    '    result = resp.json()',
                    '    if "errors" in result and not result.get("data"):',
                    '        raise ValueError(f"GraphQL errors (no data returned): {result[\'errors\'][0][\'message\']}")',
                    '    return result.get("data", result)',
                    '',
                    '',
                ])
            else:
                lines.extend([
                    f"API_{api['name'].upper()}_URL = '{api['url']}'",
                    '',
                    f'def api_{api["name"]}(method_path: str, params: dict = None) -> dict:',
                    f'    """REST call to {api["name"]}."""',
                    '    parts = method_path.split(" ", 1)',
                    '    method = parts[0].upper()',
                    '    path = parts[1] if len(parts) > 1 else "/"',
                    f'    url = API_{api["name"].upper()}_URL.rstrip("/") + ("/" + path.lstrip("/") if not path.startswith("/") else path)',
                    '    resp = requests.request(method, url, params=params)',
                    '    resp.raise_for_status()',
                    '    return resp.json()',
                    '',
                    '',
                ])

    # Add database helpers
    if databases:
        lines.append('from sqlalchemy import create_engine')
        lines.append('')
        for db in databases:
            lines.append(f"db_{db['name']} = create_engine('{db['uri']}')")
        lines.extend(['', ''])

    # LLM primitives — auto-detects provider from env vars (ANTHROPIC_API_KEY, etc.)
    lines.extend([
        '# LLM primitives — auto-detects provider from env vars (ANTHROPIC_API_KEY, etc.)',
        'from constat.llm import llm_map, llm_classify, llm_extract, llm_summarize, llm_score',
        '',
        '',
        '# ============================================================================',
        '# Inference Functions',
        '# ============================================================================',
        '',
    ])

    # Add each inference as a function
    for inf in inferences:
        iid = inf["inference_id"]
        name = inf.get("name", iid)
        operation = inf.get("operation", "")
        code = inf.get("code", "pass")

        lines.append(f'def {iid.lower()}_{name.lower().replace(" ", "_").replace("-", "_")}():')
        lines.append(f'    """{iid}: {name} = {operation}"""')
        for code_line in code.split('\n'):
            if code_line.strip():
                lines.append(f'    {code_line}')
            else:
                lines.append('')
        lines.append('    return _result')
        lines.extend(['', ''])

    # Load premises for parameter generation
    constant_premises = [p for p in premises if p.get("source") in ("embedded", "llm_knowledge")]

    # Add main runner
    lines.extend([
        '# ============================================================================',
        '# Main',
        '# ============================================================================',
        '',
    ])

    # Build run_proof signature with constant premises as kwargs
    param_parts = []
    param_names = []
    for p in constant_premises:
        pname = p["name"].lower().replace(" ", "_").replace("-", "_")
        param_names.append((pname, p["name"]))
        value = p["value"]
        try:
            literal = _ast.literal_eval(value)
            param_parts.append(f'{pname}={repr(literal)}')
        except (ValueError, SyntaxError):
            param_parts.append(f'{pname}={repr(value)}')

    sig = ', '.join(param_parts)
    lines.append(f'def run_proof({sig}):')
    lines.append('    """Execute all inferences and return collected datasets.')
    lines.append('')
    lines.append('    Returns:')
    lines.append('        dict[str, str]: Map of dataset name to Parquet file path.')
    lines.append('        The final result is also available under the "_result" key.')
    lines.append('    """')

    # Set module-level defaults from params so inference functions can read them
    if constant_premises:
        global_names = [f'_{pname}' for pname, _ in param_names]
        lines.append(f'    global {", ".join(global_names)}')
        for pname, original_name in param_names:
            lines.append(f'    _{pname} = {pname}')
        lines.append('')
        lines.append('    # Store premise constants')
        lines.append('    _premises = {}')
        for pname, original_name in param_names:
            lines.append(f'    _premises["{original_name}"] = {pname}')
        lines.append('    store.save_dataframe("_premises", pd.DataFrame([_premises]))')
        lines.append('')

    lines.append('    _last = None')
    for inf in inferences:
        iid = inf["inference_id"]
        name = inf.get("name", iid)
        func_name = f'{iid.lower()}_{name.lower().replace(" ", "_").replace("-", "_")}'
        lines.append(f'    print("\\n=== {iid}: {name} ===")')
        lines.append(f'    _last = {func_name}()')
    lines.extend([
        '',
        '    # Save final result and return file paths',
        '    if _last is not None and hasattr(_last, "to_parquet"):',
        '        store.save_dataframe("_result", _last)',
        '    return dict(store._files)',
        '',
        '',
        'if __name__ == "__main__":',
        '    paths = run_proof()',
        '    print("\\n=== Output Files ===")',
        '    for name, path in paths.items():',
        '        print(f"  {name}: {path}")',
        '',
    ])

    return '\n'.join(lines)


@router.get("/{session_id}/download-inference-code")
async def download_inference_code(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Download all inference codes as a standalone Python script.

    Generates a self-contained script with API helpers, store class,
    and each inference step as a function that can be run independently.
    """
    from fastapi.responses import Response

    managed = session_manager.get_session_or_none(session_id)

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
    else:
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=user_id)
        history_session_id = history.find_session_by_server_id(session_id)

    try:
        if not history_session_id:
            raise HTTPException(status_code=404, detail="No inference code available for this session.")

        inferences = history.list_inference_codes(history_session_id)
        if not inferences:
            raise HTTPException(status_code=404, detail="No inference code available. Run an auditable query first.")

        apis, databases = _gather_source_configs(managed)
        premises = history.list_inference_premises(history_session_id)

        script_content = generate_inference_script(
            inferences=inferences,
            premises=premises,
            apis=apis,
            databases=databases,
            session_label=session_id[:8],
        )
        return Response(
            content=script_content,
            media_type="text/x-python",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_inference.py"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading inference code: {e}")
        raise HTTPException(status_code=500, detail=str(e))
