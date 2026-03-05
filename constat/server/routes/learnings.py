# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Learnings and configuration REST endpoints."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.server.persona_config import require_write

from constat.core.config import Config
from constat.server.auth import CurrentUserId
from constat.server.config import ServerConfig
from constat.server.session_manager import SessionManager
from constat.server.models import (
    ConfigResponse,
    DomainDetailResponse,
    DomainInfo,
    DomainListResponse,
    DomainTreeNode,
    ExemplarGenerateResponse,
    LearningCreateRequest,
    LearningInfo,
    LearningListResponse,
    RuleCreateRequest,
    RuleInfo,
    RuleUpdateRequest,
)
from constat.server.config import ServerConfig
from constat.server.permissions import get_user_permissions
from constat.storage.learnings import LearningCategory, LearningSource

logger = logging.getLogger(__name__)

router = APIRouter()


def get_config(request: Request) -> Config:
    """Dependency to get config from app state."""
    return request.app.state.config


def get_server_config(request: Request) -> ServerConfig:
    """Dependency to get server config from app state."""
    return request.app.state.server_config


def _can_modify_domain(domain, user_id: str, server_config: ServerConfig) -> bool:
    """Check if user can modify a domain (admin bypasses all restrictions)."""
    perms = get_user_permissions(server_config, user_id)
    if perms.is_admin:
        return True
    if domain.tier == "system":
        return False
    if domain.owner and domain.owner != user_id and domain.steward != user_id:
        return False
    return True


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _check_domain_cycles(graph: dict[str, list[str]]) -> list[str]:
    """Check for cycles in a domain composition graph.

    Args:
        graph: dict mapping domain filename -> list of child domain filenames

    Returns:
        List of cycle descriptions (empty if no cycles).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in graph}
    cycles: list[str] = []

    def dfs(name: str, path: list[str]) -> None:
        color[name] = GRAY
        for child in graph.get(name, []):
            if child not in color:
                continue
            if color[child] == GRAY:
                cycles.append(" -> ".join(path + [child]))
            elif color[child] == WHITE:
                dfs(child, path + [child])
        color[name] = BLACK

    for name in graph:
        if color[name] == WHITE:
            dfs(name, [name])
    return cycles


def _domain_data_dirs(domain_cfg, user_id: str) -> list[Path]:
    """Directories containing resources for a single domain (no recursion).

    Returns only this domain's own directory and user overlay — no sub-domain walking.
    """
    dirs: list[Path] = []
    if domain_cfg.source_path:
        d = Path(domain_cfg.source_path).parent
        if d.is_dir():
            dirs.append(d)
    user_domain_dir = Path(".constat") / user_id / "domains" / domain_cfg.filename
    if user_domain_dir.is_dir():
        dirs.append(user_domain_dir)
    return dirs


# In-memory learnings store (would use LearningStore in production)
_learnings: list[dict[str, Any]] = []


@router.get("/learnings", response_model=LearningListResponse)
async def list_learnings(
    user_id: CurrentUserId,
    category: str | None = None,
    _config: Config = Depends(get_config),
) -> LearningListResponse:
    """Get all captured learnings for the authenticated user.

    Args:
        user_id: Authenticated user ID
        category: Optional category filter
        _config: Injected application config

    Returns:
        List of learnings
    """
    logger.info(f"[LEARNINGS] Fetching learnings for user_id={user_id}")
    # Try to get from LearningStore if available
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        logger.info(f"[LEARNINGS] LearningStore file_path={store.file_path}, exists={store.file_path.exists()}")
        cat_enum = LearningCategory(category) if category else None
        learnings_data = store.list_raw_learnings(category=cat_enum, limit=100)
        rules_data = store.list_rules(category=cat_enum, limit=50)

        # Also collect rules from domain-scoped learnings.yaml files
        import yaml as _yaml
        seen_rule_ids = {r["id"] for r in rules_data}
        for _key, domain_cfg in (_config.domains or {}).items():
            if domain_cfg.source_path:
                for d in _domain_data_dirs(domain_cfg, user_id):
                    lf = d / "learnings.yaml"
                    if lf.is_file():
                        try:
                            ldata = _yaml.safe_load(lf.read_text()) or {}
                            for rid, rdata in (ldata.get("rules") or {}).items():
                                if rid not in seen_rule_ids:
                                    seen_rule_ids.add(rid)
                                    rules_data.append({"id": rid, **rdata})
                        except Exception:
                            pass

        logger.info(f"[LEARNINGS] Loaded {len(learnings_data)} learnings, {len(rules_data)} rules")

        return LearningListResponse(
            learnings=[
                LearningInfo(
                    id=l.get("id", str(uuid.uuid4())),
                    content=l.get("correction", ""),  # YAML uses 'correction' field
                    category=l.get("category", LearningCategory.USER_CORRECTION.value),
                    source=l.get("source", LearningSource.EXPLICIT_COMMAND.value),
                    context=l.get("context"),
                    applied_count=l.get("applied_count", 0),
                    created_at=datetime.fromisoformat(l["created"]) if l.get("created") else datetime.now(timezone.utc),
                )
                for l in learnings_data
            ],
            rules=[
                RuleInfo(
                    id=r.get("id", ""),
                    summary=r.get("summary", ""),
                    category=r.get("category", LearningCategory.USER_CORRECTION.value),
                    confidence=r.get("confidence", 0.0),
                    source_count=len(r.get("source_learnings", [])),
                    tags=r.get("tags", []),
                    domain=r.get("domain", ""),
                )
                for r in rules_data
            ],
        )
    except Exception as e:
        logger.warning(f"Could not load from LearningStore: {e}")

    # Fall back to in-memory store
    filtered = _learnings
    if category:
        filtered = [l for l in _learnings if l.get("category") == category]

    return LearningListResponse(
        learnings=[
            LearningInfo(
                id=l["id"],
                content=l["content"],
                category=l.get("category", LearningCategory.USER_CORRECTION.value),
                source=l.get("source", LearningSource.EXPLICIT_COMMAND.value),
                context=l.get("context"),
                applied_count=l.get("applied_count", 0),
                created_at=datetime.fromisoformat(l["created_at"]),
            )
            for l in filtered
        ],
        rules=[],
    )


@router.post("/learnings", response_model=LearningInfo, dependencies=[Depends(require_write("learnings"))])
async def add_learning(
    body: LearningCreateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> LearningInfo:
    """Add a new learning for the authenticated user.

    Args:
        body: Learning content and category
        user_id: Authenticated user ID
        _config: Injected application config

    Returns:
        Created learning
    """
    now = datetime.now(timezone.utc)
    learning_id = str(uuid.uuid4())

    learning = {
        "id": learning_id,
        "content": body.content,
        "category": body.category,
        "source": LearningSource.EXPLICIT_COMMAND.value,
        "context": None,
        "applied_count": 0,
        "created_at": now.isoformat(),
    }

    # Try to persist to LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        store.save_learning(
            correction=body.content,
            category=LearningCategory(body.category),
            context={},
            source=LearningSource.EXPLICIT_COMMAND,
        )
    except Exception as e:
        logger.warning(f"Could not persist to LearningStore: {e}")

    # Also store in memory
    _learnings.append(learning)

    return LearningInfo(
        id=learning_id,
        content=body.content,
        category=body.category,
        source=LearningSource.EXPLICIT_COMMAND.value,
        context=None,
        applied_count=0,
        created_at=now,
    )


@router.delete("/learnings/{learning_id}", dependencies=[Depends(require_write("learnings"))])
async def delete_learning(
    learning_id: str,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> dict:
    """Delete a learning for the authenticated user.

    Args:
        learning_id: Learning ID to delete
        user_id: Authenticated user ID
        _config: Injected application config

    Returns:
        Deletion confirmation

    Raises:
        404: Learning not found
    """
    global _learnings

    # Try to delete from LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        if store.delete_learning(learning_id):
            return {"status": "deleted", "id": learning_id}
    except Exception as e:
        logger.warning(f"Could not delete from LearningStore: {e}")

    # Try in-memory store
    original_len = len(_learnings)
    _learnings = [l for l in _learnings if l["id"] != learning_id]

    if len(_learnings) == original_len:
        raise HTTPException(status_code=404, detail=f"Learning not found: {learning_id}")

    return {"status": "deleted", "id": learning_id}


@router.post("/learnings/compact")
async def compact_learnings(
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Compact similar learnings into rules using LLM.

    This analyzes pending learnings and groups similar ones into rules.

    Returns:
        Compaction results with counts of rules created/strengthened
    """
    try:
        from constat.storage.learnings import LearningStore
        from constat.learning.compactor import LearningCompactor
        from constat.providers import TaskRouter

        store = LearningStore(user_id=user_id)
        stats = store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        if unpromoted < 2:
            return {
                "status": "skipped",
                "message": f"Not enough learnings to compact ({unpromoted} pending, need at least 2)",
                "rules_created": 0,
                "learnings_archived": 0,
            }

        # Create LLM router for compaction
        llm = TaskRouter(config.llm)

        compactor = LearningCompactor(store, llm)
        result = compactor.compact(dry_run=False)

        return {
            "status": "success",
            "rules_created": result.rules_created,
            "rules_strengthened": result.rules_strengthened,
            "rules_merged": result.rules_merged,
            "learnings_archived": result.learnings_archived,
            "groups_found": result.groups_found,
            "skipped_low_confidence": result.skipped_low_confidence,
            "errors": result.errors,
        }
    except ImportError as e:
        logger.warning(f"Compactor not available: {e}")
        return {
            "status": "error",
            "message": "Learning compactor not available",
            "rules_created": 0,
            "learnings_archived": 0,
        }
    except Exception as e:
        logger.error(f"Error compacting learnings: {e}")
        return {
            "status": "error",
            "message": str(e),
            "rules_created": 0,
            "learnings_archived": 0,
        }


@router.post("/learnings/generate-exemplars", response_model=ExemplarGenerateResponse)
async def generate_exemplars(
    session_id: str,
    user_id: CurrentUserId,
    coverage: str = "standard",
    config: Config = Depends(get_config),
    session_manager: SessionManager = Depends(get_session_manager),
) -> ExemplarGenerateResponse:
    """Generate fine-tuning exemplar pairs from rules, glossary, and relationships."""
    if coverage not in ("minimal", "standard", "comprehensive"):
        raise HTTPException(status_code=400, detail="coverage must be minimal, standard, or comprehensive")

    managed = session_manager.get_session(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Get dependencies
    from constat.storage.learnings import LearningStore
    from constat.learning.exemplar_generator import ExemplarGenerator
    from constat.providers import TaskRouter

    store = LearningStore(user_id=user_id)
    llm = TaskRouter(config.llm)

    # Get vector store from session
    session = managed.session
    vs = None
    if hasattr(session, "doc_tools") and session.doc_tools:
        vs = session.doc_tools._vector_store
    if not vs:
        raise HTTPException(status_code=503, detail="Vector store not available")

    generator = ExemplarGenerator(store, vs, llm, session_id, user_id)
    result = generator.generate(coverage=coverage)

    download_urls = {
        fmt: f"/api/learnings/exemplars/download?format={fmt}&user_id={user_id}"
        for fmt in result.output_paths
    }

    return ExemplarGenerateResponse(
        status="success",
        coverage=coverage,
        rule_pairs=result.rule_pairs,
        glossary_pairs=result.glossary_pairs,
        relationship_pairs=result.relationship_pairs,
        total=result.total,
        download_urls=download_urls,
    )


@router.get("/learnings/exemplars/download")
async def download_exemplars(
    format: str,
    user_id: CurrentUserId,
):
    """Download generated exemplar JSONL file."""
    from fastapi.responses import FileResponse

    filenames = {
        "messages": "exemplars_messages.jsonl",
        "alpaca": "exemplars_alpaca.jsonl",
    }
    if format not in filenames:
        raise HTTPException(status_code=400, detail="format must be 'messages' or 'alpaca'")

    path = Path(".constat") / user_id / filenames[format]
    if not path.exists():
        raise HTTPException(status_code=404, detail="No exemplar file found. Generate exemplars first.")

    return FileResponse(
        path=str(path),
        media_type="application/jsonl",
        filename=filenames[format],
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config_sanitized(
    config: Config = Depends(get_config),
) -> ConfigResponse:
    """Get current configuration (sanitized).

    Returns config without sensitive data like API keys.

    Returns:
        Sanitized configuration
    """
    return ConfigResponse(
        databases=list(config.databases.keys()),
        apis=list(config.apis.keys()),
        documents=list(config.documents.keys()),
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        execution_timeout=config.execution.timeout_seconds,
    )


# ============================================================================
# Domain Endpoints
# ============================================================================


@router.get("/domains/tree")
async def get_domain_tree(
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> list[DomainTreeNode]:
    """Get nested domain hierarchy tree.

    Uses flat discovery + DAG assembly from config `domains:` lists.
    Filesystem nesting is for organizational ownership only — composition
    is defined by the `domains:` field in each domain's config.yaml.
    """
    from constat.core.config import DomainConfig

    def _build_node(domain_cfg: DomainConfig, tier: str = "system") -> DomainTreeNode:
        import yaml as _yaml

        skill_names: list[str] = []
        agent_names: list[str] = []
        rule_ids: list[str] = []
        fact_names: list[str] = []
        seen_skills: set[str] = set()
        seen_agents: set[str] = set()
        seen_rules: set[str] = set()
        seen_facts: set[str] = set()

        for d in _domain_data_dirs(domain_cfg, user_id):
            skills_dir = d / "skills"
            if skills_dir.is_dir():
                for sd in skills_dir.iterdir():
                    if sd.is_dir() and (sd / "SKILL.md").exists() and sd.name not in seen_skills:
                        seen_skills.add(sd.name)
                        skill_names.append(sd.name)

            agents_file = d / "agents.yaml"
            if agents_file.is_file():
                try:
                    agent_data = _yaml.safe_load(agents_file.read_text()) or {}
                    for name in agent_data:
                        if name not in seen_agents:
                            seen_agents.add(name)
                            agent_names.append(name)
                except Exception:
                    pass

            learnings_file = d / "learnings.yaml"
            if learnings_file.is_file():
                try:
                    ldata = _yaml.safe_load(learnings_file.read_text()) or {}
                    for rid in (ldata.get("rules") or {}):
                        if rid not in seen_rules:
                            seen_rules.add(rid)
                            rule_ids.append(rid)
                except Exception:
                    pass

            facts_file = d / "facts.yaml"
            if facts_file.is_file():
                try:
                    fdata = _yaml.safe_load(facts_file.read_text()) or {}
                    for fname in (fdata.get("facts") or {}):
                        if fname not in seen_facts:
                            seen_facts.add(fname)
                            fact_names.append(fname)
                except Exception:
                    pass

        # Also check user LearningStore for rules tagged with this domain
        try:
            from constat.storage.learnings import LearningStore
            store = LearningStore(user_id=user_id)
            for r in store.list_rules(domain=domain_cfg.filename):
                if r["id"] not in seen_rules:
                    seen_rules.add(r["id"])
                    rule_ids.append(r["id"])
        except Exception:
            pass

        # Also check user FactStore for facts tagged with this domain
        try:
            from constat.storage.facts import FactStore
            fact_store = FactStore(user_id=user_id)
            for fname, fdata in fact_store.list_all_facts().items():
                if fdata.get("domain") == domain_cfg.filename and fname not in seen_facts:
                    seen_facts.add(fname)
                    fact_names.append(fname)
        except Exception:
            pass

        return DomainTreeNode(
            filename=domain_cfg.filename,
            name=domain_cfg.name,
            path=domain_cfg.path,
            description=domain_cfg.description,
            tier=tier,
            active=domain_cfg.active,
            steward=domain_cfg.steward,
            owner=domain_cfg.owner,
            databases=list(domain_cfg.databases.keys()),
            apis=list(domain_cfg.apis.keys()),
            documents=list(domain_cfg.documents.keys()),
            skills=skill_names,
            agents=agent_names,
            rules=rule_ids,
            facts=fact_names,
            system_prompt=domain_cfg.system_prompt or "",
            domains=domain_cfg.domains or [],
        )

    # --- Phase 1: Flat discovery ---
    # Discover all domains from all tiers without building hierarchy.
    # all_configs/all_nodes keyed by canonical FQ name (e.g. "executive/sales-analytics").
    all_configs: dict[str, tuple[DomainConfig, str]] = {}
    all_nodes: dict[str, DomainTreeNode] = {}

    def _register(canonical: str, cfg: DomainConfig, tier: str) -> None:
        if canonical in all_configs:
            return
        cfg.filename = canonical
        all_configs[canonical] = (cfg, tier)
        node = _build_node(cfg, tier=tier)
        node.filename = canonical
        all_nodes[canonical] = node

    def _flat_scan(dir_path: Path, prefix: str = "", tier: str = "system") -> None:
        """Recursively discover domain directories, registering each one flat."""
        if not dir_path.is_dir():
            return
        for entry in sorted(dir_path.iterdir()):
            # Auto-migrate flat-file domains to directory-based
            if entry.is_file() and entry.suffix in (".yaml", ".yml"):
                target_dir = entry.with_suffix("")
                if not target_dir.is_dir():
                    import shutil
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(entry), str(target_dir / "config.yaml"))
                    logger.info(f"Migrated flat domain {entry.name} -> {target_dir.name}/config.yaml")

            if entry.is_dir() and (entry / "config.yaml").exists():
                domain_cfg = DomainConfig.from_directory(entry)
                canonical = f"{prefix}/{entry.name}" if prefix else entry.name
                _register(canonical, domain_cfg, tier)
                # Recurse into nested domains/ for discovery only
                sub_dir = entry / "domains"
                if sub_dir.is_dir():
                    _flat_scan(sub_dir, prefix=canonical, tier=tier)

    # Scan system domains
    if config.config_dir:
        domains_dir = Path(config.config_dir) / "domains"
        _flat_scan(domains_dir)

    # Normalize filenames to match config.domains keys
    if all_nodes and config.domains:
        config_keys = set(config.domains.keys())
        renames: list[tuple[str, str]] = []
        for fname in list(all_nodes.keys()):
            stem = fname.removesuffix(".yaml").removesuffix(".yml")
            if stem in config_keys and fname not in config_keys:
                renames.append((fname, stem))
        for old, new in renames:
            all_nodes[new] = all_nodes.pop(old)
            all_nodes[new].filename = new
            cfg, tier = all_configs.pop(old)
            cfg.filename = new
            all_configs[new] = (cfg, tier)

    # Fallback: config.domains dict
    if not all_nodes and config.domains:
        for key, domain_cfg in sorted(config.domains.items()):
            _register(key, domain_cfg, "system")

    # Scan shared domains
    shared_domains_dir = Path(".constat") / "shared" / "domains"
    _flat_scan(shared_domains_dir, tier="shared")

    # Scan user domains
    user_domains_dir = Path(".constat") / user_id / "domains"
    _flat_scan(user_domains_dir, tier="user")

    def _resolve_ref(ref: str, parent_key: str = "") -> str | None:
        """Resolve a domain reference to its canonical key.

        Tries exact/FQ match first, then relative to parent (parent_key/ref).
        """
        if ref in all_configs:
            return ref
        if parent_key:
            relative = f"{parent_key}/{ref}"
            if relative in all_configs:
                return relative
        return None

    # --- Phase 2: Cycle detection (3-color DFS) ---
    WHITE, GRAY, BLACK = 0, 1, 2
    _color = {name: WHITE for name in all_configs}
    bad_edges: set[tuple[str, str]] = set()

    def _dfs(name: str, path: list[str]) -> None:
        _color[name] = GRAY
        cfg, _ = all_configs[name]
        for child_ref in (cfg.domains or []):
            child = _resolve_ref(child_ref, parent_key=name)
            if child is None or child not in _color:
                continue
            if _color[child] == GRAY:
                bad_edges.add((name, child))
                logger.error(f"Domain cycle: {' -> '.join(path + [child])}")
            elif _color[child] == WHITE:
                _dfs(child, path + [child])
        _color[name] = BLACK

    for _name in all_configs:
        if _color[_name] == WHITE:
            _dfs(_name, [_name])

    # --- Phase 3: DAG assembly ---
    referenced: set[str] = set()
    for fname, (cfg, _) in all_configs.items():
        for child_ref in (cfg.domains or []):
            child_name = _resolve_ref(child_ref, parent_key=fname)
            if child_name and child_name in all_nodes and (fname, child_name) not in bad_edges:
                referenced.add(child_name)

    # Attach children
    for fname, (cfg, _) in all_configs.items():
        node = all_nodes[fname]
        for child_ref in (cfg.domains or []):
            child_name = _resolve_ref(child_ref, parent_key=fname)
            if child_name and child_name in all_nodes and (fname, child_name) not in bad_edges:
                node.children.append(all_nodes[child_name])

    # Top-level = unreferenced domains
    top_level = [all_nodes[fname] for fname in all_nodes if fname not in referenced]

    # --- Phase 4: Root wrapper ---
    # Collect user-level (unscoped) skills, agents, rules, facts for User node
    user_skill_names: list[str] = []
    user_agent_names: list[str] = []
    user_rule_ids: list[str] = []
    user_fact_names: list[str] = []
    user_skills_dir = Path(".constat") / user_id / "skills"
    if user_skills_dir.is_dir():
        for sd in user_skills_dir.iterdir():
            if sd.is_dir() and (sd / "SKILL.md").exists():
                user_skill_names.append(sd.name)
    user_agents_file = Path(".constat") / user_id / "agents.yaml"
    if user_agents_file.is_file():
        try:
            import yaml as _yaml
            agent_data = _yaml.safe_load(user_agents_file.read_text()) or {}
            user_agent_names = list(agent_data.keys())
        except Exception:
            pass
    # Collect unscoped user rules
    try:
        from constat.storage.learnings import LearningStore
        user_store = LearningStore(user_id=user_id)
        for r in user_store.list_rules(domain=""):
            user_rule_ids.append(r["id"])
    except Exception:
        pass
    # Collect unscoped user facts
    try:
        from constat.storage.facts import FactStore
        user_fact_store = FactStore(user_id=user_id)
        for fname, fdata in user_fact_store.list_all_facts().items():
            if not fdata.get("domain"):
                user_fact_names.append(fname)
    except Exception:
        pass

    # Build User child node
    user_node = DomainTreeNode(
        filename="user",
        name="User",
        path="user",
        description="",
        tier="user",
        owner=user_id,
        skills=user_skill_names,
        agents=user_agent_names,
        rules=user_rule_ids,
        facts=user_fact_names,
    )

    # Attach user-tier domains as children of User node
    user_tier_domains = [all_nodes[f] for f in all_nodes
                         if all_configs[f][1] == "user" and f not in referenced]
    system_top_level = [n for n in top_level if n not in user_tier_domains]
    user_node.children = user_tier_domains

    # Root and User as siblings
    root_node = DomainTreeNode(
        filename="root",
        name="System",
        path="root",
        description="System configuration",
        databases=list(config.databases.keys()),
        apis=list(config.apis.keys()),
        documents=[],
        domains=config.domain_refs or [],
        children=system_top_level,
    )
    return [root_node, user_node]


@router.get("/domains", response_model=DomainListResponse)
async def list_domains(
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> DomainListResponse:
    """List available domains from system and user directories.

    Returns:
        List of available domains
    """
    domain_infos = config.list_domains()

    # Add tier to system domains
    for info in domain_infos:
        info.setdefault("tier", "system")
        info.setdefault("active", True)
        info.setdefault("owner", "")

    # Scan shared domains (.constat/shared/domains/*.yaml)
    shared_domains_dir = Path(".constat") / "shared" / "domains"
    if shared_domains_dir.is_dir():
        import yaml
        for yaml_file in sorted(shared_domains_dir.glob("*.yaml")):
            filename = yaml_file.name
            if filename in config.domains:
                continue
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
                domain_infos.append({
                    "filename": filename,
                    "name": data.get("name", yaml_file.stem),
                    "description": data.get("description", ""),
                    "tier": "shared",
                    "active": data.get("active", True),
                    "owner": data.get("owner", ""),
                    "steward": data.get("steward", ""),
                })
                from constat.core.config import DomainConfig
                data["filename"] = filename
                data["source_path"] = str(yaml_file.resolve())
                data["tier"] = "shared"
                config.domains[filename] = DomainConfig(**data)
            except Exception as e:
                logger.warning(f"Failed to load shared domain {filename}: {e}")

    # Also scan user-level domains (.constat/{user_id}/domains/*.yaml)
    user_domains_dir = Path(".constat") / user_id / "domains"
    if user_domains_dir.is_dir():
        import yaml
        for yaml_file in sorted(user_domains_dir.glob("*.yaml")):
            filename = yaml_file.name
            if filename in config.domains:
                continue  # System domain takes precedence
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
                domain_infos.append({
                    "filename": filename,
                    "name": data.get("name", yaml_file.stem),
                    "description": data.get("description", ""),
                    "tier": "user",
                    "active": data.get("active", True),
                    "owner": user_id,
                    "steward": data.get("steward", ""),
                })
                # Register in config.domains so load_domain/get_domain_content work
                from constat.core.config import DomainConfig
                data["filename"] = filename
                data["source_path"] = str(yaml_file.resolve())
                data["tier"] = "user"
                data["owner"] = user_id
                config.domains[filename] = DomainConfig(**data)
            except Exception as e:
                logger.warning(f"Failed to load user domain {filename}: {e}")

    return DomainListResponse(
        domains=[DomainInfo(**p) for p in domain_infos]
    )


@router.post("/domains")
async def create_domain(
    body: dict,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Create a new user-level domain.

    Args:
        body: Dict with 'name' and optional 'description'
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Created domain info
    """
    import re
    import yaml

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    description = body.get("description", "").strip()
    system_prompt = body.get("system_prompt", "").strip()
    parent_domain = body.get("parent_domain", "").strip()
    initial_domains: list[str] = body.get("initial_domains", [])

    # Slugify name to filename
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise HTTPException(status_code=400, detail="Invalid name")
    filename = slug

    # Check for conflicts with system domains
    if filename in config.domains:
        raise HTTPException(status_code=409, detail=f"Domain '{filename}' already exists")

    # Write domain directory (slug/config.yaml)
    user_domains_dir = Path(".constat") / user_id / "domains"
    domain_dir = user_domains_dir / slug
    domain_path = domain_dir / "config.yaml"

    if domain_dir.exists():
        raise HTTPException(status_code=409, detail=f"Domain '{filename}' already exists")

    domain_dir.mkdir(parents=True, exist_ok=True)

    content: dict[str, Any] = {
        "name": name,
        "description": description,
        "owner": user_id,
        "databases": {},
        "apis": {},
        "documents": {},
    }
    if system_prompt:
        content["system_prompt"] = system_prompt
    if initial_domains:
        content["domains"] = initial_domains
    domain_path.write_text(yaml.dump(content, default_flow_style=False, sort_keys=False))

    # Register in config.domains
    from constat.core.config import DomainConfig
    content["filename"] = filename
    content["source_path"] = str(domain_path.resolve())
    config.domains[filename] = DomainConfig(**content)

    # If parent_domain specified, add this domain to parent's domains list
    if parent_domain and parent_domain != "user":
        if parent_domain == "root":
            # Add to main config.yaml domains list
            if config.config_dir:
                root_config_path = Path(config.config_dir) / "config.yaml"
                if root_config_path.is_file():
                    root_data = yaml.safe_load(root_config_path.read_text()) or {}
                    root_domains = root_data.get("domains", [])
                    if not isinstance(root_domains, list):
                        root_domains = list(root_domains.keys())
                    if filename not in root_domains:
                        root_domains.append(filename)
                        root_data["domains"] = root_domains
                        root_config_path.write_text(
                            yaml.dump(root_data, default_flow_style=False, sort_keys=False)
                        )
                        config.domain_refs = root_domains
        else:
            parent_cfg = config.domains.get(parent_domain)
            if parent_cfg and parent_cfg.source_path:
                parent_config_path = Path(parent_cfg.source_path)
                if parent_config_path.is_file():
                    parent_data = yaml.safe_load(parent_config_path.read_text()) or {}
                    domains_list = parent_data.get("domains", [])
                    if filename not in domains_list:
                        domains_list.append(filename)
                        parent_data["domains"] = domains_list
                        parent_config_path.write_text(
                            yaml.dump(parent_data, default_flow_style=False, sort_keys=False)
                        )
                        parent_cfg.domains = domains_list

    return {
        "status": "created",
        "filename": filename,
        "name": name,
        "description": description,
    }


@router.get("/domains/{filename}", response_model=DomainDetailResponse)
async def get_domain(
    filename: str,
    config: Config = Depends(get_config),
) -> DomainDetailResponse:
    """Get details for a specific domain.

    Args:
        filename: Domain YAML filename (e.g., 'sales-analytics.yaml')
        config: Injected application config

    Returns:
        Domain details including data source names

    Raises:
        404: Domain not found
    """
    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    return DomainDetailResponse(
        filename=filename,
        name=domain.name,
        description=domain.description,
        tier=domain.tier,
        active=domain.active,
        owner=domain.owner,
        steward=domain.steward,
        databases=list(domain.databases.keys()),
        apis=list(domain.apis.keys()),
        documents=list(domain.documents.keys()),
    )


@router.patch("/domains/{filename}")
async def update_domain(
    filename: str,
    body: dict,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Update domain metadata (rename, reorder, activate/deactivate)."""
    import yaml

    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    server_config = get_server_config(request)
    if not _can_modify_domain(domain, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this domain")

    # System domains: only active toggle allowed (unless admin)
    perms = get_user_permissions(server_config, user_id)
    if domain.tier == "system" and not perms.is_admin:
        disallowed = {k for k in body if k not in ("active",)}
        if disallowed:
            raise HTTPException(
                status_code=403,
                detail=f"System domains only allow toggling 'active'. Cannot modify: {', '.join(disallowed)}",
            )

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")

    domain_path = Path(domain.source_path)
    if not domain_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain file not found: {domain_path}")

    # Read current YAML
    data = yaml.safe_load(domain_path.read_text()) or {}

    # Apply updates
    allowed_fields = {"name", "description", "order", "active"}
    for key in allowed_fields:
        if key in body:
            data[key] = body[key]

    # Write back
    domain_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    # Update in-memory config
    for key in allowed_fields:
        if key in body and hasattr(domain, key):
            setattr(domain, key, body[key])

    return {"status": "updated", "filename": filename}


@router.delete("/domains/{filename}")
async def delete_domain(
    filename: str,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Delete a domain. System domains cannot be deleted."""
    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    server_config = get_server_config(request)
    if not _can_modify_domain(domain, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to delete this domain")

    # Prevent deletion if this domain is referenced in another domain's composition
    referencing = [
        k for k, d in config.domains.items()
        if k != filename and filename in (d.domains or [])
    ]
    if referencing:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: domain is composed by {', '.join(referencing)}",
        )

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")

    import shutil
    domain_path = Path(domain.source_path)
    domain_dir = domain_path.parent

    # Safety: only rmtree if the directory contains config.yaml
    if domain_dir.is_dir() and (domain_dir / "config.yaml").exists():
        shutil.rmtree(domain_dir)
    elif domain_path.exists():
        domain_path.unlink()

    # Remove from config
    config.domains.pop(filename, None)

    return {"status": "deleted", "filename": filename}


@router.get("/domains/{filename}/content")
async def get_domain_content(
    filename: str,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Get the raw YAML content of a domain file.

    Args:
        filename: Domain YAML filename
        user_id: Current user
        config: Injected application config

    Returns:
        Dict with 'content' (YAML string) and 'path' (full file path)

    Raises:
        404: Domain not found
    """
    import yaml

    # Root = main config.yaml
    if filename == "root":
        if not config.config_dir:
            raise HTTPException(status_code=400, detail="No config directory configured")
        domain_path = Path(config.config_dir) / "config.yaml"
        if not domain_path.exists():
            raise HTTPException(status_code=404, detail=f"Root config not found: {domain_path}")
        return {"content": domain_path.read_text(), "path": str(domain_path), "filename": "root"}

    # User = .constat/{user_id}/config.yaml — create default if missing
    if filename == "user":
        user_dir = Path(".constat") / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        domain_path = user_dir / "config.yaml"
        if not domain_path.exists():
            default_content = yaml.dump(
                {"name": "User", "description": "", "domains": []},
                default_flow_style=False, sort_keys=False,
            )
            domain_path.write_text(default_content)
        return {"content": domain_path.read_text(), "path": str(domain_path), "filename": "user"}

    # noinspection DuplicatedCode
    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path (inline config)")

    domain_path = Path(domain.source_path)
    if not domain_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain file not found: {domain_path}")

    content = domain_path.read_text()
    return {
        "content": content,
        "path": str(domain_path),
        "filename": filename,
    }


@router.put("/domains/{filename}/content")
async def update_domain_content(
    filename: str,
    body: dict,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Update the YAML content of a domain file.

    Args:
        filename: Domain YAML filename
        body: Dict with 'content' (new YAML string)
        user_id: Current user
        config: Injected application config

    Returns:
        Status confirmation

    Raises:
        404: Domain not found
        400: Invalid YAML
    """
    import yaml

    # Root = main config.yaml
    if filename == "root":
        if not config.config_dir:
            raise HTTPException(status_code=400, detail="No config directory configured")
        domain_path = Path(config.config_dir) / "config.yaml"
        if not domain_path.exists():
            raise HTTPException(status_code=404, detail=f"Root config not found: {domain_path}")
        content = body.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
        domain_path.write_text(content)
        return {"status": "saved", "filename": "root", "path": str(domain_path)}

    # User = .constat/{user_id}/config.yaml
    if filename == "user":
        user_dir = Path(".constat") / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        domain_path = user_dir / "config.yaml"
        content = body.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
        domain_path.write_text(content)
        return {"status": "saved", "filename": "user", "path": str(domain_path)}

    # noinspection DuplicatedCode
    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path (inline config)")

    domain_path = Path(domain.source_path)
    if not domain_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain file not found: {domain_path}")

    content = body.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    # Validate YAML before saving
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # Check for cycles if domains list changed
    new_domains = (parsed or {}).get("domains", []) if isinstance(parsed, dict) else []
    if new_domains:
        # Build canonical graph from all known domains, resolving refs relative to parent
        all_keys = set(config.domains.keys())
        all_keys.add(filename)

        def _resolve(ref: str, parent: str) -> str | None:
            if ref in all_keys:
                return ref
            relative = f"{parent}/{ref}"
            if relative in all_keys:
                return relative
            return None

        graph: dict[str, list[str]] = {}
        for k, d in config.domains.items():
            graph[k] = [r for ref in (d.domains or []) if (r := _resolve(ref, k))]
        graph[filename] = [r for ref in new_domains if (r := _resolve(ref, filename))]
        cycles = _check_domain_cycles(graph)
        if cycles:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot save: would create domain cycle(s): {'; '.join(cycles)}",
            )

    # Write the file
    domain_path.write_text(content)

    # Update in-memory config
    if isinstance(parsed, dict) and "domains" in parsed:
        domain.domains = parsed["domains"]

    return {
        "status": "saved",
        "filename": filename,
        "path": str(domain_path),
    }


@router.get("/domains/{filename}/skills")
async def list_domain_skills(
    filename: str,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """List skills belonging to a domain."""
    from constat.server.routes.skills import get_skill_manager
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)

    seen: set[str] = set()
    skills_out: list[dict] = []

    # Collect from all domain directories
    domain_cfg = config.load_domain(filename)
    if domain_cfg:
        for d in _domain_data_dirs(domain_cfg, user_id):
            skills_dir = d / "skills"
            if skills_dir.is_dir():
                for sd in skills_dir.iterdir():
                    if sd.is_dir() and (sd / "SKILL.md").exists() and sd.name not in seen:
                        seen.add(sd.name)
                        skills_out.append({"name": sd.name, "description": "", "domain": filename})

    # Also include from skill manager
    for s in manager.get_domain_skills(filename):
        if s.name not in seen:
            seen.add(s.name)
            skills_out.append({"name": s.name, "description": s.description, "domain": s.domain})

    return {"skills": skills_out}


@router.get("/domains/{filename}/agents")
async def list_domain_agents(
    filename: str,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """List agents belonging to a domain."""
    import yaml as _yaml

    seen: set[str] = set()
    agents_out: list[dict] = []

    # Collect from all domain directories
    domain_cfg = config.load_domain(filename)
    if domain_cfg:
        for d in _domain_data_dirs(domain_cfg, user_id):
            agents_file = d / "agents.yaml"
            if agents_file.is_file():
                try:
                    agent_data = _yaml.safe_load(agents_file.read_text()) or {}
                    for name, adata in agent_data.items():
                        if name not in seen:
                            seen.add(name)
                            agents_out.append({"name": name, "description": adata.get("description", ""), "domain": filename})
                except Exception:
                    pass

    # Also include from agent_manager
    sessions = list(session_manager.sessions.values())
    for managed in sessions:
        if hasattr(managed.session, "agent_manager"):
            for a in managed.session.agent_manager.get_domain_agents(filename):
                if a.name not in seen:
                    seen.add(a.name)
                    agents_out.append({"name": a.name, "description": a.description, "domain": a.domain})
            break

    return {"agents": agents_out}


@router.get("/domains/{filename}/rules")
async def list_domain_rules(
    filename: str,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """List rules belonging to a domain."""
    import yaml as _yaml

    seen_ids: set[str] = set()
    rules: list[dict] = []

    def _collect_from_file(learnings_file: Path) -> None:
        if not learnings_file.is_file():
            return
        ldata = _yaml.safe_load(learnings_file.read_text()) or {}
        for rid, rdata in (ldata.get("rules") or {}).items():
            if rid not in seen_ids:
                seen_ids.add(rid)
                rules.append({"id": rid, **rdata})

    # Collect from all domain directories (recursive)
    domain_cfg = config.load_domain(filename)
    if domain_cfg:
        for d in _domain_data_dirs(domain_cfg, user_id):
            _collect_from_file(d / "learnings.yaml")

    # Check user LearningStore for rules tagged with this domain
    from constat.storage.learnings import LearningStore
    store = LearningStore(user_id=user_id)
    for r in store.list_rules(domain=filename):
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            rules.append(r)

    return {
        "rules": [
            RuleInfo(
                id=r["id"],
                summary=r["summary"],
                category=r["category"],
                confidence=r.get("confidence", 0),
                source_count=len(r.get("source_learnings", [])),
                tags=r.get("tags", []),
                domain=r.get("domain", ""),
            )
            for r in rules
        ]
    }


@router.get("/domains/{filename}/facts")
async def list_domain_facts(
    filename: str,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """List facts belonging to a domain."""
    import yaml as _yaml

    seen: set[str] = set()
    facts_out: list[dict] = []

    # Collect from all domain directories (recursive)
    domain_cfg = config.load_domain(filename)
    if domain_cfg:
        for d in _domain_data_dirs(domain_cfg, user_id):
            facts_file = d / "facts.yaml"
            if facts_file.is_file():
                try:
                    fdata = _yaml.safe_load(facts_file.read_text()) or {}
                    for fname, fval in (fdata.get("facts") or {}).items():
                        if fname not in seen:
                            seen.add(fname)
                            facts_out.append({"name": fname, **(fval if isinstance(fval, dict) else {"value": fval})})
                except Exception:
                    pass

    # Check user FactStore for facts tagged with this domain
    from constat.storage.facts import FactStore
    fact_store = FactStore(user_id=user_id)
    for fname, fdata in fact_store.list_all_facts().items():
        if fdata.get("domain") == filename and fname not in seen:
            seen.add(fname)
            facts_out.append({"name": fname, **fdata})

    return {"facts": facts_out}


@router.post("/domains/{filename}/promote")
async def promote_domain(
    filename: str,
    body: dict,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Move a user domain to root (system domains directory)."""
    import shutil
    import yaml

    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    if domain.tier != "user":
        raise HTTPException(status_code=400, detail="Only user domains can be moved to root")

    server_config = get_server_config(request)
    if not _can_modify_domain(domain, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to move this domain")

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")

    src_config = Path(domain.source_path)
    src_dir = src_config.parent
    if not src_dir.exists():
        raise HTTPException(status_code=404, detail=f"Domain directory not found: {src_dir}")

    if not config.config_dir:
        raise HTTPException(status_code=400, detail="No config directory configured")

    # Move entire domain directory to system domains/
    system_domains_dir = Path(config.config_dir) / "domains"
    system_domains_dir.mkdir(parents=True, exist_ok=True)
    target_dir = system_domains_dir / filename

    if target_dir.exists():
        raise HTTPException(status_code=409, detail=f"Domain '{filename}' already exists under root")

    shutil.move(str(src_dir), str(target_dir))

    # Add to root composition list in config.yaml
    root_config_path = Path(config.config_dir) / "config.yaml"
    if root_config_path.is_file():
        root_data = yaml.safe_load(root_config_path.read_text()) or {}
        root_domains = root_data.get("domains", [])
        if not isinstance(root_domains, list):
            root_domains = list(root_domains.keys())
        if filename not in root_domains:
            root_domains.append(filename)
            root_data["domains"] = root_domains
            root_config_path.write_text(
                yaml.dump(root_data, default_flow_style=False, sort_keys=False)
            )
            config.domain_refs = root_domains

    # Update in-memory config
    from constat.core.config import DomainConfig as DC
    new_config_path = target_dir / "config.yaml"
    data = yaml.safe_load(new_config_path.read_text()) or {}
    data["filename"] = filename
    data["source_path"] = str(new_config_path.resolve())
    data["tier"] = "system"
    config.domains[filename] = DC(**data)

    return {"status": "promoted", "filename": filename, "new_tier": "system"}


@router.post("/domains/move-skill")
async def move_domain_skill(
    body: dict,
    request: Request,
    user_id: CurrentUserId,
) -> dict:
    """Move a skill from one domain to another."""
    import shutil

    skill_name = body.get("skill_name")
    from_domain = body.get("from_domain", "")
    to_domain = body.get("to_domain", "")

    # "root", "user", "system", "global" are synthetic — treat as user directory
    if from_domain in ("root", "user", "system", "global"):
        from_domain = ""
    if to_domain in ("root", "user", "system", "global"):
        to_domain = ""

    if not skill_name:
        raise HTTPException(status_code=400, detail="skill_name is required")

    # Ownership check on source and target domains
    config: Config = request.app.state.config
    server_config = get_server_config(request)
    for domain_key in (from_domain, to_domain):
        if domain_key:
            dcfg = config.load_domain(domain_key)
            if dcfg and not _can_modify_domain(dcfg, user_id, server_config):
                raise HTTPException(status_code=403, detail=f"You do not have permission to modify domain: {domain_key}")

    from constat.server.routes.skills import get_skill_manager
    manager = get_skill_manager(user_id, server_config.data_dir)

    skill = manager.get_skill(skill_name)
    # Also try finding by directory name (filename) since tree lists dir names
    if not skill:
        for s in manager.get_all_skills():
            if s.filename == skill_name:
                skill = s
                break
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    # Find source and target skill directories
    from_cfg = config.load_domain(from_domain) if from_domain else None
    to_cfg = config.load_domain(to_domain) if to_domain else None

    if from_domain and not from_cfg:
        raise HTTPException(status_code=404, detail=f"Source domain not found: {from_domain}")
    if to_domain and not to_cfg:
        raise HTTPException(status_code=404, detail=f"Target domain not found: {to_domain}")

    # Resolve the skills directory for a domain config
    def _skills_dir(cfg: Optional[object]) -> Path:
        if cfg and getattr(cfg, "source_path", None):
            return Path(cfg.source_path).parent / "skills"
        return manager.skills_dir

    # Determine source skill dir
    src_skill_dir = _skills_dir(from_cfg) / skill.filename

    if not src_skill_dir.exists():
        raise HTTPException(status_code=404, detail=f"Skill directory not found: {src_skill_dir}")

    # Determine target skill dir
    target_skills_dir = _skills_dir(to_cfg)

    target_skills_dir.mkdir(parents=True, exist_ok=True)
    target_skill_dir = target_skills_dir / skill.filename

    # Copy to target
    shutil.copytree(str(src_skill_dir), str(target_skill_dir), dirs_exist_ok=True)
    # Remove from source
    shutil.rmtree(str(src_skill_dir))

    # Reload
    manager.reload()

    return {"status": "moved", "skill": skill_name, "to_domain": to_domain}


@router.post("/domains/move-agent")
async def move_domain_agent(
    body: dict,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Move an agent from one domain to another."""
    import yaml

    agent_name = body.get("agent_name")
    from_domain = body.get("from_domain", "")
    to_domain = body.get("to_domain", "")

    # "root", "user", "system", "global" are synthetic — treat as user directory
    if from_domain in ("root", "user", "system", "global"):
        from_domain = ""
    if to_domain in ("root", "user", "system", "global"):
        to_domain = ""

    if not agent_name:
        raise HTTPException(status_code=400, detail="agent_name is required")

    # Ownership check on source and target domains
    server_config = get_server_config(request)
    for domain_key in (from_domain, to_domain):
        if domain_key:
            dcfg = config.load_domain(domain_key)
            if dcfg and not _can_modify_domain(dcfg, user_id, server_config):
                raise HTTPException(status_code=403, detail=f"You do not have permission to modify domain: {domain_key}")

    from_cfg = config.load_domain(from_domain) if from_domain else None
    to_cfg = config.load_domain(to_domain) if to_domain else None

    if from_domain and not from_cfg:
        raise HTTPException(status_code=404, detail=f"Source domain not found: {from_domain}")
    if to_domain and not to_cfg:
        raise HTTPException(status_code=404, detail=f"Target domain not found: {to_domain}")

    # Resolve agents.yaml path for a domain config
    def _agents_path(cfg: Optional[object]) -> Path:
        if cfg and getattr(cfg, "source_path", None):
            return Path(cfg.source_path).parent / "agents.yaml"
        return Path(".constat") / user_id / "agents.yaml"

    # Read source agents.yaml
    src_agents_file = _agents_path(from_cfg)

    if not src_agents_file.exists():
        raise HTTPException(status_code=404, detail="Source agents file not found")

    src_data = yaml.safe_load(src_agents_file.read_text()) or {}
    if agent_name not in src_data:
        raise HTTPException(status_code=404, detail=f"Agent not found in source: {agent_name}")

    agent_data = src_data.pop(agent_name)

    # Write to target agents.yaml
    tgt_agents_file = _agents_path(to_cfg)

    tgt_data = {}
    if tgt_agents_file.exists():
        tgt_data = yaml.safe_load(tgt_agents_file.read_text()) or {}

    tgt_data[agent_name] = agent_data

    tgt_agents_file.parent.mkdir(parents=True, exist_ok=True)
    tgt_agents_file.write_text(yaml.dump(tgt_data, default_flow_style=False, sort_keys=False))
    src_agents_file.write_text(yaml.dump(src_data, default_flow_style=False, sort_keys=False))

    return {"status": "moved", "agent": agent_name, "to_domain": to_domain}


@router.post("/domains/move-rule")
async def move_domain_rule(
    body: dict,
    request: Request,
    config: Config = Depends(get_config),
    user_id: CurrentUserId = "",
) -> dict:
    """Move a rule to a different domain.

    Physically relocates the rule entry from the source learnings.yaml
    to the target domain's learnings.yaml (alongside agents.yaml pattern).
    """
    import yaml as _yaml

    rule_id = body.get("rule_id")
    to_domain = body.get("to_domain", "")

    if not rule_id:
        raise HTTPException(status_code=400, detail="rule_id is required")

    # Ownership check on target domain
    server_config = get_server_config(request)
    to_cfg = None
    if to_domain:
        to_cfg = config.load_domain(to_domain)
        if to_cfg and not _can_modify_domain(to_cfg, user_id, server_config):
            raise HTTPException(status_code=403, detail=f"You do not have permission to modify domain: {to_domain}")
        if not to_cfg:
            raise HTTPException(status_code=404, detail=f"Target domain not found: {to_domain}")

    # Find and remove rule from current location
    # Check user learnings.yaml first
    from constat.storage.learnings import LearningStore
    user_store = LearningStore(user_id=user_id)
    user_data = user_store._load()

    rule_entry = None
    source_store = None

    if rule_id in user_data["rules"]:
        rule_entry = user_data["rules"].pop(rule_id)
        source_store = user_store
    else:
        # Search domain learnings.yaml files
        for domain_node in config.get_domain_tree():
            dcfg = config.load_domain(domain_node.get("filename", ""))
            if dcfg and dcfg.source_path:
                domain_learnings = Path(dcfg.source_path).parent / "learnings.yaml"
                if domain_learnings.exists():
                    ddata = _yaml.safe_load(domain_learnings.read_text()) or {}
                    if rule_id in ddata.get("rules", {}):
                        rule_entry = ddata["rules"].pop(rule_id)
                        domain_learnings.write_text(
                            _yaml.dump(ddata, default_flow_style=False, sort_keys=False)
                        )
                        break

    if rule_entry is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    # Write rule to target location
    if to_cfg and to_cfg.source_path:
        tgt_file = Path(to_cfg.source_path).parent / "learnings.yaml"
    else:
        # Moving back to user-level
        tgt_file = user_store.file_path

    tgt_data: dict = {}
    if tgt_file.exists():
        tgt_data = _yaml.safe_load(tgt_file.read_text()) or {}
    if "rules" not in tgt_data:
        tgt_data["rules"] = {}
    if "corrections" not in tgt_data:
        tgt_data["corrections"] = {}
    if "archive" not in tgt_data:
        tgt_data["archive"] = {}

    rule_entry["domain"] = to_domain
    tgt_data["rules"][rule_id] = rule_entry

    tgt_file.parent.mkdir(parents=True, exist_ok=True)
    tgt_file.write_text(_yaml.dump(tgt_data, default_flow_style=False, sort_keys=False))

    # Save source if it was the user store
    if source_store is not None:
        source_store._save()

    return {"status": "moved", "rule_id": rule_id, "to_domain": to_domain}


@router.post("/domains/move-source")
async def move_domain_source(
    body: dict,
    config: Config = Depends(get_config),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Move a data source (database/api/document) between domains.

    Reads both domain YAML files, moves the source entry, writes both files,
    reloads the domain configs in-place, and triggers entity refresh.
    """
    import yaml
    from constat.core.config import DomainConfig

    source_type = body.get("source_type")
    source_name = body.get("source_name")
    from_domain = body.get("from_domain")
    to_domain = body.get("to_domain")
    session_id = body.get("session_id")

    if source_type not in ("databases", "apis", "documents"):
        raise HTTPException(status_code=400, detail="source_type must be databases, apis, or documents")
    if not source_name or not from_domain or not to_domain:
        raise HTTPException(status_code=400, detail="source_name, from_domain, to_domain are required")
    if from_domain == to_domain:
        raise HTTPException(status_code=400, detail="from_domain and to_domain must differ")

    # Resolve file paths — "system" maps to root config.yaml
    def _resolve_path(domain_key: str) -> Path:
        if domain_key == "system":
            return Path(config.config_dir) / "config.yaml"
        dcfg = config.domains.get(domain_key)
        if not dcfg:
            raise HTTPException(status_code=404, detail=f"Domain not found: {domain_key}")
        if not dcfg.source_path:
            raise HTTPException(status_code=400, detail=f"Domain has no source path: {domain_key}")
        p = Path(dcfg.source_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Domain file not found: {p}")
        return p

    from_path = _resolve_path(from_domain)
    to_path = _resolve_path(to_domain)

    # Read and parse both files
    from_data = yaml.safe_load(from_path.read_text())
    to_data = yaml.safe_load(to_path.read_text())

    # Verify source exists in from_domain
    from_section = from_data.get(source_type)
    if not isinstance(from_section, dict) or source_name not in from_section:
        raise HTTPException(
            status_code=404,
            detail=f"{source_type}.{source_name} not found in {from_domain}",
        )

    # Verify no naming conflict in to_domain
    to_section = to_data.get(source_type)
    if isinstance(to_section, dict) and source_name in to_section:
        raise HTTPException(
            status_code=409,
            detail=f"{source_type}.{source_name} already exists in {to_domain}",
        )

    # Move the entry
    entry = from_section.pop(source_name)
    if to_section is None:
        to_data[source_type] = {}
    to_data[source_type][source_name] = entry

    # Write both files
    from_path.write_text(yaml.safe_dump(from_data, default_flow_style=False, sort_keys=False))
    to_path.write_text(yaml.safe_dump(to_data, default_flow_style=False, sort_keys=False))

    # Reload domain configs in-place
    for domain_key in (from_domain, to_domain):
        if domain_key == "system":
            continue
        try:
            path = Path(config.domains[domain_key].source_path)
            refreshed = DomainConfig.from_yaml(path)
            config.domains[domain_key] = refreshed
        except Exception as e:
            logger.warning(f"Failed to reload domain {domain_key}: {e}")

    # Trigger entity refresh for the session
    if session_id:
        try:
            session_manager.resolve_config(session_id)
            session_manager.refresh_entities_async(session_id)
        except Exception as e:
            logger.warning(f"Failed to refresh entities for session {session_id}: {e}")

    return {"status": "moved"}


# ============================================================================
# Rule Endpoints
# ============================================================================


@router.post("/rules", response_model=RuleInfo, dependencies=[Depends(require_write("learnings"))])
async def add_rule(
    body: RuleCreateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> RuleInfo:
    """Add a new rule directly.

    Args:
        body: Rule content and metadata
        user_id: Authenticated user ID
        _config: Injected application config

    Returns:
        Created rule
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)

        # Map string category to enum
        try:
            category = LearningCategory(body.category)
        except ValueError:
            category = LearningCategory.USER_CORRECTION

        rule_id = store.save_rule(
            summary=body.summary,
            category=category,
            confidence=body.confidence,
            source_learnings=[],  # User-created rules have no source learnings
            tags=body.tags,
        )

        return RuleInfo(
            id=rule_id,
            summary=body.summary,
            category=body.category,
            confidence=body.confidence,
            source_count=0,
            tags=body.tags,
        )
    except Exception as e:
        logger.error(f"Error creating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}", response_model=RuleInfo, dependencies=[Depends(require_write("learnings"))])
async def update_rule(
    rule_id: str,
    body: RuleUpdateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> RuleInfo:
    """Update an existing rule.

    Args:
        rule_id: Rule ID to update
        body: Fields to update
        user_id: Authenticated user ID
        _config: Injected application config

    Returns:
        Updated rule

    Raises:
        404: Rule not found
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)

        # Check if rule exists
        rules = store.list_rules()
        existing = next((r for r in rules if r["id"] == rule_id), None)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

        # Update rule
        success = store.update_rule(
            rule_id=rule_id,
            summary=body.summary,
            tags=body.tags,
            confidence=body.confidence,
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

        # Fetch updated rule
        rules = store.list_rules()
        updated = next((r for r in rules if r["id"] == rule_id), None)

        return RuleInfo(
            id=rule_id,
            summary=updated["summary"] if updated else body.summary or existing["summary"],
            category=updated["category"] if updated else existing["category"],
            confidence=updated["confidence"] if updated else body.confidence or existing["confidence"],
            source_count=len(updated.get("source_learnings", [])) if updated else existing.get("source_count", 0),
            tags=updated["tags"] if updated else body.tags or existing.get("tags", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}", dependencies=[Depends(require_write("learnings"))])
async def delete_rule(
    rule_id: str,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> dict:
    """Delete a rule.

    Args:
        rule_id: Rule ID to delete
        user_id: Authenticated user ID
        _config: Injected application config

    Returns:
        Deletion confirmation

    Raises:
        404: Rule not found
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)
        if store.delete_rule(rule_id):
            return {"status": "deleted", "id": rule_id}
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))
