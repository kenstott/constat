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
    if domain.owner and domain.owner != user_id:
        return False
    return True


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


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
    """Get nested domain hierarchy tree."""
    from constat.core.config import DomainConfig

    def _build_node(domain_cfg: DomainConfig, tier: str = "system") -> DomainTreeNode:
        # Scan for domain-scoped skills (skills/ subdirectory)
        skill_names: list[str] = []
        agent_names: list[str] = []
        rule_ids: list[str] = []
        if domain_cfg.source_path:
            domain_dir = Path(domain_cfg.source_path).parent
            skills_dir = domain_dir / "skills"
            if skills_dir.is_dir():
                for sd in skills_dir.iterdir():
                    if sd.is_dir() and (sd / "SKILL.md").exists():
                        skill_names.append(sd.name)
            agents_file = domain_dir / "agents.yaml"
            if agents_file.is_file():
                try:
                    import yaml as _yaml
                    agent_data = _yaml.safe_load(agents_file.read_text()) or {}
                    agent_names = list(agent_data.keys())
                except Exception:
                    pass

        # Load domain-scoped rules
        try:
            from constat.storage.learnings import LearningStore
            store = LearningStore(user_id=user_id)
            domain_key = domain_cfg.filename
            rules = store.list_rules(domain=domain_key)
            rule_ids = [r["id"] for r in rules]
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
        )

    def _scan_dir(dir_path: Path, parent_path: str = "", tier: str = "system") -> list[DomainTreeNode]:
        nodes: list[DomainTreeNode] = []
        if not dir_path.is_dir():
            return nodes
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir() and (entry / "config.yaml").exists():
                domain_cfg = DomainConfig.from_directory(entry, parent_path=parent_path)
                node = _build_node(domain_cfg, tier=tier)
                sub_dir = entry / "domains"
                if sub_dir.is_dir():
                    node.children = _scan_dir(sub_dir, parent_path=domain_cfg.path, tier=tier)
                nodes.append(node)
            elif entry.is_file() and entry.suffix in (".yaml", ".yml"):
                try:
                    domain_cfg = DomainConfig.from_yaml(entry)
                    domain_cfg.path = f"{parent_path}.{entry.stem}" if parent_path else entry.stem
                    nodes.append(_build_node(domain_cfg, tier=tier))
                except Exception:
                    pass
        return nodes

    # Build tree from config directory's domains/ folder
    result: list[DomainTreeNode] = []
    if config.config_dir:
        domains_dir = Path(config.config_dir) / "domains"
        result.extend(_scan_dir(domains_dir))

    # Normalize filenames to match config.domains keys (the canonical identifiers
    # used by active_domains, preferences, entity domain_ids, etc.)
    # _scan_dir uses the YAML filename (e.g. "sales-analytics.yaml") but
    # config.domains keys omit the suffix (e.g. "sales-analytics").
    if result and config.domains:
        config_keys = set(config.domains.keys())
        def _normalize_filename(node: DomainTreeNode) -> None:
            stem = node.filename.removesuffix(".yaml").removesuffix(".yml")
            if stem in config_keys and node.filename not in config_keys:
                node.filename = stem
            for child in node.children:
                _normalize_filename(child)
        for node in result:
            _normalize_filename(node)

    # If no directory-based domains found, build flat tree from config.domains
    if not result and config.domains:
        for key, domain_cfg in sorted(config.domains.items()):
            node = _build_node(domain_cfg)
            # Use the config dict key as filename — this is what domain_id
            # stores in entities/embeddings and what active_domains contains
            node.filename = key
            result.append(node)

    # Scan shared domains
    shared_domains_dir = Path(".constat") / "shared" / "domains"
    if shared_domains_dir.is_dir():
        existing_filenames = {n.filename for n in result}
        for node in _scan_dir(shared_domains_dir, tier="shared"):
            if node.filename not in existing_filenames:
                result.append(node)

    # Scan user domains
    user_domains_dir = Path(".constat") / user_id / "domains"
    if user_domains_dir.is_dir():
        existing_filenames = {n.filename for n in result}
        for node in _scan_dir(user_domains_dir, tier="user"):
            if node.filename not in existing_filenames:
                result.append(node)

    # Collect user-level (unscoped) skills and agents for the System root node
    user_skill_names: list[str] = []
    user_agent_names: list[str] = []
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

    # Wrap all domains under a System root node — domains are children of system
    if result:
        system_node = DomainTreeNode(
            filename="system",
            name="System",
            path="system",
            description="Root configuration",
            databases=list(config.databases.keys()),
            apis=list(config.apis.keys()),
            documents=[],
            children=result,
            skills=user_skill_names,
            agents=user_agent_names,
        )
        return [system_node]

    return result


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

    # Slugify name to filename
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise HTTPException(status_code=400, detail="Invalid name")
    filename = f"{slug}.yaml"

    # Check for conflicts with system domains
    if filename in config.domains:
        raise HTTPException(status_code=409, detail=f"Domain '{filename}' already exists")

    # Write domain file
    user_domains_dir = Path(".constat") / user_id / "domains"
    user_domains_dir.mkdir(parents=True, exist_ok=True)
    domain_path = user_domains_dir / filename

    if domain_path.exists():
        raise HTTPException(status_code=409, detail=f"Domain '{filename}' already exists")

    content = {
        "name": name,
        "description": description,
        "owner": user_id,
        "databases": {},
        "apis": {},
        "documents": {},
    }
    domain_path.write_text(yaml.dump(content, default_flow_style=False, sort_keys=False))

    # Register in config.domains
    from constat.core.config import DomainConfig
    content["filename"] = filename
    content["source_path"] = str(domain_path.resolve())
    config.domains[filename] = DomainConfig(**content)

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

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")

    domain_path = Path(domain.source_path)
    if domain_path.exists():
        domain_path.unlink()

    # Remove from config
    config.domains.pop(filename, None)

    return {"status": "deleted", "filename": filename}


@router.get("/domains/{filename}/content")
async def get_domain_content(
    filename: str,
    config: Config = Depends(get_config),
) -> dict:
    """Get the raw YAML content of a domain file.

    Args:
        filename: Domain YAML filename
        config: Injected application config

    Returns:
        Dict with 'content' (YAML string) and 'path' (full file path)

    Raises:
        404: Domain not found
    """
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
    config: Config = Depends(get_config),
) -> dict:
    """Update the YAML content of a domain file.

    Args:
        filename: Domain YAML filename
        body: Dict with 'content' (new YAML string)
        config: Injected application config

    Returns:
        Status confirmation

    Raises:
        404: Domain not found
        400: Invalid YAML
    """
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
    import yaml
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # Write the file
    domain_path.write_text(content)

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
) -> dict:
    """List skills belonging to a domain."""
    from constat.server.routes.skills import get_skill_manager
    server_config = get_server_config(request)
    manager = get_skill_manager(user_id, server_config.data_dir)
    skills = manager.get_domain_skills(filename)
    return {
        "skills": [
            {"name": s.name, "description": s.description, "domain": s.domain}
            for s in skills
        ]
    }


@router.get("/domains/{filename}/agents")
async def list_domain_agents(
    filename: str,
    request: Request,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """List agents belonging to a domain."""
    # Get agent_manager from session if available
    agents = []
    sessions = list(session_manager.sessions.values())
    for managed in sessions:
        if hasattr(managed.session, "agent_manager"):
            agents = managed.session.agent_manager.get_domain_agents(filename)
            break
    return {
        "agents": [
            {"name": a.name, "description": a.description, "domain": a.domain}
            for a in agents
        ]
    }


@router.get("/domains/{filename}/rules")
async def list_domain_rules(
    filename: str,
    user_id: CurrentUserId,
) -> dict:
    """List rules belonging to a domain."""
    from constat.storage.learnings import LearningStore
    store = LearningStore(user_id=user_id)
    rules = store.list_rules(domain=filename)
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


@router.post("/domains/{filename}/promote")
async def promote_domain(
    filename: str,
    body: dict,
    request: Request,
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Promote a user domain to shared tier."""
    import shutil
    import yaml

    domain = config.load_domain(filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {filename}")

    if domain.tier != "user":
        raise HTTPException(status_code=400, detail="Only user domains can be promoted")

    server_config = get_server_config(request)
    if not _can_modify_domain(domain, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to promote this domain")

    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")

    src_path = Path(domain.source_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain file not found: {src_path}")

    target_name = body.get("target_name", filename)
    shared_dir = Path(".constat") / "shared" / "domains"
    shared_dir.mkdir(parents=True, exist_ok=True)
    target_path = shared_dir / target_name

    # Copy YAML file
    data = yaml.safe_load(src_path.read_text()) or {}
    data["owner"] = user_id
    target_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    # Copy scoped content if exists (skills/, agents.yaml)
    src_dir = src_path.parent
    tgt_dir = target_path.parent / target_path.stem
    # For file-based domains, scoped content is alongside the yaml
    skills_dir = src_dir / "skills"
    if skills_dir.is_dir():
        tgt_skills = shared_dir / target_path.stem.replace(".yaml", "") / "skills"
        tgt_skills.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(skills_dir), str(tgt_skills), dirs_exist_ok=True)

    agents_file = src_dir / "agents.yaml"
    if agents_file.is_file():
        tgt_agents = shared_dir / target_path.stem.replace(".yaml", "") / "agents.yaml"
        tgt_agents.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(agents_file), str(tgt_agents))

    # Register promoted domain
    from constat.core.config import DomainConfig as DC
    data["filename"] = target_name
    data["source_path"] = str(target_path.resolve())
    data["tier"] = "shared"
    config.domains[target_name] = DC(**data)

    return {"status": "promoted", "filename": target_name, "new_tier": "shared"}


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

    # "system" and "global" are synthetic — treat as user directory
    if from_domain in ("system", "global"):
        from_domain = ""
    if to_domain in ("system", "global"):
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

    # Determine source skill dir
    if from_cfg and from_cfg.source_path:
        src_skill_dir = Path(from_cfg.source_path).parent / "skills" / skill.filename
    else:
        src_skill_dir = manager.skills_dir / skill.filename

    if not src_skill_dir.exists():
        raise HTTPException(status_code=404, detail=f"Skill directory not found: {src_skill_dir}")

    # Determine target skill dir
    if to_cfg and to_cfg.source_path:
        target_skills_dir = Path(to_cfg.source_path).parent / "skills"
    else:
        target_skills_dir = manager.skills_dir

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

    # "system" and "global" are synthetic — treat as user directory
    if from_domain in ("system", "global"):
        from_domain = ""
    if to_domain in ("system", "global"):
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

    # Read source agents.yaml
    if from_cfg and from_cfg.source_path:
        src_agents_file = Path(from_cfg.source_path).parent / "agents.yaml"
    else:
        src_agents_file = Path(".constat") / user_id / "agents.yaml"

    if not src_agents_file.exists():
        raise HTTPException(status_code=404, detail="Source agents file not found")

    src_data = yaml.safe_load(src_agents_file.read_text()) or {}
    if agent_name not in src_data:
        raise HTTPException(status_code=404, detail=f"Agent not found in source: {agent_name}")

    agent_data = src_data.pop(agent_name)

    # Write to target agents.yaml
    if to_cfg and to_cfg.source_path:
        tgt_agents_file = Path(to_cfg.source_path).parent / "agents.yaml"
    else:
        tgt_agents_file = Path(".constat") / user_id / "agents.yaml"

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
    user_id: CurrentUserId,
) -> dict:
    """Move a rule to a different domain."""
    rule_id = body.get("rule_id")
    to_domain = body.get("to_domain", "")

    if not rule_id:
        raise HTTPException(status_code=400, detail="rule_id is required")

    # Ownership check on target domain
    if to_domain:
        config: Config = request.app.state.config
        server_config = get_server_config(request)
        dcfg = config.load_domain(to_domain)
        if dcfg and not _can_modify_domain(dcfg, user_id, server_config):
            raise HTTPException(status_code=403, detail=f"You do not have permission to modify domain: {to_domain}")

    from constat.storage.learnings import LearningStore
    store = LearningStore(user_id=user_id)
    data = store._load()

    if rule_id not in data["rules"]:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    data["rules"][rule_id]["domain"] = to_domain
    store._save()

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
