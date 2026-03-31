# Copyright (c) 2025 Kenneth Stott
# Canary: 134f76c0-e417-4346-9126-7bb5015bb711
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for domain management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import strawberry

from constat.core.paths import user_vault_dir
from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    CreateDomainInput,
    CreateDomainResultType,
    DeleteDomainResultType,
    DomainAgentType,
    DomainContentSaveResultType,
    DomainContentType,
    DomainDetailType,
    DomainFactType,
    DomainInfoType,
    DomainListType,
    DomainRuleType,
    DomainSkillType,
    DomainTreeNodeType,
    MoveDomainAgentInput,
    MoveDomainAgentResultType,
    MoveDomainRuleInput,
    MoveDomainRuleResultType,
    MoveDomainSkillInput,
    MoveDomainSkillResultType,
    MoveDomainSourceInput,
    MoveDomainSourceResultType,
    PromoteDomainResultType,
    UpdateDomainInput,
    UpdateDomainResultType,
)
from constat.server.permissions import get_user_permissions
from constat.server.routes.learnings import (
    _can_modify_domain,
    _check_domain_cycles,
    _domain_data_dirs,
    _ensure_user_domain_config,
)

logger = logging.getLogger(__name__)


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


@strawberry.type
class Query:
    @strawberry.field
    async def domains(self, info: Info) -> DomainListType:
        user_id = _require_auth(info)
        config = info.context.config
        _ensure_user_domain_config(user_id, config)

        domain_infos = config.list_domains()
        for d in domain_infos:
            d.setdefault("tier", "system")
            d.setdefault("active", True)
            d.setdefault("owner", "")

        # Scan shared domains
        shared_domains_dir = Path(".constat") / "shared" / "domains"
        if shared_domains_dir.is_dir():
            import yaml
            for domain_dir in sorted(shared_domains_dir.iterdir()):
                config_file = domain_dir / "config.yaml"
                if not domain_dir.is_dir() or not config_file.exists():
                    continue
                filename = domain_dir.name
                if filename in config.domains:
                    continue
                try:
                    data = yaml.safe_load(config_file.read_text()) or {}
                    domain_infos.append({
                        "filename": filename,
                        "name": data.get("name", filename),
                        "description": data.get("description", ""),
                        "tier": "shared",
                        "active": data.get("active", True),
                        "owner": data.get("owner", ""),
                        "steward": data.get("steward", ""),
                    })
                    from constat.core.config import DomainConfig
                    data["filename"] = filename
                    data["source_path"] = str(config_file.resolve())
                    data["tier"] = "shared"
                    config.domains[filename] = DomainConfig(**data)
                except Exception as e:
                    logger.warning(f"Failed to load shared domain {filename}: {e}")

        # Scan user domains
        user_domains_dir = user_vault_dir(Path(".constat"), user_id) / "domains"
        if user_domains_dir.is_dir():
            import yaml
            for domain_dir in sorted(user_domains_dir.iterdir()):
                config_file = domain_dir / "config.yaml"
                if not domain_dir.is_dir() or not config_file.exists():
                    continue
                filename = domain_dir.name
                if filename in config.domains:
                    continue
                try:
                    data = yaml.safe_load(config_file.read_text()) or {}
                    domain_infos.append({
                        "filename": filename,
                        "name": data.get("name", filename),
                        "description": data.get("description", ""),
                        "tier": "user",
                        "active": data.get("active", True),
                        "owner": user_id,
                        "steward": data.get("steward", ""),
                    })
                    from constat.core.config import DomainConfig
                    data["filename"] = filename
                    data["source_path"] = str(config_file.resolve())
                    data["tier"] = "user"
                    data["owner"] = user_id
                    config.domains[filename] = DomainConfig(**data)
                except Exception as e:
                    logger.warning(f"Failed to load user domain {filename}: {e}")

        return DomainListType(
            domains=[
                DomainInfoType(
                    filename=d.get("filename", ""),
                    name=d.get("name", ""),
                    description=d.get("description", ""),
                    path=d.get("path", ""),
                    tier=d.get("tier", "system"),
                    active=d.get("active", True),
                    owner=d.get("owner", ""),
                    steward=d.get("steward", ""),
                )
                for d in domain_infos
            ]
        )

    @strawberry.field
    async def domain_tree(self, info: Info) -> list[DomainTreeNodeType]:
        user_id = _require_auth(info)
        config = info.context.config

        from constat.core.config import DomainConfig

        def _build_node(domain_cfg: DomainConfig, tier: str = "system") -> DomainTreeNodeType:
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

            # User LearningStore rules tagged with this domain
            try:
                from constat.storage.learnings import LearningStore
                store = LearningStore(user_id=user_id)
                for r in store.list_rules(domain=domain_cfg.filename):
                    if r["id"] not in seen_rules:
                        seen_rules.add(r["id"])
                        rule_ids.append(r["id"])
            except Exception:
                pass

            # User FactStore facts tagged with this domain
            try:
                from constat.storage.facts import FactStore
                fact_store = FactStore(user_id=user_id)
                for fname, fdata in fact_store.list_all_facts().items():
                    if fdata.get("domain") == domain_cfg.filename and fname not in seen_facts:
                        seen_facts.add(fname)
                        fact_names.append(fname)
            except Exception:
                pass

            return DomainTreeNodeType(
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
        all_configs: dict[str, tuple[DomainConfig, str]] = {}
        all_nodes: dict[str, DomainTreeNodeType] = {}

        def _register(canonical: str, cfg: DomainConfig, tier: str) -> None:
            if canonical in all_configs:
                return
            cfg.filename = canonical
            all_configs[canonical] = (cfg, tier)
            node = _build_node(cfg, tier=tier)
            node.filename = canonical
            all_nodes[canonical] = node

        def _flat_scan(dir_path: Path, prefix: str = "", tier: str = "system") -> None:
            if not dir_path.is_dir():
                return
            for entry in sorted(dir_path.iterdir()):
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
                    sub_dir = entry / "domains"
                    if sub_dir.is_dir():
                        _flat_scan(sub_dir, prefix=canonical, tier=tier)

        # Scan system domains
        if config.config_dir:
            domains_dir = Path(config.config_dir) / "domains"
            _flat_scan(domains_dir)

        # Normalize filenames
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
        user_domains_dir = user_vault_dir(Path(".constat"), user_id) / "domains"
        _flat_scan(user_domains_dir, tier="user")

        def _resolve_ref(ref: str, parent_key: str = "") -> str | None:
            if ref in all_configs:
                return ref
            if parent_key:
                relative = f"{parent_key}/{ref}"
                if relative in all_configs:
                    return relative
            return None

        # --- Phase 2: Cycle detection ---
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

        for fname, (cfg, _) in all_configs.items():
            node = all_nodes[fname]
            for child_ref in (cfg.domains or []):
                child_name = _resolve_ref(child_ref, parent_key=fname)
                if child_name and child_name in all_nodes and (fname, child_name) not in bad_edges:
                    node.children.append(all_nodes[child_name])

        top_level = [all_nodes[fname] for fname in all_nodes if fname not in referenced]

        # --- Phase 4: Root wrapper ---
        user_skill_names: list[str] = []
        user_agent_names: list[str] = []
        user_rule_ids: list[str] = []
        user_fact_names: list[str] = []
        user_skills_dir = user_vault_dir(Path(".constat"), user_id) / "skills"
        if user_skills_dir.is_dir():
            for sd in user_skills_dir.iterdir():
                if sd.is_dir() and (sd / "SKILL.md").exists():
                    user_skill_names.append(sd.name)
        user_agents_file = user_vault_dir(Path(".constat"), user_id) / "agents.yaml"
        if user_agents_file.is_file():
            try:
                import yaml as _yaml
                agent_data = _yaml.safe_load(user_agents_file.read_text()) or {}
                user_agent_names = list(agent_data.keys())
            except Exception:
                pass
        try:
            from constat.storage.learnings import LearningStore
            user_store = LearningStore(user_id=user_id)
            for r in user_store.list_rules(domain=""):
                user_rule_ids.append(r["id"])
        except Exception:
            pass
        try:
            from constat.storage.facts import FactStore
            user_fact_store = FactStore(user_id=user_id)
            for fname, fdata in user_fact_store.list_all_facts().items():
                if not fdata.get("domain"):
                    user_fact_names.append(fname)
        except Exception:
            pass

        user_node = DomainTreeNodeType(
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

        user_tier_domains = [all_nodes[f] for f in all_nodes
                             if all_configs[f][1] == "user" and f not in referenced]
        system_top_level = [n for n in top_level if n not in user_tier_domains]
        user_node.children = user_tier_domains

        root_node = DomainTreeNodeType(
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

    @strawberry.field
    async def domain(self, info: Info, filename: str) -> DomainDetailType:
        config = info.context.config
        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")

        return DomainDetailType(
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

    @strawberry.field
    async def domain_content(self, info: Info, filename: str) -> DomainContentType:
        user_id = _require_auth(info)
        config = info.context.config
        import yaml

        if filename == "root":
            if not config.config_dir:
                raise ValueError("No config directory configured")
            domain_path = Path(config.config_dir) / "config.yaml"
            if not domain_path.exists():
                raise ValueError(f"Root config not found: {domain_path}")
            return DomainContentType(content=domain_path.read_text(), path=str(domain_path), filename="root")

        if filename == "user":
            user_dir = user_vault_dir(Path(".constat"), user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            domain_path = user_dir / "config.yaml"
            if not domain_path.exists():
                default_content = yaml.dump(
                    {"name": "User", "description": "", "domains": []},
                    default_flow_style=False, sort_keys=False,
                )
                domain_path.write_text(default_content)
            return DomainContentType(content=domain_path.read_text(), path=str(domain_path), filename="user")

        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")
        if not domain.source_path:
            raise ValueError("Domain has no source path (inline config)")
        domain_path = Path(domain.source_path)
        if not domain_path.exists():
            raise ValueError(f"Domain file not found: {domain_path}")
        return DomainContentType(content=domain_path.read_text(), path=str(domain_path), filename=filename)

    @strawberry.field
    async def domain_skills(self, info: Info, filename: str) -> list[DomainSkillType]:
        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        from constat.server.routes.skills import get_skill_manager
        manager = get_skill_manager(user_id, server_config.data_dir)

        seen: set[str] = set()
        skills_out: list[DomainSkillType] = []

        domain_cfg = config.load_domain(filename)
        if domain_cfg:
            for d in _domain_data_dirs(domain_cfg, user_id):
                skills_dir = d / "skills"
                if skills_dir.is_dir():
                    for sd in skills_dir.iterdir():
                        if sd.is_dir() and (sd / "SKILL.md").exists() and sd.name not in seen:
                            seen.add(sd.name)
                            skills_out.append(DomainSkillType(name=sd.name, description="", domain=filename))

        for s in manager.get_domain_skills(filename):
            if s.name not in seen:
                seen.add(s.name)
                skills_out.append(DomainSkillType(name=s.name, description=s.description, domain=s.domain))

        return skills_out

    @strawberry.field
    async def domain_agents(self, info: Info, filename: str) -> list[DomainAgentType]:
        user_id = _require_auth(info)
        config = info.context.config
        sm = info.context.session_manager
        import yaml as _yaml

        seen: set[str] = set()
        agents_out: list[DomainAgentType] = []

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
                                agents_out.append(DomainAgentType(
                                    name=name,
                                    description=adata.get("description", ""),
                                    domain=filename,
                                ))
                    except Exception:
                        pass

        sessions = list(sm.sessions.values())
        for managed in sessions:
            if hasattr(managed.session, "agent_manager"):
                for a in managed.session.agent_manager.get_domain_agents(filename):
                    if a.name not in seen:
                        seen.add(a.name)
                        agents_out.append(DomainAgentType(name=a.name, description=a.description, domain=a.domain))
                break

        return agents_out

    @strawberry.field
    async def domain_rules(self, info: Info, filename: str) -> list[DomainRuleType]:
        user_id = _require_auth(info)
        config = info.context.config
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

        domain_cfg = config.load_domain(filename)
        if domain_cfg:
            for d in _domain_data_dirs(domain_cfg, user_id):
                _collect_from_file(d / "learnings.yaml")

        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        for r in store.list_rules(domain=filename):
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                rules.append(r)

        return [
            DomainRuleType(
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

    @strawberry.field
    async def domain_facts(self, info: Info, filename: str) -> list[DomainFactType]:
        user_id = _require_auth(info)
        config = info.context.config
        import yaml as _yaml

        seen: set[str] = set()
        facts_out: list[DomainFactType] = []

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
                                if isinstance(fval, dict):
                                    facts_out.append(DomainFactType(
                                        name=fname,
                                        value=fval.get("value"),
                                        domain=filename,
                                        source=fval.get("source"),
                                        confidence=fval.get("confidence"),
                                    ))
                                else:
                                    facts_out.append(DomainFactType(name=fname, value=fval, domain=filename))
                    except Exception:
                        pass

        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=user_id)
        for fname, fdata in fact_store.list_all_facts().items():
            if fdata.get("domain") == filename and fname not in seen:
                seen.add(fname)
                facts_out.append(DomainFactType(
                    name=fname,
                    value=fdata.get("value"),
                    domain=filename,
                    source=fdata.get("source"),
                    confidence=fdata.get("confidence"),
                ))

        return facts_out


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_domain(self, info: Info, input: CreateDomainInput) -> CreateDomainResultType:
        import re
        import yaml

        user_id = _require_auth(info)
        config = info.context.config

        name = (input.name or "").strip()
        if not name:
            raise ValueError("Name is required")

        description = (input.description or "").strip()
        system_prompt = (input.system_prompt or "").strip()
        parent_domain = (input.parent_domain or "").strip()
        initial_domains: list[str] = input.initial_domains or []

        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            raise ValueError("Invalid name")
        filename = slug

        if filename in config.domains:
            raise ValueError(f"Domain '{filename}' already exists")

        user_domains_dir = user_vault_dir(Path(".constat"), user_id) / "domains"
        domain_dir = user_domains_dir / slug
        domain_path = domain_dir / "config.yaml"

        if domain_dir.exists():
            raise ValueError(f"Domain '{filename}' already exists")

        domain_dir.mkdir(parents=True, exist_ok=True)

        content: dict = {
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

        from constat.core.config import DomainConfig
        content["filename"] = filename
        content["source_path"] = str(domain_path.resolve())
        config.domains[filename] = DomainConfig(**content)

        if parent_domain and parent_domain != "user":
            if parent_domain == "root":
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

        return CreateDomainResultType(
            status="created", filename=filename, name=name, description=description,
        )

    @strawberry.mutation
    async def update_domain(
        self, info: Info, filename: str, input: UpdateDomainInput,
    ) -> UpdateDomainResultType:
        import yaml

        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")

        if not _can_modify_domain(domain, user_id, server_config):
            raise ValueError("You do not have permission to modify this domain")

        perms = get_user_permissions(server_config, user_id)
        if domain.tier == "system" and not perms.is_admin:
            fields = {}
            if input.name is not None:
                fields["name"] = input.name
            if input.description is not None:
                fields["description"] = input.description
            if input.order is not None:
                fields["order"] = input.order
            disallowed = {k for k in fields if k not in ("active",)}
            if disallowed:
                raise ValueError(
                    f"System domains only allow toggling 'active'. Cannot modify: {', '.join(disallowed)}"
                )

        if not domain.source_path:
            raise ValueError("Domain has no source path")

        domain_path = Path(domain.source_path)
        if not domain_path.exists():
            raise ValueError(f"Domain file not found: {domain_path}")

        data = yaml.safe_load(domain_path.read_text()) or {}

        allowed_fields = {"name", "description", "order", "active"}
        updates = {}
        if input.name is not None:
            updates["name"] = input.name
        if input.description is not None:
            updates["description"] = input.description
        if input.order is not None:
            updates["order"] = input.order
        if input.active is not None:
            updates["active"] = input.active

        for key in allowed_fields:
            if key in updates:
                data[key] = updates[key]

        domain_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

        for key in allowed_fields:
            if key in updates and hasattr(domain, key):
                setattr(domain, key, updates[key])

        return UpdateDomainResultType(status="updated", filename=filename)

    @strawberry.mutation
    async def delete_domain(self, info: Info, filename: str) -> DeleteDomainResultType:
        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")

        if not _can_modify_domain(domain, user_id, server_config):
            raise ValueError("You do not have permission to delete this domain")

        referencing = [
            k for k, d in config.domains.items()
            if k != filename and filename in (d.domains or [])
        ]
        if referencing:
            raise ValueError(f"Cannot delete: domain is composed by {', '.join(referencing)}")

        if not domain.source_path:
            raise ValueError("Domain has no source path")

        import shutil
        domain_path = Path(domain.source_path)
        domain_dir = domain_path.parent

        if domain_dir.is_dir() and (domain_dir / "config.yaml").exists():
            shutil.rmtree(domain_dir)
        elif domain_path.exists():
            domain_path.unlink()

        config.domains.pop(filename, None)
        return DeleteDomainResultType(status="deleted", filename=filename)

    @strawberry.mutation
    async def update_domain_content(
        self, info: Info, filename: str, content: str,
    ) -> DomainContentSaveResultType:
        import yaml

        user_id = _require_auth(info)
        config = info.context.config

        if filename == "root":
            if not config.config_dir:
                raise ValueError("No config directory configured")
            domain_path = Path(config.config_dir) / "config.yaml"
            if not domain_path.exists():
                raise ValueError(f"Root config not found: {domain_path}")
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML: {e}")
            domain_path.write_text(content)
            return DomainContentSaveResultType(status="saved", filename="root", path=str(domain_path))

        if filename == "user":
            user_dir = user_vault_dir(Path(".constat"), user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            domain_path = user_dir / "config.yaml"
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML: {e}")
            domain_path.write_text(content)
            return DomainContentSaveResultType(status="saved", filename="user", path=str(domain_path))

        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")
        if not domain.source_path:
            raise ValueError("Domain has no source path (inline config)")
        domain_path = Path(domain.source_path)
        if not domain_path.exists():
            raise ValueError(f"Domain file not found: {domain_path}")

        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

        # Cycle check
        new_domains = (parsed or {}).get("domains", []) if isinstance(parsed, dict) else []
        if new_domains:
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
                raise ValueError(f"Cannot save: would create domain cycle(s): {'; '.join(cycles)}")

        domain_path.write_text(content)

        if isinstance(parsed, dict) and "domains" in parsed:
            domain.domains = parsed["domains"]

        return DomainContentSaveResultType(status="saved", filename=filename, path=str(domain_path))

    @strawberry.mutation
    async def promote_domain(self, info: Info, filename: str) -> PromoteDomainResultType:
        import shutil
        import yaml

        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        domain = config.load_domain(filename)
        if not domain:
            raise ValueError(f"Domain not found: {filename}")

        if domain.tier != "user":
            raise ValueError("Only user domains can be moved to root")

        if not _can_modify_domain(domain, user_id, server_config):
            raise ValueError("You do not have permission to move this domain")

        if not domain.source_path:
            raise ValueError("Domain has no source path")

        src_config = Path(domain.source_path)
        src_dir = src_config.parent
        if not src_dir.exists():
            raise ValueError(f"Domain directory not found: {src_dir}")

        if not config.config_dir:
            raise ValueError("No config directory configured")

        system_domains_dir = Path(config.config_dir) / "domains"
        system_domains_dir.mkdir(parents=True, exist_ok=True)
        target_dir = system_domains_dir / filename

        if target_dir.exists():
            raise ValueError(f"Domain '{filename}' already exists under root")

        shutil.move(str(src_dir), str(target_dir))

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

        from constat.core.config import DomainConfig as DC
        new_config_path = target_dir / "config.yaml"
        data = yaml.safe_load(new_config_path.read_text()) or {}
        data["filename"] = filename
        data["source_path"] = str(new_config_path.resolve())
        data["tier"] = "system"
        config.domains[filename] = DC(**data)

        return PromoteDomainResultType(status="promoted", filename=filename, new_tier="system")

    @strawberry.mutation
    async def move_domain_skill(
        self, info: Info, input: MoveDomainSkillInput,
    ) -> MoveDomainSkillResultType:
        import shutil

        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        skill_name = input.skill_name
        from_domain = input.from_domain
        to_domain = input.to_domain
        validate_only = input.validate_only or False

        if from_domain in ("root", "user", "system", "global"):
            from_domain = ""
        if to_domain in ("root", "user", "system", "global"):
            to_domain = ""

        # Ownership check
        for domain_key in (from_domain, to_domain):
            if domain_key:
                dcfg = config.load_domain(domain_key)
                if dcfg and not _can_modify_domain(dcfg, user_id, server_config):
                    raise ValueError(f"You do not have permission to modify domain: {domain_key}")

        from constat.server.routes.skills import get_skill_manager
        manager = get_skill_manager(user_id, server_config.data_dir)

        skill = manager.get_skill(skill_name)
        if not skill:
            for s in manager.get_all_skills():
                if s.filename == skill_name:
                    skill = s
                    break
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        from constat.core.resource_validation import validate_resource_compatibility

        to_cfg = config.load_domain(to_domain) if to_domain else None
        warnings: list[str] = []
        if skill.required_resources and to_cfg:
            warnings = validate_resource_compatibility(skill.required_resources, to_cfg, to_domain)

        if validate_only:
            return MoveDomainSkillResultType(status="validation", warnings=warnings)

        from_cfg = config.load_domain(from_domain) if from_domain else None

        if from_domain and not from_cfg:
            raise ValueError(f"Source domain not found: {from_domain}")
        if to_domain and not to_cfg:
            raise ValueError(f"Target domain not found: {to_domain}")

        def _skills_dir(cfg) -> Path:
            if cfg and getattr(cfg, "source_path", None):
                return Path(cfg.source_path).parent / "skills"
            return manager.skills_dir

        src_skill_dir = _skills_dir(from_cfg) / skill.filename
        if not src_skill_dir.exists():
            raise ValueError(f"Skill directory not found: {src_skill_dir}")

        target_skills_dir = _skills_dir(to_cfg)
        target_skills_dir.mkdir(parents=True, exist_ok=True)
        target_skill_dir = target_skills_dir / skill.filename

        shutil.copytree(str(src_skill_dir), str(target_skill_dir), dirs_exist_ok=True)
        shutil.rmtree(str(src_skill_dir))
        manager.reload()

        return MoveDomainSkillResultType(
            status="moved", skill=skill_name, to_domain=to_domain, warnings=warnings,
        )

    @strawberry.mutation
    async def move_domain_agent(
        self, info: Info, input: MoveDomainAgentInput,
    ) -> MoveDomainAgentResultType:
        import yaml

        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        agent_name = input.agent_name
        from_domain = input.from_domain
        to_domain = input.to_domain

        if from_domain in ("root", "user", "system", "global"):
            from_domain = ""
        if to_domain in ("root", "user", "system", "global"):
            to_domain = ""

        # Ownership check
        for domain_key in (from_domain, to_domain):
            if domain_key:
                dcfg = config.load_domain(domain_key)
                if dcfg and not _can_modify_domain(dcfg, user_id, server_config):
                    raise ValueError(f"You do not have permission to modify domain: {domain_key}")

        from_cfg = config.load_domain(from_domain) if from_domain else None
        to_cfg = config.load_domain(to_domain) if to_domain else None

        if from_domain and not from_cfg:
            raise ValueError(f"Source domain not found: {from_domain}")
        if to_domain and not to_cfg:
            raise ValueError(f"Target domain not found: {to_domain}")

        def _agents_path(cfg) -> Path:
            if cfg and getattr(cfg, "source_path", None):
                return Path(cfg.source_path).parent / "agents.yaml"
            return user_vault_dir(Path(".constat"), user_id) / "agents.yaml"

        src_agents_file = _agents_path(from_cfg)
        if not src_agents_file.exists():
            raise ValueError("Source agents file not found")

        src_data = yaml.safe_load(src_agents_file.read_text()) or {}
        if agent_name not in src_data:
            raise ValueError(f"Agent not found in source: {agent_name}")

        agent_data = src_data.pop(agent_name)

        tgt_agents_file = _agents_path(to_cfg)
        tgt_data = {}
        if tgt_agents_file.exists():
            tgt_data = yaml.safe_load(tgt_agents_file.read_text()) or {}

        tgt_data[agent_name] = agent_data
        tgt_agents_file.parent.mkdir(parents=True, exist_ok=True)
        tgt_agents_file.write_text(yaml.dump(tgt_data, default_flow_style=False, sort_keys=False))
        src_agents_file.write_text(yaml.dump(src_data, default_flow_style=False, sort_keys=False))

        return MoveDomainAgentResultType(status="moved", agent=agent_name, to_domain=to_domain)

    @strawberry.mutation
    async def move_domain_rule(
        self, info: Info, input: MoveDomainRuleInput,
    ) -> MoveDomainRuleResultType:
        import yaml as _yaml

        user_id = _require_auth(info)
        config = info.context.config
        server_config = info.context.server_config

        rule_id = input.rule_id
        to_domain = input.to_domain

        # Ownership check on target domain
        to_cfg = None
        if to_domain:
            to_cfg = config.load_domain(to_domain)
            if to_cfg and not _can_modify_domain(to_cfg, user_id, server_config):
                raise ValueError(f"You do not have permission to modify domain: {to_domain}")
            if not to_cfg:
                raise ValueError(f"Target domain not found: {to_domain}")

        from constat.storage.learnings import LearningStore
        user_store = LearningStore(user_id=user_id)
        user_data = user_store._load()

        rule_entry = None
        source_store = None

        if rule_id in user_data["rules"]:
            rule_entry = user_data["rules"].pop(rule_id)
            source_store = user_store
        else:
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
            raise ValueError(f"Rule not found: {rule_id}")

        if to_cfg and to_cfg.source_path:
            tgt_file = Path(to_cfg.source_path).parent / "learnings.yaml"
        else:
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

        if source_store is not None:
            source_store._save()

        return MoveDomainRuleResultType(status="moved", rule_id=rule_id, to_domain=to_domain)

    @strawberry.mutation
    async def move_domain_source(
        self, info: Info, input: MoveDomainSourceInput,
    ) -> MoveDomainSourceResultType:
        import yaml
        from constat.core.config import DomainConfig

        config = info.context.config
        sm = info.context.session_manager

        source_type = input.source_type
        source_name = input.source_name
        from_domain = input.from_domain
        to_domain = input.to_domain
        session_id = input.session_id

        if source_type not in ("databases", "apis", "documents"):
            raise ValueError("source_type must be databases, apis, or documents")
        if not source_name or not from_domain or not to_domain:
            raise ValueError("source_name, from_domain, to_domain are required")
        if from_domain == to_domain:
            raise ValueError("from_domain and to_domain must differ")

        def _resolve_path(domain_key: str) -> Path:
            if domain_key == "system":
                return Path(config.config_dir) / "config.yaml"
            dcfg = config.domains.get(domain_key)
            if not dcfg:
                raise ValueError(f"Domain not found: {domain_key}")
            if not dcfg.source_path:
                raise ValueError(f"Domain has no source path: {domain_key}")
            p = Path(dcfg.source_path)
            if not p.exists():
                raise ValueError(f"Domain file not found: {p}")
            return p

        from_path = _resolve_path(from_domain)
        to_path = _resolve_path(to_domain)

        from_data = yaml.safe_load(from_path.read_text())
        to_data = yaml.safe_load(to_path.read_text())

        from_section = from_data.get(source_type)
        if not isinstance(from_section, dict) or source_name not in from_section:
            raise ValueError(f"{source_type}.{source_name} not found in {from_domain}")

        to_section = to_data.get(source_type)
        if isinstance(to_section, dict) and source_name in to_section:
            raise ValueError(f"{source_type}.{source_name} already exists in {to_domain}")

        entry = from_section.pop(source_name)
        if to_section is None:
            to_data[source_type] = {}
        to_data[source_type][source_name] = entry

        from_path.write_text(yaml.safe_dump(from_data, default_flow_style=False, sort_keys=False))
        to_path.write_text(yaml.safe_dump(to_data, default_flow_style=False, sort_keys=False))

        for domain_key in (from_domain, to_domain):
            if domain_key == "system":
                continue
            try:
                path = Path(config.domains[domain_key].source_path)
                refreshed = DomainConfig.from_yaml(path)
                config.domains[domain_key] = refreshed
            except Exception as e:
                logger.warning(f"Failed to reload domain {domain_key}: {e}")

        if session_id:
            try:
                sm.resolve_config(session_id)
                sm.refresh_entities_async(session_id)
            except Exception as e:
                logger.warning(f"Failed to refresh entities for session {session_id}: {e}")

        return MoveDomainSourceResultType(status="moved")
