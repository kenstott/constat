# Copyright (c) 2025 Kenneth Stott
# Canary: aef8ad53-f062-4916-b519-d183ac518616
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for golden question testing (Phase 9)."""

from __future__ import annotations

import logging
from typing import Optional

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    CreateGoldenQuestionInput,
    DeleteResultType,
    GoldenQuestionExpectInput,
    GoldenQuestionExpectationType,
    GoldenQuestionType,
    MoveGoldenQuestionInput,
    TestableDomainType,
    UpdateGoldenQuestionInput,
)

logger = logging.getLogger(__name__)


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


def _load_domain_or_raise(config, domain_filename: str):
    domain = config.load_domain(domain_filename)
    if not domain:
        raise ValueError(f"Domain not found: {domain_filename}")
    return domain


def _gq_response_to_type(resp, warnings: list[str] | None = None) -> GoldenQuestionType:
    expect = resp.expect.model_dump() if hasattr(resp.expect, "model_dump") else resp.expect
    return GoldenQuestionType(
        question=resp.question,
        tags=resp.tags,
        expect=expect,
        objectives=resp.objectives or [],
        system_prompt=resp.system_prompt,
        index=resp.index,
        warnings=warnings,
    )


@strawberry.type
class Query:
    @strawberry.field
    async def testable_domains(
        self, info: Info, session_id: str
    ) -> list[TestableDomainType]:
        _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        config = managed.session.config

        from constat.testing.models import parse_golden_questions

        results: list[TestableDomainType] = []
        for filename, dc in config.domains.items():
            if not dc.golden_questions:
                continue
            questions = parse_golden_questions(dc.golden_questions)
            all_tags: set[str] = set()
            for q in questions:
                all_tags.update(q.tags)
            results.append(
                TestableDomainType(
                    filename=filename,
                    name=dc.name,
                    question_count=len(questions),
                    tags=sorted(all_tags),
                )
            )
        return results

    @strawberry.field
    async def golden_questions(
        self, info: Info, session_id: str, domain: str
    ) -> list[GoldenQuestionType]:
        _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        dc = _load_domain_or_raise(managed.session.config, domain)

        from constat.server.routes.testing import _gq_to_response

        return [
            _gq_response_to_type(_gq_to_response(i, q))
            for i, q in enumerate(dc.golden_questions)
        ]


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def extract_expectations(
        self,
        info: Info,
        session_id: str,
        input: GoldenQuestionExpectInput,
    ) -> GoldenQuestionExpectationType:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)

        import json as _json
        import re

        from constat.testing.grounding import _NON_GROUNDABLE_SOURCES, build_source_patterns
        from constat.server.routes.testing import (
            _resolve_entity_with_term,
            _generate_test_metadata,
        )

        # Resolve proof nodes — server-side first, then persisted, then client
        proof_nodes: list[dict] = []
        proof = managed.session.last_proof_result
        if proof and proof.get("proof_nodes"):
            proof_nodes = proof["proof_nodes"]
        else:
            state = managed.session.history.load_state(session_id)
            if state and "last_proof_result" in state:
                persisted_proof = state["last_proof_result"]
                if persisted_proof and persisted_proof.get("proof_nodes"):
                    proof_nodes = persisted_proof["proof_nodes"]
            if not proof_nodes and input.proof_nodes:
                proof_nodes = input.proof_nodes if isinstance(input.proof_nodes, list) else []

        if not proof_nodes:
            raise ValueError("No reasoning chain result available")

        doc_tools = managed.session.doc_tools
        relational = None
        if doc_tools and hasattr(doc_tools, "_vector_store"):
            relational = getattr(doc_tools._vector_store, "_relational", None)

        domain_ids = (
            list(getattr(doc_tools, "_active_domain_ids", None) or [])
            if doc_tools
            else []
        )

        grounding: list[dict] = []
        seen: set[str] = set()

        for node in proof_nodes:
            raw_source = node.get("source", "")
            source_type = raw_source.split(":")[0] if raw_source else ""
            if source_type in _NON_GROUNDABLE_SOURCES or not source_type:
                continue

            raw_name = node.get("name", "")
            if not raw_name:
                continue
            name = re.sub(r"^[A-Z]\d+:\s*", "", raw_name)
            if not name:
                continue

            resolved_name = None
            if relational:
                resolved_name, _ = _resolve_entity_with_term(
                    relational, name, session_id, user_id, domain_ids
                )
            if not resolved_name:
                resolved_name = name.strip().lower().replace(" ", "_")
            if resolved_name in seen:
                continue
            seen.add(resolved_name)

            resolves_to = build_source_patterns(node)
            if not resolves_to:
                resolves_to = [raw_source]

            grounding.append({"entity": resolved_name, "resolves_to": resolves_to, "strict": True})

        # Extract relationships
        relationships: list[dict] = []
        grounded_entities = [g["entity"] for g in grounding]
        if relational and grounded_entities:
            entity_set = set(e.lower() for e in grounded_entities)
            rel_seen: set[tuple[str, str, str]] = set()
            for entity_name in grounded_entities:
                try:
                    rels = relational.get_relationships_for_entity(entity_name, session_id)
                except Exception as e:
                    logger.warning(f"Failed to get relationships for {entity_name!r}: {e}")
                    continue
                for r in rels:
                    subj = r["subject_name"]
                    obj = r["object_name"]
                    if subj.lower() not in entity_set or obj.lower() not in entity_set:
                        continue
                    triple = (subj.lower(), r["verb"].lower(), obj.lower())
                    if triple in rel_seen:
                        continue
                    rel_seen.add(triple)
                    relationships.append({
                        "subject": subj,
                        "verb": r["verb"],
                        "object": obj,
                        "min_confidence": r.get("confidence", 0.0),
                    })

        # LLM-generated question and criteria
        suggested_question = None
        end_to_end = None
        proof_summary = input.proof_summary

        original_question = None
        has_clarifications = False
        question_source = "none"

        if managed.session.datastore:
            obj_json = managed.session.datastore.get_session_meta("objectives_log")
            if obj_json:
                try:
                    obj_log = _json.loads(obj_json)
                    q_entry = next(
                        (e for e in obj_log if e.get("type") == "question"), None
                    )
                    if q_entry and q_entry.get("text"):
                        original_question = q_entry["text"]
                        question_source = "objectives_log"
                        clarifs = [e for e in obj_log if e.get("type") == "clarification"]
                        if clarifs:
                            clarif_lines = [
                                f"{c['question']}: {c['answer']}" for c in clarifs
                            ]
                            original_question = (
                                f"{original_question}\n\nClarifications:\n"
                                + "\n".join(clarif_lines)
                            )
                            has_clarifications = True
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse objectives_log: {e}")

        if not original_question and managed.session.datastore:
            original_question = managed.session.datastore.get_session_meta("problem")
            if original_question:
                question_source = "datastore"

        if not original_question:
            proof_result = getattr(managed.session, "last_proof_result", None)
            if proof_result:
                original_question = proof_result.get("problem")
                if original_question:
                    question_source = "last_proof_result"

        if not original_question:
            client_q = input.original_question
            if client_q and not client_q.strip().startswith("/"):
                original_question = client_q
                question_source = "client"

        if (
            not has_clarifications
            and original_question
            and "\nClarifications:\n" not in original_question
        ):
            if managed.session.datastore:
                clarif_json = managed.session.datastore.get_session_meta("clarifications")
                if clarif_json:
                    try:
                        qa_pairs = _json.loads(clarif_json)
                        if qa_pairs:
                            clarif_lines = [
                                f"{qa['question']}: {qa['answer']}" for qa in qa_pairs
                            ]
                            original_question = (
                                f"{original_question}\n\nClarifications:\n"
                                + "\n".join(clarif_lines)
                            )
                            has_clarifications = True
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Failed to parse clarifications: {e}")
        elif original_question and "\nClarifications:\n" in original_question:
            has_clarifications = True

        if original_question or grounded_entities:
            try:
                from constat.testing.models import _DEFAULT_JUDGE_PROMPT

                config = managed.session.config
                suggested_question, criteria = _generate_test_metadata(
                    config, original_question, proof_summary, grounded_entities
                )
                if criteria:
                    end_to_end = {
                        "judge_prompt": f"{_DEFAULT_JUDGE_PROMPT}\n\nCriteria: {criteria}",
                    }
                else:
                    end_to_end = {"judge_prompt": _DEFAULT_JUDGE_PROMPT}
            except Exception as e:
                logger.warning(f"Failed to generate test metadata: {e}")

        # Build objectives
        objectives: list[str] = []
        if original_question:
            if "Follow-up requests:" in original_question:
                parts = original_question.split("Follow-up requests:")
                main_q = parts[0].removeprefix("Original request:").strip()
                if main_q:
                    objectives.append(main_q)
                if len(parts) > 1:
                    followup_text = parts[1].split("Prove all of the above")[0].strip()
                    for line in followup_text.split("\n"):
                        line = line.strip().lstrip("- ").strip()
                        if line:
                            objectives.append(line)
            else:
                objectives.append(
                    original_question.removeprefix("Original request:").strip()
                )

        # Step code hints
        step_hints: list[dict] = []
        session = managed.session
        if session.history and session.session_id:
            try:
                step_codes = session.history.list_step_codes(session.session_id)
                step_hints = [
                    {
                        "step_number": s.get("step_number"),
                        "goal": s.get("goal", ""),
                        "code": s.get("code", ""),
                    }
                    for s in step_codes
                    if s.get("code")
                ]
            except Exception as e:
                logger.debug(f"Could not load step codes: {e}")

        # Auto-extract expected outputs
        expected_outputs: list[dict] = []
        datastore = getattr(session, "datastore", None)
        if datastore:
            try:
                all_tables = datastore.list_tables()
                if all_tables:
                    step_numbers = [t.get("step_number", 0) or 0 for t in all_tables]
                    max_abs_step = max(abs(s) for s in step_numbers)
                    for t in all_tables:
                        step = t.get("step_number", 0) or 0
                        is_final = abs(step) == max_abs_step or t.get("is_published")
                        if not is_final:
                            continue
                        name = t["name"]
                        schema = datastore.get_table_schema(name)
                        columns = [c["name"] for c in schema] if schema else []
                        expected_outputs.append({
                            "name": name,
                            "type": "table",
                            "columns": columns,
                        })
            except Exception as e:
                logger.debug(f"Could not extract expected outputs: {e}")

        system_prompt = managed.session._get_system_prompt()

        return GoldenQuestionExpectationType(
            terms=[],
            grounding=grounding,
            relationships=relationships,
            expected_outputs=expected_outputs,
            end_to_end=end_to_end,
            suggested_question=suggested_question,
            objectives=objectives,
            step_hints=step_hints,
            system_prompt=system_prompt,
        )

    @strawberry.mutation
    async def create_golden_question(
        self,
        info: Info,
        session_id: str,
        domain: str,
        input: CreateGoldenQuestionInput,
    ) -> GoldenQuestionType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        dc = _load_domain_or_raise(managed.session.config, domain)

        from constat.server.routes.testing import (
            _can_modify_domain,
            _gq_to_response,
            _read_domain_yaml,
            _resolve_expectations,
            _write_domain_yaml,
        )

        if not _can_modify_domain(dc, user_id, server_config):
            raise ValueError("You do not have permission to modify this domain")

        expect = input.expect if isinstance(input.expect, dict) else dict(input.expect)
        has_grounding = bool(expect.get("grounding"))
        if not has_grounding:
            doc_tools = managed.session.doc_tools
            if doc_tools and hasattr(doc_tools, "_vector_store"):
                relational = getattr(doc_tools._vector_store, "_relational", None)
                if relational:
                    domain_ids = list(
                        getattr(doc_tools, "_active_domain_ids", None) or []
                    )
                    expect = _resolve_expectations(
                        relational, expect, session_id, user_id, domain_ids
                    )

        domain_path, data = _read_domain_yaml(dc)
        gq_list = data.setdefault("golden_questions", [])
        new_entry: dict = {"question": input.question, "tags": input.tags, "expect": expect}
        sp = expect.get("system_prompt") or input.system_prompt
        if sp:
            new_entry["system_prompt"] = sp
        if input.objectives:
            new_entry["objectives"] = input.objectives
        step_hints = expect.get("step_hints") if isinstance(expect, dict) else []
        if step_hints:
            new_entry["step_hints"] = step_hints
        gq_list.append(new_entry)
        _write_domain_yaml(domain_path, data, dc)

        return _gq_response_to_type(_gq_to_response(len(gq_list) - 1, new_entry))

    @strawberry.mutation
    async def update_golden_question(
        self,
        info: Info,
        session_id: str,
        domain: str,
        index: int,
        input: UpdateGoldenQuestionInput,
    ) -> GoldenQuestionType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        dc = _load_domain_or_raise(managed.session.config, domain)

        from constat.server.routes.testing import (
            _can_modify_domain,
            _gq_to_response,
            _read_domain_yaml,
            _write_domain_yaml,
        )

        if not _can_modify_domain(dc, user_id, server_config):
            raise ValueError("You do not have permission to modify this domain")

        domain_path, data = _read_domain_yaml(dc)
        gq_list = data.get("golden_questions", [])
        if index < 0 or index >= len(gq_list):
            raise ValueError(f"Golden question index {index} out of range")

        expect = input.expect if isinstance(input.expect, dict) else dict(input.expect)
        updated_entry: dict = {
            "question": input.question,
            "tags": input.tags,
            "expect": expect,
        }
        if input.objectives:
            updated_entry["objectives"] = input.objectives
        gq_list[index] = updated_entry
        data["golden_questions"] = gq_list
        _write_domain_yaml(domain_path, data, dc)

        return _gq_response_to_type(_gq_to_response(index, updated_entry))

    @strawberry.mutation
    async def delete_golden_question(
        self,
        info: Info,
        session_id: str,
        domain: str,
        index: int,
    ) -> DeleteResultType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        dc = _load_domain_or_raise(managed.session.config, domain)

        from constat.server.routes.testing import (
            _can_modify_domain,
            _read_domain_yaml,
            _write_domain_yaml,
        )

        if not _can_modify_domain(dc, user_id, server_config):
            raise ValueError("You do not have permission to modify this domain")

        domain_path, data = _read_domain_yaml(dc)
        gq_list = data.get("golden_questions", [])
        if index < 0 or index >= len(gq_list):
            raise ValueError(f"Golden question index {index} out of range")

        gq_list.pop(index)
        data["golden_questions"] = gq_list
        _write_domain_yaml(domain_path, data, dc)

        return DeleteResultType(status="deleted", name=f"{domain}[{index}]")

    @strawberry.mutation
    async def move_golden_question(
        self,
        info: Info,
        session_id: str,
        domain: str,
        index: int,
        input: MoveGoldenQuestionInput,
    ) -> GoldenQuestionType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)
        config = managed.session.config

        from constat.server.routes.testing import (
            _can_modify_domain,
            _gq_to_response,
            _read_domain_yaml,
            _write_domain_yaml,
        )
        from constat.core.resource_validation import (
            extract_resources_from_grounding,
            validate_resource_compatibility,
        )

        src_dc = _load_domain_or_raise(config, domain)
        if not _can_modify_domain(src_dc, user_id, server_config):
            raise ValueError("You do not have permission to modify the source domain")

        src_path, src_data = _read_domain_yaml(src_dc)
        src_list = src_data.get("golden_questions", [])
        if index < 0 or index >= len(src_list):
            raise ValueError(f"Golden question index {index} out of range")

        entry = src_list[index]

        tgt_dc = _load_domain_or_raise(config, input.target_domain)
        grounding = entry.get("grounding", [])
        required_resources = extract_resources_from_grounding(grounding)
        warnings: list[str] = []
        if required_resources:
            warnings = validate_resource_compatibility(
                required_resources, tgt_dc, input.target_domain
            )

        if input.validate_only:
            resp = _gq_to_response(index, entry)
            return _gq_response_to_type(resp, warnings=warnings)

        src_list.pop(index)

        if not _can_modify_domain(tgt_dc, user_id, server_config):
            raise ValueError("You do not have permission to modify the target domain")

        tgt_path, tgt_data = _read_domain_yaml(tgt_dc)
        tgt_list = tgt_data.setdefault("golden_questions", [])
        tgt_list.append(entry)

        _write_domain_yaml(src_path, src_data, src_dc)
        _write_domain_yaml(tgt_path, tgt_data, tgt_dc)

        resp = _gq_to_response(len(tgt_list) - 1, entry)
        return _gq_response_to_type(resp, warnings=warnings)
