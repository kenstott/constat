# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for glossary GraphQL mutation types and helper functions."""

import pytest

from constat.server.graphql.types import (
    DraftAliasesType,
    DraftDefinitionType,
    DraftTagsType,
    GenerateResultType,
    RefineResultType,
    RenameResultType,
    TaxonomySuggestionType,
    TaxonomySuggestionsType,
)


class TestGlossaryMutationTypes:
    """Verify new Strawberry types can be instantiated correctly."""

    def test_generate_result_type(self):
        t = GenerateResultType(status="generating", message="Glossary generation started")
        assert t.status == "generating"
        assert t.message == "Glossary generation started"

    def test_draft_definition_type(self):
        t = DraftDefinitionType(name="revenue", draft="Total income from sales")
        assert t.name == "revenue"
        assert t.draft == "Total income from sales"

    def test_draft_aliases_type(self):
        t = DraftAliasesType(name="revenue", aliases=["income", "sales revenue"])
        assert t.name == "revenue"
        assert t.aliases == ["income", "sales revenue"]

    def test_draft_aliases_empty(self):
        t = DraftAliasesType(name="x", aliases=[])
        assert t.aliases == []

    def test_draft_tags_type(self):
        t = DraftTagsType(name="ssn", tags=["PII", "SENSITIVE"])
        assert t.name == "ssn"
        assert t.tags == ["PII", "SENSITIVE"]

    def test_refine_result_type(self):
        t = RefineResultType(name="revenue", before="old def", after="new def")
        assert t.name == "revenue"
        assert t.before == "old def"
        assert t.after == "new def"

    def test_refine_result_type_no_before(self):
        t = RefineResultType(name="revenue", before=None, after="new def")
        assert t.before is None

    def test_taxonomy_suggestion_type(self):
        t = TaxonomySuggestionType(
            child="savings_account",
            parent="account",
            parent_verb="HAS_KIND",
            confidence="high",
            reason="Savings account is a type of account",
        )
        assert t.child == "savings_account"
        assert t.parent_verb == "HAS_KIND"

    def test_taxonomy_suggestions_type(self):
        s = TaxonomySuggestionType(
            child="a", parent="b", parent_verb="HAS_ONE",
            confidence="medium", reason="test",
        )
        t = TaxonomySuggestionsType(suggestions=[s], message=None)
        assert len(t.suggestions) == 1
        assert t.message is None

    def test_taxonomy_suggestions_with_message(self):
        t = TaxonomySuggestionsType(
            suggestions=[], message="Need at least 2 defined terms",
        )
        assert t.suggestions == []
        assert t.message == "Need at least 2 defined terms"

    def test_rename_result_type(self):
        t = RenameResultType(
            old_name="old", new_name="new",
            display_name="New", relationships_updated=3,
        )
        assert t.old_name == "old"
        assert t.new_name == "new"
        assert t.display_name == "New"
        assert t.relationships_updated == 3


class TestGlossaryHelperImports:
    """Verify helper functions can be imported from the refactored glossary module."""

    def test_import_generate_glossary_op(self):
        from constat.server.routes.data.glossary import generate_glossary_op
        assert callable(generate_glossary_op)

    def test_import_rename_term_op(self):
        from constat.server.routes.data.glossary import rename_term_op
        assert callable(rename_term_op)

    def test_import_draft_definition_op(self):
        from constat.server.routes.data.glossary import draft_definition_op
        assert callable(draft_definition_op)

    def test_import_draft_aliases_op(self):
        from constat.server.routes.data.glossary import draft_aliases_op
        assert callable(draft_aliases_op)

    def test_import_draft_tags_op(self):
        from constat.server.routes.data.glossary import draft_tags_op
        assert callable(draft_tags_op)

    def test_import_refine_definition_op(self):
        from constat.server.routes.data.glossary import refine_definition_op
        assert callable(refine_definition_op)

    def test_import_suggest_taxonomy_op(self):
        from constat.server.routes.data.glossary import suggest_taxonomy_op
        assert callable(suggest_taxonomy_op)

    def test_no_router_in_glossary(self):
        """Verify glossary module no longer exports a FastAPI router."""
        import constat.server.routes.data.glossary as mod
        assert not hasattr(mod, "router")

    def test_no_fastapi_imports(self):
        """Verify no FastAPI dependencies remain in the module."""
        import constat.server.routes.data.glossary as mod
        source = open(mod.__file__).read()
        assert "from fastapi" not in source
        assert "APIRouter" not in source
        assert "Depends" not in source
        assert "HTTPException" not in source


class TestGlossaryHelperValidation:
    """Test helper function input validation."""

    @pytest.mark.asyncio
    async def test_rename_empty_name_raises(self):
        from constat.server.routes.data.glossary import rename_term_op
        with pytest.raises(ValueError, match="new_name is required"):
            await rename_term_op("sid", None, None, "old", "")

    @pytest.mark.asyncio
    async def test_generate_glossary_no_vs_raises(self):
        from constat.server.routes.data.glossary import generate_glossary_op
        from unittest.mock import MagicMock
        managed = MagicMock()
        managed.session.doc_tools = None
        with pytest.raises(ValueError, match="Vector store not available"):
            await generate_glossary_op("sid", managed, None)

    @pytest.mark.asyncio
    async def test_draft_definition_no_router_raises(self):
        from constat.server.routes.data.glossary import draft_definition_op
        from unittest.mock import MagicMock
        managed = MagicMock()
        managed.session.router = None
        with pytest.raises(ValueError, match="LLM router not available"):
            await draft_definition_op("sid", managed, MagicMock(), "test")

    @pytest.mark.asyncio
    async def test_refine_term_not_found_raises(self):
        from constat.server.routes.data.glossary import refine_definition_op
        from unittest.mock import MagicMock
        managed = MagicMock()
        managed.user_id = "default"
        vs = MagicMock()
        vs.get_glossary_term.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await refine_definition_op("sid", managed, vs, "nonexistent")


class TestSchemaIntrospection:
    """Test that the GraphQL schema includes new mutations via direct import."""

    def test_schema_has_new_mutations(self):
        from constat.server.graphql import schema
        introspection = schema.introspect()
        mutation_type = None
        for t in introspection["__schema"]["types"]:
            if t["name"] == "Mutation":
                mutation_type = t
                break
        assert mutation_type is not None
        field_names = {f["name"] for f in mutation_type["fields"]}
        assert "generateGlossary" in field_names
        assert "draftDefinition" in field_names
        assert "draftAliases" in field_names
        assert "draftTags" in field_names
        assert "refineDefinition" in field_names
        assert "suggestTaxonomy" in field_names
        assert "renameTerm" in field_names

    def test_schema_has_new_types(self):
        from constat.server.graphql import schema
        introspection = schema.introspect()
        type_names = {t["name"] for t in introspection["__schema"]["types"]}
        assert "GenerateResultType" in type_names
        assert "DraftDefinitionType" in type_names
        assert "DraftAliasesType" in type_names
        assert "DraftTagsType" in type_names
        assert "RefineResultType" in type_names
        assert "TaxonomySuggestionsType" in type_names
        assert "TaxonomySuggestionType" in type_names
        assert "RenameResultType" in type_names

    def test_generate_glossary_mutation_args(self):
        """Verify generateGlossary mutation has correct argument types."""
        from constat.server.graphql import schema
        introspection = schema.introspect()
        mutation_type = next(
            t for t in introspection["__schema"]["types"] if t["name"] == "Mutation"
        )
        generate = next(
            f for f in mutation_type["fields"] if f["name"] == "generateGlossary"
        )
        arg_names = {a["name"] for a in generate["args"]}
        assert "sessionId" in arg_names
        assert "phases" in arg_names

    def test_rename_term_mutation_args(self):
        """Verify renameTerm mutation has correct argument types."""
        from constat.server.graphql import schema
        introspection = schema.introspect()
        mutation_type = next(
            t for t in introspection["__schema"]["types"] if t["name"] == "Mutation"
        )
        rename = next(
            f for f in mutation_type["fields"] if f["name"] == "renameTerm"
        )
        arg_names = {a["name"] for a in rename["args"]}
        assert "sessionId" in arg_names
        assert "name" in arg_names
        assert "newName" in arg_names
