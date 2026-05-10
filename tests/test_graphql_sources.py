# Copyright (c) 2025 Kenneth Stott
# Canary: 620e1215-1bc1-418f-849e-76ff728fad2c
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL source resolvers — schema stitching / SDL presence tests.

Resolver tests are split by concern:
  - test_graphql_sources_queries.py    — query resolvers
  - test_graphql_sources_mutations.py  — mutation resolvers
"""

from __future__ import annotations


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestSourceSchemaStitching:
    """Verify all source operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    # Queries
    def test_files_query(self):
        assert "files(" in self._get_sdl()

    def test_file_refs_query(self):
        assert "fileRefs(" in self._get_sdl()

    def test_databases_query(self):
        assert "databases(" in self._get_sdl()

    def test_data_sources_query(self):
        assert "dataSources(" in self._get_sdl()

    def test_database_table_preview_query(self):
        assert "databaseTablePreview(" in self._get_sdl()

    def test_document_query(self):
        assert "document(" in self._get_sdl()

    def test_user_sources_query(self):
        assert "userSources" in self._get_sdl()

    # Mutations
    def test_upload_file_mutation(self):
        assert "uploadFile(" in self._get_sdl()

    def test_upload_file_data_uri_mutation(self):
        assert "uploadFileDataUri(" in self._get_sdl()

    def test_delete_file_mutation(self):
        assert "deleteFile(" in self._get_sdl()

    def test_add_file_ref_mutation(self):
        assert "addFileRef(" in self._get_sdl()

    def test_delete_file_ref_mutation(self):
        assert "deleteFileRef(" in self._get_sdl()

    def test_add_database_mutation(self):
        assert "addDatabase(" in self._get_sdl()

    def test_remove_database_mutation(self):
        assert "removeDatabase(" in self._get_sdl()

    def test_test_database_mutation(self):
        assert "testDatabase(" in self._get_sdl()

    def test_add_api_mutation(self):
        assert "addApi(" in self._get_sdl()

    def test_remove_api_mutation(self):
        assert "removeApi(" in self._get_sdl()

    def test_add_document_uri_mutation(self):
        assert "addDocumentUri(" in self._get_sdl()

    def test_add_email_source_mutation(self):
        assert "addEmailSource(" in self._get_sdl()

    def test_upload_documents_mutation(self):
        assert "uploadDocuments(" in self._get_sdl()

    def test_refresh_documents_mutation(self):
        assert "refreshDocuments(" in self._get_sdl()

    def test_remove_user_source_mutation(self):
        assert "removeUserSource(" in self._get_sdl()

    def test_move_source_mutation(self):
        assert "moveSource(" in self._get_sdl()
