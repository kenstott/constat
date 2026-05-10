from __future__ import annotations
# Copyright (c) 2025 Kenneth Stott
# Canary: d3ad1cb0-1f56-4d7b-96d6-ecf7c8b8b014
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Error handling tests for APIExecutor.

Tests HTTP error responses (4xx, 5xx) and related APIExecutionError behaviour.
"""

import json
from unittest.mock import Mock, patch

import pytest
import httpx

from constat.core.config import Config, APIConfig
from constat.catalog.api_executor import APIExecutor, APIExecutionError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_response():
    """Factory for creating mock HTTP responses."""
    def _create(
        status_code: int = 200,
        json_data: dict = None,
        text: str = None,
        content_type: str = "application/json",
    ):
        response = Mock(spec=httpx.Response)
        response.status_code = status_code
        response.headers = {"content-type": content_type}
        if json_data is not None:
            response.json.return_value = json_data
            response.text = json.dumps(json_data)
        elif text is not None:
            response.text = text
            response.json.side_effect = json.JSONDecodeError("", "", 0)
        else:
            response.json.return_value = {}
            response.text = "{}"
        return response
    return _create


# =============================================================================
# Authentication Error Handling Tests
# =============================================================================


class TestAuthErrorHandling:
    """Tests for authentication error handling."""

    def test_401_unauthorized_raises_error(self, mock_response):
        """401 response should raise APIExecutionError."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="graphql",
                    url="https://api.example.com/graphql",
                    auth_type="bearer",
                    auth_token="invalid_token",
                ),
            }
        )

        unauthorized_response = mock_response(
            status_code=401,
            json_data={"error": "Unauthorized"},
        )

        with patch.object(httpx.Client, 'post', return_value=unauthorized_response):
            with APIExecutor(config) as executor:
                with pytest.raises(APIExecutionError) as exc_info:
                    executor.execute_graphql("test_api", "{ users { id } }")

                assert exc_info.value.status_code == 401
                assert "401" in str(exc_info.value)

    def test_403_forbidden_raises_error(self, mock_response):
        """403 response should raise APIExecutionError."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="openapi",
                    url="https://api.example.com",
                    auth_type="api_key",
                    api_key="insufficient_permissions_key",
                ),
            }
        )

        forbidden_response = mock_response(
            status_code=403,
            json_data={"error": "Forbidden"},
        )

        with patch.object(httpx.Client, 'get', return_value=forbidden_response):
            with APIExecutor(config) as executor:
                with pytest.raises(APIExecutionError) as exc_info:
                    executor.execute_rest("test_api", "/admin/users")

                assert exc_info.value.status_code == 403

    def test_rest_auth_failure_includes_response_body(self, mock_response):
        """Auth failure error should include response body for debugging."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="openapi",
                    url="https://api.example.com",
                    auth_type="bearer",
                    auth_token="expired_token",
                ),
            }
        )

        error_response = mock_response(
            status_code=401,
            json_data={"error": "token_expired", "message": "Token has expired"},
        )

        with patch.object(httpx.Client, 'get', return_value=error_response):
            with APIExecutor(config) as executor:
                with pytest.raises(APIExecutionError) as exc_info:
                    executor.execute_rest("test_api", "/users")

                assert exc_info.value.response_body is not None
                assert "token_expired" in exc_info.value.response_body


# =============================================================================
# Parametrized Error Status Code Tests
# =============================================================================


class TestErrorStatusCodes:
    """Parametrized tests covering all error status codes."""

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 502, 503])
    def test_error_status_codes_raise_exception(self, status_code, mock_response):
        """All error status codes should raise APIExecutionError."""
        config = Config(
            apis={
                "test_api": APIConfig(
                    type="openapi",
                    url="https://api.example.com",
                    auth_type="bearer",
                    auth_token="test_token",
                ),
            }
        )

        error_response = mock_response(
            status_code=status_code,
            json_data={"error": f"Error {status_code}"},
        )

        with patch.object(httpx.Client, 'get', return_value=error_response):
            with APIExecutor(config) as executor:
                with pytest.raises(APIExecutionError) as exc_info:
                    executor.execute_rest("test_api", "/users")

                assert exc_info.value.status_code == status_code
