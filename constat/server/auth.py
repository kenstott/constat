# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Firebase authentication middleware and utilities."""

import logging
from typing import Annotated, TypeAlias

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Optional Firebase Admin SDK import
try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    FIREBASE_AVAILABLE = True
except ImportError:
    google_requests = None  # type: ignore[assignment]
    id_token = None  # type: ignore[assignment]
    FIREBASE_AVAILABLE = False
    logger.warning(
        "google-auth not installed. Firebase authentication will not work. "
        "Install with: pip install google-auth"
    )

# Security scheme for JWT Bearer tokens
security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> str:
    """Extract and validate user ID from Firebase JWT token.

    When auth is disabled (AUTH_DISABLED=true), returns "default".
    When auth is enabled, validates the Firebase JWT and returns the user's UID.

    Args:
        request: The FastAPI request object
        credentials: Optional Bearer token credentials

    Returns:
        The user ID (Firebase UID or "default")

    Raises:
        HTTPException: If auth is enabled and token is invalid/missing
    """
    server_config = request.app.state.server_config

    # Debug logging
    logger.info(f"[AUTH] auth_disabled={server_config.auth_disabled}, has_credentials={credentials is not None}")

    # If auth is disabled, return default user
    if server_config.auth_disabled:
        logger.info("[AUTH] Auth disabled, returning 'default'")
        return "default"

    # Auth is enabled - validate token
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not FIREBASE_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Firebase authentication not available. Install google-auth package.",
        )

    if not server_config.firebase_project_id:
        raise HTTPException(
            status_code=500,
            detail="Firebase project ID not configured",
        )

    try:
        # Verify the Firebase ID token
        # This validates the token signature, expiration, and audience
        # noinspection PyTypeChecker
        decoded_token = id_token.verify_firebase_token(
            credentials.credentials,
            google_requests.Request(),
            audience=server_config.firebase_project_id,
        )

        # Extract user ID from the token
        user_id = decoded_token.get("sub") or decoded_token.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing user ID",
            )

        logger.info(f"[AUTH] Authenticated user: {user_id}")

        # Store email in request state for other dependencies
        email = decoded_token.get("email")
        request.state.user_email = email

        return user_id

    except ValueError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_email(
    request: Request,
    _user_id: Annotated[str, Depends(get_current_user_id)],  # Ensures auth runs first
) -> str | None:
    """Get the current user's email from the validated token.

    This depends on get_current_user_id running first to validate the token
    and store the email in request state.

    Returns:
        The user's email address, or None if not available
    """
    return getattr(request.state, "user_email", None)


# Type aliases for dependency injection
CurrentUserId: TypeAlias = Annotated[str, Depends(get_current_user_id)]
CurrentUserEmail: TypeAlias = Annotated[str | None, Depends(get_current_user_email)]