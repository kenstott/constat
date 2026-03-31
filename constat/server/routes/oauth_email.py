# Copyright (c) 2025 Kenneth Stott
# Canary: 2fab1cbb-56a8-42cc-81b8-eb2be9b8ac7f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""OAuth2 browser-based email (IMAP) authentication routes."""

import logging
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_EXPIRY_SECONDS = 600  # 10 minutes


def _cleanup_expired(app_state) -> None:
    """Remove entries older than 10 minutes from pending and completed dicts."""
    now = time.time()
    if hasattr(app_state, "oauth_pending"):
        expired = [k for k, v in app_state.oauth_pending.items() if now - v["created_at"] > _EXPIRY_SECONDS]
        for k in expired:
            del app_state.oauth_pending[k]
    if hasattr(app_state, "oauth_completed"):
        expired = [k for k, v in app_state.oauth_completed.items() if now - v["created_at"] > _EXPIRY_SECONDS]
        for k in expired:
            del app_state.oauth_completed[k]


def _ensure_state_dicts(app_state) -> None:
    """Initialize oauth state dicts if not present."""
    if not hasattr(app_state, "oauth_pending"):
        app_state.oauth_pending = {}
    if not hasattr(app_state, "oauth_completed"):
        app_state.oauth_completed = {}


@router.get("/providers")
async def get_providers(request: Request) -> dict:
    """Return which OAuth2 email providers are configured."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    server_config = request.app.state.server_config
    return {
        "google": server_config.google_email_client_id is not None,
        "microsoft": server_config.microsoft_email_client_id is not None,
    }


@router.get("/authorize")
async def authorize(request: Request, provider: str, session_id: str) -> RedirectResponse:
    """Redirect user to OAuth2 provider authorization page."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    server_config = request.app.state.server_config

    if provider not in ("google", "microsoft"):
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    state = secrets.token_urlsafe(32)
    request.app.state.oauth_pending[state] = {
        "provider": provider,
        "session_id": session_id,
        "created_at": time.time(),
    }

    redirect_uri = f"{request.base_url}api/oauth/email/callback"

    if provider == "google":
        if not server_config.google_email_client_id:
            raise HTTPException(status_code=400, detail="Google email OAuth2 not configured")
        params = {
            "client_id": server_config.google_email_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "https://mail.google.com/ email",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    else:
        if not server_config.microsoft_email_client_id:
            raise HTTPException(status_code=400, detail="Microsoft email OAuth2 not configured")
        tenant = server_config.microsoft_email_tenant_id
        params = {
            "client_id": server_config.microsoft_email_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access",
            "state": state,
        }
        auth_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{urlencode(params)}"

    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = "") -> HTMLResponse:
    """Handle OAuth2 callback from provider."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    if error:
        return HTMLResponse(
            content=f"<html><body><h2>OAuth Error</h2><p>{error}</p></body></html>",
            status_code=400,
        )

    if state not in request.app.state.oauth_pending:
        raise HTTPException(status_code=400, detail="Invalid or expired state token")

    pending = request.app.state.oauth_pending.pop(state)
    provider = pending["provider"]
    server_config = request.app.state.server_config
    redirect_uri = f"{request.base_url}api/oauth/email/callback"

    async with httpx.AsyncClient() as client:
        if provider == "google":
            # Exchange code for tokens
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": server_config.google_email_client_id,
                    "client_secret": server_config.google_email_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            access_token = token_data["access_token"]
            refresh_token = token_data.get("refresh_token", "")

            # Get email via userinfo
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_resp.raise_for_status()
            email_address = userinfo_resp.json()["email"]

        else:
            # Microsoft
            tenant = server_config.microsoft_email_tenant_id
            token_resp = await client.post(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                data={
                    "code": code,
                    "client_id": server_config.microsoft_email_client_id,
                    "client_secret": server_config.microsoft_email_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                    "scope": "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access",
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            refresh_token = token_data.get("refresh_token", "")

            # Extract email from id_token JWT (base64 decode middle segment)
            import base64
            import json

            id_token = token_data.get("id_token", "")
            if id_token:
                payload_segment = id_token.split(".")[1]
                # Add padding
                payload_segment += "=" * (4 - len(payload_segment) % 4)
                payload_bytes = base64.urlsafe_b64decode(payload_segment)
                claims = json.loads(payload_bytes)
                email_address = claims.get("preferred_username", claims.get("email", ""))
            else:
                email_address = ""

    # Store completed result
    request.app.state.oauth_completed[state] = {
        "provider": provider,
        "email": email_address,
        "refresh_token": refresh_token,
        "created_at": time.time(),
    }

    return HTMLResponse(content=f"""<html><body><script>
window.opener.postMessage({{
  type: 'oauth-email-complete',
  provider: '{provider}',
  email: '{email_address}',
  refresh_token: '{refresh_token}',
  state: '{state}'
}}, '*');
window.close();
</script></body></html>""")


@router.get("/result/{state}")
async def get_result(request: Request, state: str) -> dict:
    """Fallback endpoint to retrieve OAuth result (for popup-blocked scenarios)."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    if state not in request.app.state.oauth_completed:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    return request.app.state.oauth_completed[state]
