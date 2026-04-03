# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Generalized OAuth2 routes for multi-resource authentication (email, drive, calendar, SharePoint)."""

import base64
import json
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

SCOPE_MAP: dict[tuple[str, str], str] = {
    ("google", "email"): "https://mail.google.com/ email",
    ("google", "drive"): "https://www.googleapis.com/auth/drive.readonly email",
    ("google", "calendar"): "https://www.googleapis.com/auth/calendar.readonly email",
    ("microsoft", "email"): "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access",
    ("microsoft", "drive"): "Files.Read.All offline_access",
    ("microsoft", "calendar"): "Calendars.Read offline_access",
    ("microsoft", "sharepoint"): "Sites.Read.All Files.Read.All offline_access",
}

VALID_PROVIDERS = {"google", "microsoft"}
VALID_RESOURCE_TYPES = {"email", "drive", "calendar", "sharepoint"}


def _cleanup_expired(app_state) -> None:
    """Remove entries older than 10 minutes from pending and completed dicts."""
    now = time.time()
    if hasattr(app_state, "oauth_gen_pending"):
        expired = [k for k, v in app_state.oauth_gen_pending.items() if now - v["created_at"] > _EXPIRY_SECONDS]
        for k in expired:
            del app_state.oauth_gen_pending[k]
    if hasattr(app_state, "oauth_gen_completed"):
        expired = [k for k, v in app_state.oauth_gen_completed.items() if now - v["created_at"] > _EXPIRY_SECONDS]
        for k in expired:
            del app_state.oauth_gen_completed[k]


def _ensure_state_dicts(app_state) -> None:
    """Initialize generalized oauth state dicts if not present."""
    if not hasattr(app_state, "oauth_gen_pending"):
        app_state.oauth_gen_pending = {}
    if not hasattr(app_state, "oauth_gen_completed"):
        app_state.oauth_gen_completed = {}


def _get_client_id(server_config, provider: str) -> str | None:
    """Get OAuth client ID for a provider (prefers generalized, falls back to email-specific)."""
    if provider == "google":
        return server_config.google_oauth_client_id or server_config.google_email_client_id
    elif provider == "microsoft":
        return server_config.microsoft_oauth_client_id or server_config.microsoft_email_client_id
    return None


def _get_client_secret(server_config, provider: str) -> str | None:
    """Get OAuth client secret for a provider."""
    if provider == "google":
        return server_config.google_oauth_client_secret or server_config.google_email_client_secret
    elif provider == "microsoft":
        return server_config.microsoft_oauth_client_secret or server_config.microsoft_email_client_secret
    return None


def _get_tenant_id(server_config, provider: str) -> str:
    """Get tenant ID for Microsoft (ignored for Google)."""
    if provider == "microsoft":
        return (
            getattr(server_config, "microsoft_oauth_tenant_id", None)
            or server_config.microsoft_email_tenant_id
        )
    return "common"


@router.get("/providers")
async def providers(request: Request) -> dict:
    """List available OAuth providers and resource types."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    server_config = request.app.state.server_config
    result: dict[str, list[str]] = {"google": [], "microsoft": []}

    if server_config.google_oauth_client_id or server_config.google_email_client_id:
        result["google"] = ["email", "drive", "calendar"]
    if server_config.microsoft_oauth_client_id or server_config.microsoft_email_client_id:
        result["microsoft"] = ["email", "drive", "calendar", "sharepoint"]

    return result


@router.get("/authorize")
async def authorize(
    request: Request,
    provider: str,
    resource_type: str,
    redirect_uri: str,
) -> RedirectResponse:
    """Redirect to OAuth provider with type-specific scopes."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    if provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    if resource_type not in VALID_RESOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported resource type: {resource_type}")

    scope_key = (provider, resource_type)
    if scope_key not in SCOPE_MAP:
        raise HTTPException(status_code=400, detail=f"No scopes for {provider}/{resource_type}")

    server_config = request.app.state.server_config
    client_id = _get_client_id(server_config, provider)
    if not client_id:
        raise HTTPException(status_code=400, detail=f"OAuth not configured for {provider}")

    scopes = SCOPE_MAP[scope_key]
    state = secrets.token_urlsafe(32)

    callback_uri = f"{request.base_url}api/oauth/callback"

    request.app.state.oauth_gen_pending[state] = {
        "provider": provider,
        "resource_type": resource_type,
        "redirect_uri": redirect_uri,
        "created_at": time.time(),
    }

    if provider == "google":
        params = {
            "client_id": client_id,
            "redirect_uri": callback_uri,
            "response_type": "code",
            "scope": scopes,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    else:
        tenant = _get_tenant_id(server_config, provider)
        params = {
            "client_id": client_id,
            "redirect_uri": callback_uri,
            "response_type": "code",
            "scope": scopes,
            "state": state,
        }
        auth_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{urlencode(params)}"

    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
) -> HTMLResponse:
    """Exchange auth code for tokens, return refresh_token + email via postMessage."""
    _ensure_state_dicts(request.app.state)
    _cleanup_expired(request.app.state)

    if error:
        return HTMLResponse(
            content=f"<html><body><h2>OAuth Error</h2><p>{error}</p></body></html>",
            status_code=400,
        )

    if state not in request.app.state.oauth_gen_pending:
        raise HTTPException(status_code=400, detail="Invalid or expired state token")

    pending = request.app.state.oauth_gen_pending.pop(state)
    provider = pending["provider"]
    resource_type = pending["resource_type"]
    server_config = request.app.state.server_config
    callback_uri = f"{request.base_url}api/oauth/callback"

    client_id = _get_client_id(server_config, provider)
    client_secret = _get_client_secret(server_config, provider)

    async with httpx.AsyncClient() as client:
        if provider == "google":
            token_data = await _exchange_google(client, code, client_id, client_secret, callback_uri)
            access_token = token_data["access_token"]
            refresh_token = token_data.get("refresh_token", "")
            email_address = await _get_google_email(client, access_token)
        else:
            tenant = _get_tenant_id(server_config, provider)
            scopes = SCOPE_MAP[(provider, resource_type)]
            token_data = await _exchange_microsoft(client, code, client_id, client_secret, callback_uri, tenant, scopes)
            refresh_token = token_data.get("refresh_token", "")
            email_address = _extract_microsoft_email(token_data)

    # Store completed result
    request.app.state.oauth_gen_completed[state] = {
        "provider": provider,
        "resource_type": resource_type,
        "email": email_address,
        "refresh_token": refresh_token,
        "created_at": time.time(),
    }

    return HTMLResponse(content=f"""<html><body><script>
window.opener.postMessage({{
  type: 'oauth-complete',
  provider: '{provider}',
  resourceType: '{resource_type}',
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

    if state not in request.app.state.oauth_gen_completed:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    return request.app.state.oauth_gen_completed[state]


async def _exchange_google(
    client: httpx.AsyncClient,
    code: str,
    client_id: str | None,
    client_secret: str | None,
    redirect_uri: str,
) -> dict:
    """Exchange Google auth code for tokens."""
    resp = await client.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()


async def _get_google_email(client: httpx.AsyncClient, access_token: str) -> str:
    """Get email address from Google userinfo endpoint."""
    resp = await client.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    resp.raise_for_status()
    return resp.json()["email"]


async def _exchange_microsoft(
    client: httpx.AsyncClient,
    code: str,
    client_id: str | None,
    client_secret: str | None,
    redirect_uri: str,
    tenant: str,
    scopes: str,
) -> dict:
    """Exchange Microsoft auth code for tokens."""
    resp = await client.post(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "scope": scopes,
        },
    )
    resp.raise_for_status()
    return resp.json()


def _extract_microsoft_email(token_data: dict) -> str:
    """Extract email from Microsoft id_token JWT."""
    id_token = token_data.get("id_token", "")
    if not id_token:
        return ""
    payload_segment = id_token.split(".")[1]
    # Add padding
    payload_segment += "=" * (4 - len(payload_segment) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_segment)
    claims = json.loads(payload_bytes)
    return claims.get("preferred_username", claims.get("email", ""))
