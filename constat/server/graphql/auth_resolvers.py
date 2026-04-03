# Copyright (c) 2025 Kenneth Stott
# Canary: ad90349d-9230-4006-a3c0-7f11cd1622b0
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for auth, config, and permissions."""

from __future__ import annotations

import json
import logging

import strawberry
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    AuthPayload,
    EmailOAuthProvidersType,
    PasskeyOptions,
    ServerConfigType,
    UserPermissionsType,
)

logger = logging.getLogger(__name__)


@strawberry.type
class Query:
    @strawberry.field
    async def config(self, info: Info) -> ServerConfigType:
        cfg = info.context.config
        routing = cfg.llm.get_task_routing()
        task_routing: dict[str, list[dict[str, str]]] = {}
        for task_type, entry in routing.routes.items():
            task_routing[task_type] = [
                {"provider": spec.provider or cfg.llm.provider, "model": spec.model}
                for spec in entry.models
            ]
        return ServerConfigType(
            databases=list(cfg.databases.keys()),
            apis=list(cfg.apis.keys()),
            documents=list(cfg.documents.keys()),
            llm_provider=cfg.llm.provider,
            llm_model=cfg.llm.model,
            execution_timeout=cfg.execution.timeout_seconds,
            task_routing=task_routing,
        )

    @strawberry.field(name="emailOAuthProviders")
    async def email_oauth_providers(self, info: Info) -> EmailOAuthProvidersType:
        server_config = info.context.server_config
        return EmailOAuthProvidersType(
            google=server_config.google_email_client_id is not None,
            microsoft=server_config.microsoft_email_client_id is not None,
        )

    @strawberry.field
    async def my_permissions(self, info: Info) -> UserPermissionsType:
        from constat.server.permissions import get_user_permissions
        from constat.server.persona_config import PersonasConfig

        server_config = info.context.server_config
        user_id = info.context.user_id
        if not user_id:
            raise PermissionError("Authentication required")
        perms = get_user_permissions(server_config, user_id=user_id)

        personas_config: PersonasConfig | None = getattr(
            info.context.request.app.state, "personas_config", None
        )
        if personas_config:
            persona_def = personas_config.get_persona(perms.persona)
            visibility = persona_def.visibility
            writes = persona_def.writes
            feedback = persona_def.feedback
        else:
            visibility = {}
            writes = {}
            feedback = {}

        return UserPermissionsType(
            user_id=perms.user_id,
            email=perms.email,
            admin=perms.is_admin,
            persona=perms.persona,
            domains=perms.domains,
            databases=perms.databases,
            documents=perms.documents,
            apis=perms.apis,
            visibility=visibility,
            writes=writes,
            feedback=feedback,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def login(self, info: Info, email: str, password: str) -> AuthPayload:
        from constat.server.local_auth import create_local_token, verify_password

        server_config = info.context.server_config

        # Try local auth first
        for username, local_user in server_config.local_users.items():
            if username == email or local_user.email == email:
                if verify_password(password, local_user.password_hash):
                    token = create_local_token(username, local_user.email or email)
                    return AuthPayload(
                        token=token,
                        user_id=username,
                        email=local_user.email or email,
                    )

        # Try Firebase server-side auth
        if server_config.firebase_api_key:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={server_config.firebase_api_key}",
                    json={
                        "email": email,
                        "password": password,
                        "returnSecureToken": True,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return AuthPayload(
                        token=data["idToken"],
                        user_id=data["localId"],
                        email=data.get("email", email),
                    )

        raise ValueError("Invalid email or password")

    @strawberry.mutation
    async def logout(self, info: Info) -> bool:
        return True

    @strawberry.mutation
    async def register(
        self, info: Info, username: str, password: str, email: str = "",
    ) -> AuthPayload:
        """Register a new local user account."""
        from constat.server.local_auth import hash_password, create_local_token
        from constat.server.config import LocalUser

        server_config = info.context.server_config

        # Check username not taken
        if username in server_config.local_users:
            raise ValueError(f"Username already taken: {username}")
        for _, user in server_config.local_users.items():
            if email and user.email == email:
                raise ValueError(f"Email already registered: {email}")

        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")

        # Hash and persist
        pw_hash = hash_password(password)
        new_user = LocalUser(password_hash=pw_hash, email=email)

        # Persist to local_users.yaml in data_dir
        import yaml
        users_file = server_config.data_dir / "local_users.yaml"
        existing: dict = {}
        if users_file.exists():
            with open(users_file) as f:
                existing = yaml.safe_load(f) or {}
        existing[username] = {"password_hash": pw_hash, "email": email}
        users_file.parent.mkdir(parents=True, exist_ok=True)
        with open(users_file, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)

        # Update in-memory config
        server_config.local_users[username] = new_user

        # Auto-login
        token = create_local_token(username, email)
        return AuthPayload(token=token, user_id=username, email=email)

    @strawberry.mutation
    async def passkey_register_begin(self, info: Info, user_id: str) -> PasskeyOptions:
        from constat.server.routes.passkey import (
            _load_credentials,
            _pending_challenges,
            RP_ID,
            RP_NAME,
        )
        from webauthn import generate_registration_options
        from webauthn.helpers import options_to_json, base64url_to_bytes
        from webauthn.helpers.structs import (
            AuthenticatorSelectionCriteria,
            PublicKeyCredentialDescriptor,
            ResidentKeyRequirement,
            UserVerificationRequirement,
        )

        data_dir = info.context.server_config.data_dir
        existing = _load_credentials(data_dir, user_id)
        exclude = [
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["credential_id"]))
            for c in existing
        ]

        options = generate_registration_options(
            rp_id=RP_ID,
            rp_name=RP_NAME,
            user_id=user_id.encode(),
            user_name=user_id,
            user_display_name=user_id,
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
            exclude_credentials=exclude,
        )

        _pending_challenges[user_id] = options.challenge
        return PasskeyOptions(options_json=json.loads(options_to_json(options)))

    @strawberry.mutation
    async def passkey_register_complete(
        self, info: Info, user_id: str, credential: JSON
    ) -> bool:
        from constat.server.routes.passkey import (
            _load_credentials,
            _pending_challenges,
            _save_credentials,
            RP_ID,
            ORIGIN,
        )
        from webauthn import verify_registration_response
        from webauthn.helpers import bytes_to_base64url

        challenge = _pending_challenges.pop(user_id, None)
        if challenge is None:
            raise ValueError("No pending registration challenge")

        verification = verify_registration_response(
            credential=credential,
            expected_challenge=challenge,
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
        )

        data_dir = info.context.server_config.data_dir
        creds = _load_credentials(data_dir, user_id)
        creds.append({
            "credential_id": bytes_to_base64url(verification.credential_id),
            "public_key": bytes_to_base64url(verification.credential_public_key),
            "sign_count": verification.sign_count,
        })
        _save_credentials(data_dir, user_id, creds)
        return True

    @strawberry.mutation
    async def passkey_auth_begin(self, info: Info, user_id: str) -> PasskeyOptions:
        from constat.server.routes.passkey import (
            _load_credentials,
            _pending_challenges,
            RP_ID,
        )
        from webauthn import generate_authentication_options
        from webauthn.helpers import options_to_json, base64url_to_bytes
        from webauthn.helpers.structs import (
            PublicKeyCredentialDescriptor,
            UserVerificationRequirement,
        )

        data_dir = info.context.server_config.data_dir
        creds = _load_credentials(data_dir, user_id)
        if not creds:
            raise ValueError("No passkey registered for this user")

        allow = [
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["credential_id"]))
            for c in creds
        ]

        options = generate_authentication_options(
            rp_id=RP_ID,
            allow_credentials=allow,
            user_verification=UserVerificationRequirement.PREFERRED,
        )

        _pending_challenges[user_id] = options.challenge
        return PasskeyOptions(options_json=json.loads(options_to_json(options)))

    @strawberry.mutation
    async def passkey_auth_complete(
        self,
        info: Info,
        user_id: str,
        credential: JSON,
        prf_output: str | None = None,
    ) -> AuthPayload:
        from constat.server.local_auth import create_local_token
        from constat.server.routes.passkey import (
            _load_credentials,
            _pending_challenges,
            _save_credentials,
            RP_ID,
            ORIGIN,
        )
        from webauthn import verify_authentication_response
        from webauthn.helpers import base64url_to_bytes

        challenge = _pending_challenges.pop(user_id, None)
        if challenge is None:
            raise ValueError("No pending authentication challenge")

        data_dir = info.context.server_config.data_dir
        creds = _load_credentials(data_dir, user_id)

        cred_id_b64 = credential.get("id", "")
        matched = None
        for c in creds:
            if c["credential_id"] == cred_id_b64:
                matched = c
                break
        if matched is None:
            raise ValueError("Unknown credential")

        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=challenge,
            expected_rp_id=RP_ID,
            expected_origin=ORIGIN,
            credential_public_key=base64url_to_bytes(matched["public_key"]),
            credential_current_sign_count=matched["sign_count"],
        )

        matched["sign_count"] = verification.new_sign_count
        _save_credentials(data_dir, user_id, creds)

        # Unlock vault if PRF output provided and vault encryption enabled
        vault_unlocked = False
        server_config = info.context.server_config
        if prf_output and server_config.vault_encrypt:
            from constat.core.paths import user_vault_dir
            from constat.server.vault import UserVault

            prf_bytes = base64url_to_bytes(prf_output)
            user_dir = user_vault_dir(data_dir, user_id)
            vault = UserVault(user_dir, encrypt=True)
            vault.unlock(prf_bytes)
            vault_unlocked = True

        token = create_local_token(user_id, user_id)
        return AuthPayload(token=token, user_id=user_id, email=user_id, vault_unlocked=vault_unlocked)
