# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Authentication guards and user retrieval."""


import asyncio
import logging
import re
from typing import Any

from fastapi import Depends, HTTPException, status, Request, Header
from firebase_admin import auth


# --- Google Auth for Identity Platform ---
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token

from src.config.config_service import config_service
from src.users.user_model import UserModel, UserRoleEnum
from src.users.user_service import UserService

# Initialize the service once to be used by dependencies.
# user_service = UserService()  <-- REMOVED

import fastapi.security.utils

logger = logging.getLogger(__name__)

_WORKFORCE_SUB_PREFIX = (
    "principal://iam.googleapis.com/locations/global/workforcePools/"
)
_GUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _extract_workforce_oid(decoded_token: dict[str, Any]) -> str | None:
    """Extracts and validates the Entra `oid` from a Workforce IAP `sub` claim.

    Returns the lowercased GUID for workforce principals, or None for
    non-workforce tokens when WORKFORCE_POOL_ID is not configured.
    Raises 401 if a workforce token has a mismatched pool or a non-GUID subject.
    """
    sub = decoded_token.get("sub")
    identity_source = decoded_token.get("identity_source")
    configured_pool = (
        getattr(config_service, "WORKFORCE_POOL_ID", "") or ""
    ).strip()
    is_workforce = (
        bool(configured_pool)
        or (isinstance(sub, str) and sub.startswith("principal://"))
        or identity_source
        in ("WORKFORCE_POOL", "WORKFORCE_IDENTITY_FEDERATION")
    )
    if not is_workforce:
        return None

    if not isinstance(sub, str) or not sub.startswith(_WORKFORCE_SUB_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid IAP authentication token: missing or malformed "
                "workforce principal subject."
            ),
        )
    rest = sub[len(_WORKFORCE_SUB_PREFIX) :]
    if "/subject/" not in rest:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid IAP authentication token: missing subject segment "
                "in workforce principal."
            ),
        )
    pool_id, oid_part = rest.split("/subject/", 1)
    if not pool_id or "/" in pool_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid IAP authentication token: invalid workforce pool "
                "identifier."
            ),
        )
    if configured_pool and pool_id != configured_pool:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid IAP authentication token: workforce pool does not "
                "match configured pool."
            ),
        )
    if not _GUID_RE.match(oid_part):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid IAP authentication token: workforce subject is not "
                "a valid Entra OID."
            ),
        )
    return oid_part.lower()


async def get_iap_jwt(
    request: Request, x_goog_iap_jwt_assertion: str | None = Header(None)
) -> str | None:
    """Extracts the IAP JWT assertion. In local environment, this is optional."""
    if config_service.ENVIRONMENT == "local":
        return x_goog_iap_jwt_assertion or "mock_local_token"
    if not x_goog_iap_jwt_assertion:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing X-Goog-Iap-Jwt-Assertion header.",
        )
    return x_goog_iap_jwt_assertion


async def get_current_user(
    request: Request,
    token: str = Depends(get_iap_jwt),
    user_service: UserService = Depends(UserService),
) -> UserModel:
    """Dependency that handles the entire authentication and user
    provisioning flow via Identity-Aware Proxy (IAP).

    1. Checks if running locally to bypass verification.
    2. Verifies the Google-signed IAP JWT token.
    3. Extracts user information (entra_oid, email, name, picture).
    4. If the user is new, creates their profile JIT.
    5. Returns a Pydantic model with the user's data.
    """
    try:
        if config_service.ENVIRONMENT == "local":
            # Local Dev Bypass: Use mock user
            mock_email = request.headers.get(
                "X-Mock-User-Email", "local-dev@example.com"
            )
            mock_name = request.headers.get(
                "X-Mock-User-Name", "Local Dev User"
            )
            user_doc = await user_service.create_user_if_not_exists(
                email=mock_email,
                name=mock_name,
                picture="",
            )
            return user_doc

        # Verify Google-signed IAP JWT assertion
        try:
            decoded_token = await asyncio.to_thread(
                id_token.verify_token,
                token,
                google_auth_requests.Request(),
                audience=config_service.IAP_EXPECTED_AUDIENCE,
                certs_url="https://www.gstatic.com/iap/verify/public_key",
            )
        except ValueError as exc:
            logger.error("[get_current_user - Invalid IAP Token]: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid IAP authentication token: {exc}",
            ) from exc

        logger.info("Decoded IAP Token Claims: %s", list(decoded_token.keys()))
        logger.info(
            "Decoded Token values - email: %s, sub: %s, hd: %s",
            decoded_token.get("email"),
            decoded_token.get("sub"),
            decoded_token.get("hd"),
        )

        entra_oid = _extract_workforce_oid(decoded_token)

        email = decoded_token.get("email")
        # In Workforce Identity Federation, the email might be in preferred_username or upn.
        # Do not fall back to `sub` (which is an opaque principal identifier, not an email).
        if not email:
            email = decoded_token.get("preferred_username")
        if not email:
            email = decoded_token.get("upn")

        if not email or not isinstance(email, str) or "@" not in email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Unauthorized: User email could not be confirmed from "
                    "IAP token."
                ),
            )

        name = decoded_token.get("name", email.split("@")[0])
        picture = decoded_token.get("picture", "")

        token_info_hd = decoded_token.get("hd")
        if not token_info_hd:
            token_info_hd = email.split("@")[-1]

        # If ALLOWED_ORGS is configured, check the user's organization (case-insensitive).
        if config_service.ALLOWED_ORGS:
            if (
                not token_info_hd
                or token_info_hd.lower() not in config_service.ALLOWED_ORGS
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        f"User from '{token_info_hd}' is not part of an "
                        "allowed organization."
                    ),
                )

        # Just-In-Time (JIT) User Provisioning:
        # Create a user profile in our database on their first API call.
        # Roles are reconciled against Entra ID (Microsoft Graph) inside the service.
        create_kwargs: dict[str, Any] = {
            "email": email,
            "name": name,
            "picture": picture,
        }
        if entra_oid is not None:
            create_kwargs["entra_oid"] = entra_oid
        user_doc = await user_service.create_user_if_not_exists(**create_kwargs)

        if not user_doc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create or retrieve user profile.",
            )

        if not user_doc.picture and picture:
            logger.info("Updating picture for user: %s", email)
            user_doc.picture = picture
            if user_doc.id:
                await user_service.user_repo.update(
                    user_doc.id, {"picture": picture}
                )

        return user_doc

    except HTTPException as e:
        logger.error("[get_current_user - HTTPException]: %s", e)
        raise e
    except Exception as e:
        logger.error("[get_current_user - Unexpected Exception]: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during authentication: {e}",
        ) from e


class RoleChecker:
    """Dependency that checks if the authenticated user has the required roles.
    It depends on `get_current_user` to ensure the user is authenticated first.
    """

    def __init__(self, allowed_roles: list[UserRoleEnum]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: UserModel = Depends(get_current_user)):
        """Checks the user's roles against the allowed roles."""
        is_authorized = any(role in self.allowed_roles for role in user.roles)

        if not is_authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "You do not have sufficient permissions to perform this "
                    "action."
                ),
            )
