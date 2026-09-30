# Copyright 2026 Google LLC
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
"""Microsoft Graph client for Entra ID group-membership lookups (app-only).

The IAP signed-header JWT carries no group claims, so the backend asks Graph
directly which of the configured role groups a user belongs to.
"""

import asyncio
import logging
import time
from collections.abc import Iterable
from functools import lru_cache
from urllib.parse import quote

import httpx

from src.config.config_service import config_service

logger = logging.getLogger(__name__)

_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
# Documented limit of POST /users/{id}/checkMemberGroups.
_MAX_GROUP_IDS_PER_CALL = 20
# Refresh the app token this many seconds before Entra says it expires.
_TOKEN_EXPIRY_SKEW_SECONDS = 60


class EntraGraphError(Exception):
    """Graph or token endpoint failed; callers keep existing roles."""


class EntraUserNotFoundError(EntraGraphError):
    """No unique Entra user matched the email."""


class EntraGraphClient:
    """Resolves which configured Entra groups a user is a (transitive) member of."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._token_url = (
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http_client or httpx.AsyncClient(timeout=5.0)
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Future[set[str]]] = {}

    async def member_group_ids(
        self, email: str, group_ids: Iterable[str]
    ) -> set[str]:
        """Returns the lowercased subset of `group_ids` the user belongs to.

        Concurrent calls for the same email share one Graph round trip.
        """
        task = self._inflight.get(email)
        if task is None:
            task = asyncio.ensure_future(self._lookup(email, sorted(group_ids)))
            self._inflight[email] = task
            task.add_done_callback(lambda _: self._inflight.pop(email, None))
        return await asyncio.shield(task)

    async def _lookup(self, email: str, group_ids: list[str]) -> set[str]:
        try:
            # Most tenants use the email as the UPN: one call on the happy path.
            return await self._check_member_groups(
                quote(email, safe="@"), group_ids
            )
        except EntraUserNotFoundError:
            user_id = await self._find_user_id_by_mail(email)
            return await self._check_member_groups(user_id, group_ids)

    async def _check_member_groups(
        self, user_ref: str, group_ids: list[str]
    ) -> set[str]:
        matched: set[str] = set()
        for start in range(0, len(group_ids), _MAX_GROUP_IDS_PER_CALL):
            value = await self._request(
                "POST",
                f"/users/{user_ref}/checkMemberGroups",
                json={
                    "groupIds": group_ids[
                        start : start + _MAX_GROUP_IDS_PER_CALL
                    ]
                },
            )
            if not all(isinstance(g, str) for g in value):
                raise EntraGraphError(
                    f"Malformed Graph response for {user_ref}: non-string "
                    "group id"
                )
            matched.update(g.lower() for g in value)
        return matched

    async def _find_user_id_by_mail(self, email: str) -> str:
        literal = email.replace("'", "''")  # OData string-literal escaping
        users = await self._request(
            "GET",
            "/users",
            params={
                "$filter": (
                    f"mail eq '{literal}' or "
                    f"otherMails/any(m:m eq '{literal}')"
                ),
                "$select": "id",
                "$count": "true",
            },
            headers={"ConsistencyLevel": "eventual"},
        )
        if len(users) != 1:
            raise EntraUserNotFoundError(
                f"{len(users)} Entra users match mail {email!r}"
            )
        try:
            return users[0]["id"]
        except (KeyError, TypeError) as exc:
            raise EntraGraphError(
                f"Malformed Graph user entry for mail {email!r}: {exc}"
            ) from exc

    async def _request(
        self, method: str, path: str, headers: dict | None = None, **kwargs
    ) -> list:
        """Returns the `value` list of a Graph response; any malformed body
        raises EntraGraphError so callers keep existing roles."""
        token = await self._get_app_token()
        try:
            response = await self._http.request(
                method,
                f"{_GRAPH_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {token}", **(headers or {})},
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise EntraGraphError(f"Graph request failed: {exc}") from exc
        if response.status_code == 404:
            raise EntraUserNotFoundError(f"Graph 404 for {path}")
        if response.status_code >= 400:
            raise EntraGraphError(
                f"Graph HTTP {response.status_code} for {path}"
            )
        try:
            value = response.json()["value"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EntraGraphError(
                f"Malformed Graph response for {path}: {exc}"
            ) from exc
        if not isinstance(value, list):
            raise EntraGraphError(
                f"Malformed Graph response for {path}: 'value' is "
                f"{type(value).__name__}, not list"
            )
        return value

    async def _get_app_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            try:
                response = await self._http.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "scope": _GRAPH_SCOPE,
                    },
                )
            except httpx.HTTPError as exc:
                raise EntraGraphError(f"Token request failed: {exc}") from exc
            if response.status_code != 200:
                raise EntraGraphError(
                    f"Token request failed: HTTP {response.status_code}"
                )
            try:
                body = response.json()
                token = body["access_token"]
                expires_in = int(body.get("expires_in", 3600))
            except (ValueError, KeyError, TypeError) as exc:
                raise EntraGraphError(
                    f"Malformed token response: {exc}"
                ) from exc
            self._token = token
            self._token_expires_at = (
                time.monotonic() + expires_in - _TOKEN_EXPIRY_SKEW_SECONDS
            )
            return self._token


@lru_cache(maxsize=1)
def get_entra_graph_client() -> EntraGraphClient | None:
    """Process-wide client, or None when Entra role sync is not configured."""
    if not config_service.ENTRA_ROLE_SYNC_ENABLED:
        return None
    return EntraGraphClient(
        tenant_id=config_service.ENTRA_TENANT_ID,
        client_id=config_service.ENTRA_GRAPH_CLIENT_ID,
        client_secret=config_service.ENTRA_GRAPH_CLIENT_SECRET,
    )
