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
"""Regression tests for the Entra ID group -> app role sync design
(TDD_ENTRA_AD_GROUP_SYNC.md): roles come from Microsoft Graph, never from the
IAP token, and the surrounding auth defects stay fixed.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request

from src.auth.auth_guard import get_current_user
from src.auth.entra_graph_client import EntraGraphClient
from src.config.config_service import config_service
from src.users.user_model import UserModel, UserRoleEnum
from src.users.user_service import UserService

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _iap_production_config():
    config_service.ENVIRONMENT = "production"
    config_service.IAP_EXPECTED_AUDIENCE = "test-iap-audience"
    config_service.ALLOWED_ORGS_STR = ""
    yield
    config_service.ALLOWED_ORGS_STR = ""


def _user(**overrides) -> UserModel:
    fields = {
        "id": 1,
        "email": "alice@company.com",
        "name": "Alice",
        "roles": [UserRoleEnum.USER],
        "picture": "",
    }
    fields.update(overrides)
    return UserModel(**fields)


class TestRoleSourceIsNotTheIapToken:
    def test_wif_attribute_mapping_uses_oid_subject(self):
        """IAP drops google.groups from its JWT, so mapping it is dead weight,
        and google.subject maps to assertion.oid (immutable Entra object ID)."""
        content = (
            REPO_ROOT / "infra/modules/iap-load-balancer/main.tf"
        ).read_text(encoding="utf-8")
        assert '"google.subject"      = "assertion.oid"' in content
        assert "google.groups" not in content

    @pytest.mark.anyio
    @patch("src.auth.auth_guard.id_token.verify_token")
    async def test_guard_ignores_group_and_role_claims(self, mock_verify):
        mock_verify.return_value = {
            "email": "alice@company.com",
            "name": "Alice",
            "groups": ["11111111-1111-1111-1111-111111111111"],
            "roles": ["admin"],
            "google": {"groups": ["admins"]},
        }
        user_service = AsyncMock()
        user_service.create_user_if_not_exists.return_value = _user()

        await get_current_user(
            request=MagicMock(spec=Request),
            token="jwt",
            user_service=user_service,
        )

        user_service.create_user_if_not_exists.assert_called_once_with(
            email="alice@company.com", name="Alice", picture=""
        )


class TestAuthDefectsStayFixed:
    def test_removed_auth_session_endpoint_is_not_called(self, api_client):
        """IAP is the only login path: POST /api/auth/session stays removed
        and the frontend no longer calls it."""
        assert api_client.post("/api/auth/session").status_code == 404
        auth_service_ts = (
            REPO_ROOT / "frontend/src/app/common/services/auth.service.ts"
        ).read_text(encoding="utf-8")
        assert "/auth/session" not in auth_service_ts

    @pytest.mark.anyio
    @patch("src.auth.auth_guard.id_token.verify_token")
    async def test_mixed_case_upn_domain_passes_allowed_orgs(self, mock_verify):
        config_service.ALLOWED_ORGS_STR = "yourcompany.com"
        mock_verify.return_value = {"upn": "Alice.Smith@YourCompany.com"}
        user_service = AsyncMock()
        user_service.create_user_if_not_exists.return_value = _user()

        user = await get_current_user(
            request=MagicMock(spec=Request),
            token="jwt",
            user_service=user_service,
        )
        assert user is not None

    @pytest.mark.anyio
    @patch("src.auth.auth_guard.id_token.verify_token")
    async def test_sub_only_token_is_rejected_even_without_allowed_orgs(
        self, mock_verify
    ):
        """A WIF principal URI without an email claim or a non-GUID subject is
        rejected (fail closed) instead of using sub as email."""
        mock_verify.return_value = {
            "sub": "principal://iam.googleapis.com/locations/global/"
            "workforcePools/pool/subject/abc123"
        }
        user_service = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=MagicMock(spec=Request),
                token="jwt",
                user_service=user_service,
            )
        assert exc_info.value.status_code == 401
        user_service.create_user_if_not_exists.assert_not_called()

    @pytest.mark.anyio
    @patch("src.auth.auth_guard.id_token.verify_token")
    async def test_soft_deleted_user_gets_403_not_500(self, mock_verify):
        mock_verify.return_value = {"email": "gone@company.com", "name": "Gone"}
        repo = AsyncMock()
        repo.get_by_email.return_value = _user(
            email="gone@company.com", deleted_at="2026-01-01T00:00:00Z"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=MagicMock(spec=Request),
                token="jwt",
                user_service=UserService(user_repo=repo),
            )

        assert exc_info.value.status_code == 403
        repo.get_by_email.assert_called_once_with(
            "gone@company.com", include_deleted=True
        )
        repo.create.assert_not_called()


_ADMIN_GROUP = "aaaaaaaa-0000-0000-0000-000000000001"
_TOKEN_OK = {"access_token": "tok", "expires_in": 3600}


class TestMalformedGraphResponsesFailStatic:
    """A 200 with an unparseable Graph or token body is a Graph failure:
    the user keeps their roles and the TTL marker advances (no 401/500)."""

    @pytest.mark.parametrize(
        "token_body, graph_body",
        [
            pytest.param(b"<html>login</html>", None, id="a-token-non-json"),
            pytest.param(_TOKEN_OK, b"not json", id="b-graph-non-json"),
            pytest.param(
                {"token_type": "Bearer", "expires_in": 3600},
                None,
                id="c-token-missing-access-token",
            ),
        ],
    )
    @pytest.mark.anyio
    @patch("src.auth.auth_guard.id_token.verify_token")
    async def test_malformed_body_keeps_roles_and_advances_marker(
        self, mock_verify, token_body, graph_body
    ):
        mock_verify.return_value = {"email": "alice@company.com"}
        stale_marker = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            seconds=601
        )
        stale_user = _user(
            roles=[UserRoleEnum.USER, UserRoleEnum.ADMIN],
            roles_checked_at=stale_marker,
        )
        repo = AsyncMock()
        repo.get_by_email.return_value = stale_user
        repo.update.side_effect = lambda uid, data: stale_user.model_copy(
            update=data
        )

        def respond(body):
            if isinstance(body, bytes):
                return httpx.Response(200, content=body)
            return httpx.Response(200, json=body)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "login.microsoftonline.com":
                return respond(token_body)
            return respond(graph_body)

        graph_client = EntraGraphClient(
            "tenant",
            "client",
            "secret",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
        )
        sync_config = SimpleNamespace(
            ENTRA_ROLE_SYNC_ENABLED=True,
            ENTRA_ROLE_SYNC_TTL_SECONDS=600,
            ENTRA_GROUP_ROLES={_ADMIN_GROUP: frozenset({"admin"})},
        )

        with (
            patch("src.users.user_service.config_service", sync_config),
            patch(
                "src.users.user_service.get_entra_graph_client",
                return_value=graph_client,
            ),
        ):
            user = await get_current_user(
                request=MagicMock(spec=Request),
                token="jwt",
                user_service=UserService(user_repo=repo),
            )

        assert {UserRoleEnum(r) for r in user.roles} == {
            UserRoleEnum.USER,
            UserRoleEnum.ADMIN,
        }
        repo.update.assert_called_once()
        uid, updates = repo.update.call_args.args
        assert uid == stale_user.id
        assert set(updates) == {"roles_checked_at"}
        assert updates["roles_checked_at"] > stale_marker
