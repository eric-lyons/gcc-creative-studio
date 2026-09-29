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
"""Tests for Entra group -> app role reconciliation in UserService, plus the
config parsing and repository query shape it relies on."""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.entra_graph_client import EntraGraphError
from src.config.config_service import ConfigService
from src.users.repository.user_repository import UserRepository
from src.users.user_model import UserModel, UserRoleEnum
from src.users.user_service import UserService

ADMIN_G = "aaaaaaaa-0000-0000-0000-000000000001"
CREATOR_G = "cccccccc-0000-0000-0000-000000000002"
NOW = datetime.datetime.now(datetime.UTC)


@pytest.fixture(name="config")
def fixture_config():
    cfg = SimpleNamespace(
        ENTRA_ROLE_SYNC_ENABLED=True,
        ENTRA_ROLE_SYNC_TTL_SECONDS=600,
        ENTRA_GROUP_ROLES={
            ADMIN_G: frozenset({"admin"}),
            CREATOR_G: frozenset({"creator"}),
        },
    )
    with patch("src.users.user_service.config_service", cfg):
        yield cfg


@pytest.fixture(name="graph")
def fixture_graph():
    client = MagicMock()
    client.member_group_ids = AsyncMock(return_value=set())
    with patch(
        "src.users.user_service.get_entra_graph_client", return_value=client
    ):
        yield client


@pytest.fixture(name="repo")
def fixture_repo():
    repo = AsyncMock()
    repo.update.side_effect = lambda uid, data: SimpleNamespace(id=uid, **data)
    repo.count_admins.return_value = 5
    return repo


def _user(roles, checked_at=None, **overrides) -> UserModel:
    fields = {
        "id": 7,
        "email": "alice@corp.com",
        "name": "Alice",
        "roles": roles,
        "roles_checked_at": checked_at,
    }
    fields.update(overrides)
    return UserModel(**fields)


def _stale():
    return NOW - datetime.timedelta(seconds=601)


async def _call(repo, email="alice@corp.com"):
    return await UserService(user_repo=repo).create_user_if_not_exists(
        email=email, name="Alice", picture=""
    )


@pytest.mark.usefixtures("config")
class TestEntraRoleSync:
    @pytest.mark.anyio
    async def test_fresh_ttl_skips_graph_and_writes(self, repo, graph):
        user = _user([UserRoleEnum.USER], checked_at=NOW)
        repo.get_by_email.return_value = user

        assert await _call(repo) is user
        graph.member_group_ids.assert_not_called()
        repo.update.assert_not_called()

    @pytest.mark.anyio
    async def test_stale_promotion_updates_roles_once(self, repo, graph):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.USER], checked_at=_stale()
        )
        graph.member_group_ids.return_value = {ADMIN_G}

        await _call(repo)

        repo.update.assert_called_once()
        uid, data = repo.update.call_args.args
        assert uid == 7
        assert data["roles"] == ["user", "admin"]
        assert data["roles_checked_at"] >= NOW

    @pytest.mark.anyio
    async def test_stale_unchanged_only_bumps_marker(self, repo, graph):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.ADMIN, UserRoleEnum.USER], checked_at=_stale()
        )
        graph.member_group_ids.return_value = {ADMIN_G}

        await _call(repo)

        _, data = repo.update.call_args.args
        assert set(data) == {"roles_checked_at"}

    @pytest.mark.anyio
    async def test_demotion_when_removed_from_group(self, repo, graph):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.USER, UserRoleEnum.CREATOR], checked_at=None
        )
        graph.member_group_ids.return_value = set()

        await _call(repo)

        _, data = repo.update.call_args.args
        assert data["roles"] == ["user"]

    @pytest.mark.anyio
    async def test_last_admin_is_never_demoted(self, repo, graph):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.USER, UserRoleEnum.ADMIN], checked_at=_stale()
        )
        repo.count_admins.return_value = 1
        graph.member_group_ids.return_value = set()

        await _call(repo)

        _, data = repo.update.call_args.args
        assert "roles" not in data

    @pytest.mark.anyio
    async def test_graph_failure_is_fail_static_and_bumps_marker(
        self, repo, graph
    ):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.USER, UserRoleEnum.ADMIN], checked_at=_stale()
        )
        graph.member_group_ids.side_effect = EntraGraphError("down")

        await _call(repo)

        _, data = repo.update.call_args.args
        assert set(data) == {"roles_checked_at"}

    @pytest.mark.anyio
    async def test_sync_disabled_returns_existing_user(
        self, repo, graph, config
    ):
        config.ENTRA_ROLE_SYNC_ENABLED = False
        user = _user([UserRoleEnum.USER], checked_at=None)
        repo.get_by_email.return_value = user

        assert await _call(repo) is user
        graph.member_group_ids.assert_not_called()
        repo.update.assert_not_called()

    @pytest.mark.anyio
    async def test_new_user_gets_entra_roles_and_marker(self, repo, graph):
        repo.get_by_email.return_value = None
        graph.member_group_ids.return_value = {CREATOR_G}

        await _call(repo)

        data = repo.create.call_args.args[0]
        assert data["roles"] == ["user", "creator"]
        assert data["roles_checked_at"] >= NOW

    @pytest.mark.anyio
    async def test_new_user_defaults_to_user_when_graph_fails(
        self, repo, graph
    ):
        repo.get_by_email.return_value = None
        graph.member_group_ids.side_effect = EntraGraphError("down")

        await _call(repo)

        assert repo.create.call_args.args[0]["roles"] == ["user"]

    @pytest.mark.anyio
    async def test_email_is_lowercased_before_lookup_and_graph(
        self, repo, graph
    ):
        repo.get_by_email.return_value = None

        await _call(repo, email="  Alice@Corp.COM ")

        repo.get_by_email.assert_called_once_with(
            "alice@corp.com", include_deleted=True
        )
        assert graph.member_group_ids.call_args.args[0] == "alice@corp.com"
        assert repo.create.call_args.args[0]["email"] == "alice@corp.com"

    @pytest.mark.anyio
    async def test_soft_deleted_user_is_forbidden(self, repo, graph):
        repo.get_by_email.return_value = _user(
            [UserRoleEnum.USER], deleted_at=NOW
        )

        with pytest.raises(HTTPException) as exc_info:
            await _call(repo)

        assert exc_info.value.status_code == 403
        graph.member_group_ids.assert_not_called()
        repo.create.assert_not_called()


class TestEntraConfig:
    def _config(self, **env):
        return ConfigService(_env_file=None, PROJECT_ID="p", **env)

    def test_group_roles_parsed_once_lowercased_and_merged(self):
        cfg = self._config(
            ENTRA_ADMIN_GROUPS=f" {ADMIN_G.upper()} ,",
            ENTRA_CREATOR_GROUPS=f"{ADMIN_G},{CREATOR_G}",
        )
        roles = cfg.ENTRA_GROUP_ROLES
        assert roles == {
            ADMIN_G: frozenset({"admin", "creator"}),
            CREATOR_G: frozenset({"creator"}),
        }
        assert cfg.ENTRA_GROUP_ROLES is roles  # cached, not re-parsed

    def test_no_hardcoded_group_name_defaults(self):
        assert self._config().ENTRA_GROUP_ROLES == {}

    def test_sync_enabled_requires_credentials_and_groups(self):
        creds = {
            "ENTRA_TENANT_ID": "t",
            "ENTRA_GRAPH_CLIENT_ID": "c",
            "ENTRA_GRAPH_CLIENT_SECRET": "s",
        }
        assert not self._config(**creds).ENTRA_ROLE_SYNC_ENABLED
        assert not self._config(
            ENTRA_ADMIN_GROUPS=ADMIN_G
        ).ENTRA_ROLE_SYNC_ENABLED
        assert self._config(
            ENTRA_ADMIN_GROUPS=ADMIN_G, **creds
        ).ENTRA_ROLE_SYNC_ENABLED


class TestUserRepositoryQueries:
    @pytest.mark.anyio
    async def test_get_by_email_is_index_friendly_and_honours_include_deleted(
        self,
    ):
        db = AsyncMock()
        db.execute.return_value.scalar_one_or_none = MagicMock(
            return_value=None
        )
        repo = UserRepository(db=db)

        await repo.get_by_email(" Bob@Corp.com ", include_deleted=True)

        stmt = db.execute.call_args.args[0]
        assert "lower(" not in str(stmt).lower()
        assert stmt.compile().params["email_1"] == "bob@corp.com"
        assert stmt.get_execution_options()["include_deleted"] is True
