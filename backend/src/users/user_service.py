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


import datetime
import logging
from typing import Any

from fastapi import Depends, HTTPException, status

from src.auth.entra_graph_client import EntraGraphError, get_entra_graph_client
from src.common.dto.pagination_response_dto import PaginationResponseDto
from src.config.config_service import config_service
from src.users.dto.user_create_dto import UserCreateDto, UserUpdateRoleDto
from src.users.dto.user_search_dto import UserSearchDto
from src.users.repository.user_repository import UserRepository
from src.users.user_model import UserModel, UserRoleEnum

logger = logging.getLogger(__name__)

_ROLE_ORDER = {role: index for index, role in enumerate(UserRoleEnum)}


def _role_values(roles: set[UserRoleEnum]) -> list[str]:
    """Canonical (enum-ordered) string list for storage."""
    return [r.value for r in sorted(roles, key=_ROLE_ORDER.__getitem__)]


def _roles_check_due(user: UserModel, now: datetime.datetime) -> bool:
    return user.roles_checked_at is None or (
        now - user.roles_checked_at
    ) >= datetime.timedelta(seconds=config_service.ENTRA_ROLE_SYNC_TTL_SECONDS)


class UserService:
    """Handles the business logic for user management."""

    def __init__(self, user_repo: UserRepository = Depends()):
        self.user_repo = user_repo

    async def create_user_if_not_exists(
        self,
        email: str,
        name: str,
        picture: str | None,
        entra_oid: str | None = None,
    ) -> UserModel:
        """Gets or JIT-creates the user and, when Entra role sync is enabled,
        reconciles admin/creator/workflows roles against Entra group
        membership at most once per ENTRA_ROLE_SYNC_TTL_SECONDS.

        Users with `entra_oid` are keyed by that immutable object ID. An
        existing unlinked row with the same email is linked once (confirmed via
        Microsoft Graph when role sync is enabled).
        """
        email = email.strip().lower()
        if entra_oid is not None:
            entra_oid = entra_oid.strip().lower() or None

        sync_enabled = config_service.ENTRA_ROLE_SYNC_ENABLED
        pending_link_oid: str | None = None

        if entra_oid is not None:
            existing_user = await self.user_repo.get_by_entra_oid(
                entra_oid, include_deleted=True
            )
            if existing_user is None:
                email_user = await self.user_repo.get_by_email(
                    email, include_deleted=True
                )
                if email_user is not None:
                    if email_user.deleted_at is not None:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Forbidden: this account has been deactivated.",
                        )
                    if (
                        email_user.entra_oid is not None
                        and email_user.entra_oid != entra_oid
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=(
                                "Forbidden: email is already linked to a "
                                "different Entra identity."
                            ),
                        )
                    if sync_enabled:
                        await self._confirm_oid_email_via_graph(
                            entra_oid, email
                        )
                    else:
                        logger.info(
                            "Linking existing user %s to Entra OID %s "
                            "(Entra role sync disabled).",
                            email,
                            entra_oid,
                        )
                    pending_link_oid = entra_oid
                    existing_user = email_user
        else:
            existing_user = await self.user_repo.get_by_email(
                email, include_deleted=True
            )

        if existing_user and existing_user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: this account has been deactivated.",
            )

        now = datetime.datetime.now(datetime.UTC)

        if existing_user is None:
            user_data = UserCreateDto(
                email=email,
                name=name,
                picture=picture or "",
                entra_oid=entra_oid,
            ).model_dump(exclude_none=True)
            lookup_ref = entra_oid or email
            entra_roles = (
                await self._fetch_entra_roles(lookup_ref, email)
                if sync_enabled
                else None
            )
            user_data["roles"] = _role_values(
                entra_roles or {UserRoleEnum.USER}
            )
            if sync_enabled:
                user_data["roles_checked_at"] = now
            return await self.user_repo.create(user_data)

        if not sync_enabled or not _roles_check_due(existing_user, now):
            if pending_link_oid is not None:
                return (
                    await self.user_repo.update(
                        existing_user.id, {"entra_oid": pending_link_oid}
                    )
                    or existing_user
                )
            return existing_user

        # TTL expired: bump the marker even if Graph fails (fail-static,
        # retry after the next TTL) so an outage doesn't hammer Graph.
        updates: dict[str, Any] = {"roles_checked_at": now}
        if pending_link_oid is not None:
            updates["entra_oid"] = pending_link_oid
        lookup_ref = entra_oid or existing_user.entra_oid or existing_user.email
        entra_roles = await self._fetch_entra_roles(
            lookup_ref, existing_user.email
        )
        if entra_roles is not None:
            target = await self._apply_last_admin_safeguard(
                existing_user, entra_roles
            )
            current = {UserRoleEnum(r) for r in existing_user.roles}
            if target != current:
                updates["roles"] = _role_values(target)
                logger.info(
                    "Entra role sync for %s: %s -> %s",
                    existing_user.email,
                    _role_values(current),
                    updates["roles"],
                )
        return (
            await self.user_repo.update(existing_user.id, updates)
            or existing_user
        )

    async def _confirm_oid_email_via_graph(
        self, entra_oid: str, email: str
    ) -> None:
        """Confirms via Graph GET /users/{oid} that `email` belongs to `entra_oid`."""
        client = get_entra_graph_client()
        if client is None:
            return
        try:
            graph_emails = await client.get_user_emails(entra_oid)
        except EntraGraphError as exc:
            logger.error(
                "Failed to confirm Entra OID %s for %s via Graph: %s",
                entra_oid,
                email,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Unable to verify user identity with Microsoft Graph; "
                    "please try again later."
                ),
            ) from exc
        if email.strip().lower() not in graph_emails:
            logger.warning(
                "Refusing to link existing user %s to Entra OID %s: Graph "
                "reported emails %s",
                email,
                entra_oid,
                sorted(graph_emails),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Forbidden: Entra user profile email does not match "
                    "existing account."
                ),
            )

    async def _fetch_entra_roles(
        self, user_ref: str, log_email: str | None = None
    ) -> set[UserRoleEnum] | None:
        """Roles granted by Entra group membership, or None on any failure."""
        client = get_entra_graph_client()
        if client is None:
            return None
        group_roles = config_service.ENTRA_GROUP_ROLES
        try:
            matched = await client.member_group_ids(
                user_ref, group_roles.keys()
            )
        except EntraGraphError as exc:
            logger.error(
                "Entra role sync failed for %s; keeping existing roles: %s",
                log_email or user_ref,
                exc,
            )
            return None
        roles = {UserRoleEnum.USER}
        for group_id in matched:
            roles.update(UserRoleEnum(r) for r in group_roles.get(group_id, ()))
        return roles

    async def _apply_last_admin_safeguard(
        self, existing_user: UserModel, target: set[UserRoleEnum]
    ) -> set[UserRoleEnum]:
        """Entra is authoritative, except it may not remove the last admin."""
        if (
            UserRoleEnum.ADMIN in existing_user.roles
            and UserRoleEnum.ADMIN not in target
            and await self.user_repo.count_admins() <= 1
        ):
            logger.warning(
                "Entra removed admin from %s, but they are the last admin; "
                "keeping the admin role.",
                existing_user.email,
            )
            return target | {UserRoleEnum.ADMIN}
        return target

    async def get_user_by_id(self, user_id: int) -> UserModel | None:
        """Finds a single user by their ID."""
        return await self.user_repo.get_by_id(user_id)

    async def find_all_users(
        self,
        search_dto: UserSearchDto,
    ) -> PaginationResponseDto[UserModel]:
        """Retrieves a paginated list of all users."""
        return await self.user_repo.query(search_dto)

    async def delete_user(
        self, user_id: int, deleted_by: int | None = None
    ) -> bool:
        """Soft deletes a user."""
        return await self.user_repo.soft_delete(user_id, deleted_by=deleted_by)

    async def restore_user(self, user_id: int) -> bool:
        """Restores a soft-deleted user."""
        return await self.user_repo.restore(user_id)

    async def update_user_role(
        self,
        user_id: int,
        role_data: UserUpdateRoleDto,
    ) -> UserModel | None:
        """Updates the role of a specific user with safeties."""
        existing_user = await self.user_repo.get_by_id(user_id)
        if not existing_user:
            return None

        was_admin = "admin" in existing_user.roles
        will_be_admin = "admin" in [role.value for role in role_data.roles]

        if was_admin and not will_be_admin:
            if await self.user_repo.count_admins() <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="There must be at least 1 admin on the app.",
                )

        roles_as_strings = [role.value for role in role_data.roles]
        return await self.user_repo.update(user_id, {"roles": roles_as_strings})
