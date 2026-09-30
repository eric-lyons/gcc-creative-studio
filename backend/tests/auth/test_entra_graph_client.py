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
"""Hermetic tests for EntraGraphClient (httpx.MockTransport, no network)."""

import asyncio
import json

import httpx
import pytest

from src.auth.entra_graph_client import (
    EntraGraphClient,
    EntraGraphError,
    EntraUserNotFoundError,
)

TOKEN_HOST = "login.microsoftonline.com"


class FakeGraph:
    """Records requests and serves canned Graph / token responses."""

    def __init__(self, members=(), upn_exists=True, mail_matches=1):
        self.members = {m.lower() for m in members}
        self.upn_exists = upn_exists
        self.mail_matches = mail_matches
        self.requests: list[httpx.Request] = []
        self.graph_status: int | None = None
        self.raise_transport = False
        self.delay = 0.0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if request.url.host == TOKEN_HOST:
            return httpx.Response(
                200, json={"access_token": "tok", "expires_in": 3600}
            )
        if self.raise_transport:
            raise httpx.ConnectError("boom", request=request)
        if self.graph_status:
            return httpx.Response(self.graph_status)
        path = request.url.path
        if path == "/v1.0/users" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": f"id{i}"} for i in range(self.mail_matches)
                    ]
                },
            )
        if path.endswith("/checkMemberGroups"):
            if "@" in path and not self.upn_exists:
                return httpx.Response(404)
            ids = json.loads(request.content)["groupIds"]
            return httpx.Response(
                200,
                json={"value": [g for g in ids if g.lower() in self.members]},
            )
        return httpx.Response(500)

    def calls(self, host_or_suffix: str) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if r.url.host == host_or_suffix
            or r.url.path.endswith(host_or_suffix)
        ]


def make_client(fake: FakeGraph) -> EntraGraphClient:
    return EntraGraphClient(
        "tenant",
        "client",
        "secret",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
    )


@pytest.mark.anyio
async def test_happy_path_uses_upn_and_one_graph_call():
    fake = FakeGraph(members={"AAA"})
    client = make_client(fake)

    result = await client.member_group_ids("alice@corp.com", ["aaa", "bbb"])

    assert result == {"aaa"}
    graph_calls = fake.calls("/checkMemberGroups")
    assert len(graph_calls) == 1
    assert (
        graph_calls[0].url.path
        == "/v1.0/users/alice@corp.com/checkMemberGroups"
    )
    assert graph_calls[0].headers["Authorization"] == "Bearer tok"


@pytest.mark.anyio
async def test_app_token_is_cached_across_calls():
    fake = FakeGraph()
    client = make_client(fake)

    await client.member_group_ids("a@corp.com", ["g1"])
    await client.member_group_ids("b@corp.com", ["g1"])

    assert len(fake.calls(TOKEN_HOST)) == 1
    token_form = dict(
        pair.split("=")
        for pair in fake.calls(TOKEN_HOST)[0].content.decode().split("&")
    )
    assert token_form["grant_type"] == "client_credentials"


@pytest.mark.anyio
async def test_group_ids_are_chunked_at_20_per_call():
    groups = [f"g{i:02d}" for i in range(45)]
    fake = FakeGraph(members={"g00", "g44"})
    client = make_client(fake)

    result = await client.member_group_ids("a@corp.com", groups)

    sizes = [
        len(json.loads(r.content)["groupIds"])
        for r in fake.calls("/checkMemberGroups")
    ]
    assert sizes == [20, 20, 5]
    assert result == {"g00", "g44"}


@pytest.mark.anyio
async def test_upn_404_falls_back_to_mail_filter_with_escaping():
    fake = FakeGraph(members={"g1"}, upn_exists=False)
    client = make_client(fake)

    result = await client.member_group_ids("o'neil@corp.com", ["g1"])

    assert result == {"g1"}
    lookup = next(r for r in fake.requests if r.url.path == "/v1.0/users")
    assert lookup.headers["ConsistencyLevel"] == "eventual"
    assert lookup.url.params["$count"] == "true"
    assert "mail eq 'o''neil@corp.com'" in lookup.url.params["$filter"]
    assert fake.calls("/checkMemberGroups")[-1].url.path == (
        "/v1.0/users/id0/checkMemberGroups"
    )


@pytest.mark.parametrize("matches", [0, 2])
@pytest.mark.anyio
async def test_mail_fallback_requires_exactly_one_match(matches):
    fake = FakeGraph(upn_exists=False, mail_matches=matches)
    client = make_client(fake)

    with pytest.raises(EntraUserNotFoundError):
        await client.member_group_ids("a@corp.com", ["g1"])


@pytest.mark.anyio
async def test_graph_5xx_raises_entra_graph_error():
    fake = FakeGraph()
    fake.graph_status = 503
    client = make_client(fake)

    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["g1"])


@pytest.mark.anyio
async def test_transport_error_raises_entra_graph_error():
    fake = FakeGraph()
    fake.raise_transport = True
    client = make_client(fake)

    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["g1"])


@pytest.mark.anyio
async def test_token_failure_raises_entra_graph_error():
    async def handler(request):
        return httpx.Response(401)

    client = EntraGraphClient(
        "t",
        "c",
        "s",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["g1"])


def _body_response(body) -> httpx.Response:
    if isinstance(body, bytes):
        return httpx.Response(200, content=body)
    return httpx.Response(200, json=body)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        {"token_type": "Bearer"},
        [],
        {"access_token": "tok", "expires_in": "soon"},
    ],
    ids=["non-json", "no-access-token", "list", "bad-expires-in"],
)
@pytest.mark.anyio
async def test_malformed_token_body_raises_entra_graph_error(body):
    async def handler(request):
        return _body_response(body)

    client = EntraGraphClient(
        "t",
        "c",
        "s",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["g1"])
    assert client._token is None  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        {},
        {"value": None},
        {"value": 5},
        {"value": "abc"},
        [],
        {"value": [1]},
    ],
    ids=[
        "non-json",
        "missing-value",
        "null-value",
        "int-value",
        "str-value",
        "list-body",
        "non-str-group",
    ],
)
@pytest.mark.anyio
async def test_malformed_check_member_groups_body_raises_entra_graph_error(
    body,
):
    async def handler(request):
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={"access_token": "tok"})
        return _body_response(body)

    client = EntraGraphClient(
        "t",
        "c",
        "s",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["abc"])


@pytest.mark.parametrize(
    "body",
    [{"value": [{}]}, {"value": ["x"]}],
    ids=["user-without-id", "non-object-user"],
)
@pytest.mark.anyio
async def test_malformed_mail_lookup_body_raises_entra_graph_error(body):
    async def handler(request):
        if request.url.host == TOKEN_HOST:
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path.endswith("/checkMemberGroups"):
            return httpx.Response(404)
        return _body_response(body)

    client = EntraGraphClient(
        "t",
        "c",
        "s",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EntraGraphError):
        await client.member_group_ids("a@corp.com", ["g1"])


@pytest.mark.anyio
async def test_concurrent_lookups_for_same_email_share_one_round_trip():
    fake = FakeGraph(members={"g1"})
    fake.delay = 0.01
    client = make_client(fake)

    results = await asyncio.gather(
        *(client.member_group_ids("a@corp.com", ["g1"]) for _ in range(5))
    )

    assert results == [{"g1"}] * 5
    assert len(fake.calls("/checkMemberGroups")) == 1
    assert client._inflight == {}  # pylint: disable=protected-access
