"""Redis connection URI redaction (Medium finding).

``REDIS_URI`` commonly embeds credentials (``redis://user:pass@host:port``),
and this module used to log it verbatim on pool creation, ref-count changes,
pool close/error, and the eviction-policy refusal -- at INFO/DEBUG/ERROR
levels. Any log aggregator or loosely-permissioned log file would then
capture the plaintext Redis password on every server start or reconnect.

``_redact_uri`` masks the password (and a bare username, if there is no
separate one) via ``urllib.parse`` rather than a naive string replace, so it
is robust to port-less/schemeless/IPv6/query-string URIs and passes a
credential-free URI through completely unchanged. This file pins both the
helper's own behavior and that every log call site in ``redis_impl.py`` that
touches the URI actually routes through it.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from lightrag.exceptions import StorageControlPlaneError
from lightrag.kg.redis_impl import (
    RedisConnectionManager,
    _ensure_no_eviction_policy,
    _redact_uri,
)
from lightrag.utils import logger

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline

# A distinctive fake password: asserting its exact absence is what proves
# redaction happened, rather than merely proving *some* string changed.
_FAKE_PASSWORD = "sUp3r-S3cr3t-Pa55w0rd!"


# --------------------------------------------------------------------------- #
# _redact_uri: unit behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "uri, expected",
    [
        (
            f"redis://user:{_FAKE_PASSWORD}@localhost:6379/0",
            "redis://user:***@localhost:6379/0",
        ),
        (
            f"redis://:{_FAKE_PASSWORD}@localhost:6379",
            "redis://***@localhost:6379",
        ),
        (
            f"rediss://admin:{_FAKE_PASSWORD}@redis.example.com:6380/2?ssl_cert_reqs=none",
            "rediss://admin:***@redis.example.com:6380/2?ssl_cert_reqs=none",
        ),
    ],
)
def test_redacts_the_password_but_keeps_host_port_db_visible(uri, expected):
    redacted = _redact_uri(uri)

    assert _FAKE_PASSWORD not in redacted
    assert redacted == expected


@pytest.mark.parametrize(
    "uri",
    [
        "redis://localhost:6379",
        "redis://localhost:6379/0",
        "redis://redis-host",
        "redis://user@localhost:6379",  # bare username, no password: no secret
    ],
)
def test_uri_with_no_credentials_passes_through_unchanged(uri):
    """A URI carrying no userinfo component at all must come back byte-for-byte
    identical -- not merely "close enough" -- so this never mangles the
    overwhelming majority of REDIS_URI values operators actually configure."""
    assert _redact_uri(uri) == uri


def test_malformed_uri_is_masked_wholesale_not_raised():
    """``urlsplit`` itself can raise (e.g. an unbracketed IPv6 host) -- nothing
    left that is safe to parse, so this masks wholesale rather than
    propagating the exception or risking an unredacted echo."""
    malformed = f"redis://user:{_FAKE_PASSWORD}@[::1:6379"

    redacted = _redact_uri(malformed)

    assert _FAKE_PASSWORD not in redacted


# --------------------------------------------------------------------------- #
# Integration: the actual log lines never carry the raw password
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_pool_registry():
    """``RedisConnectionManager`` pools are a process-wide registry; isolate
    it between tests so a URL reused across parametrized cases doesn't hit
    the "pool already exists" branch unexpectedly."""
    RedisConnectionManager._pools.clear()
    RedisConnectionManager._pool_refs.clear()
    yield
    RedisConnectionManager._pools.clear()
    RedisConnectionManager._pool_refs.clear()


@pytest.fixture(autouse=True)
def _reset_eviction_cache():
    """The eviction guard checks once per Redis URL per process; isolate that
    state the same way test_redis_eviction_guard.py does, so an earlier test's
    cache entry can't make ``_ensure_no_eviction_policy`` short-circuit before
    it ever reaches the log/exception line this test asserts on."""
    from lightrag.kg import redis_impl

    cache = getattr(redis_impl, "_eviction_checked", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def _fake_pool(monkeypatch):
    """A pool double whose ``aclose()`` is a real awaitable, standing in for
    the real ``redis.asyncio.ConnectionPool`` without any network dependency
    (per AGENTS.md: mock external services, don't depend on live ones)."""
    fake_pool = AsyncMock(name="fake_pool")
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.ConnectionPool.from_url", lambda *a, **k: fake_pool
    )
    return fake_pool


def test_get_pool_never_logs_the_raw_password(monkeypatch, caplog):
    uri = f"redis://admin:{_FAKE_PASSWORD}@redis-host:6379/0"
    _fake_pool(monkeypatch)

    logger.propagate = True
    try:
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            RedisConnectionManager.get_pool(uri)  # creation: logs at INFO
            RedisConnectionManager.get_pool(uri)  # second ref: logs at DEBUG
    finally:
        logger.propagate = False

    assert _FAKE_PASSWORD not in caplog.text
    assert "redis-host:6379" in caplog.text  # host/port stay visible for debugging


async def test_release_pool_never_logs_the_raw_password(monkeypatch, caplog):
    uri = f"redis://admin:{_FAKE_PASSWORD}@redis-host:6379/0"
    _fake_pool(monkeypatch)
    RedisConnectionManager.get_pool(uri)

    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            await RedisConnectionManager.release_pool(uri)  # ref count -> 0: closes
    finally:
        logger.propagate = False

    assert _FAKE_PASSWORD not in caplog.text
    assert "redis-host:6379" in caplog.text


async def test_close_all_pools_never_logs_the_raw_password(monkeypatch, caplog):
    uri = f"redis://admin:{_FAKE_PASSWORD}@redis-host:6379/0"
    _fake_pool(monkeypatch)
    RedisConnectionManager.get_pool(uri)

    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            await RedisConnectionManager.close_all_pools()
    finally:
        logger.propagate = False

    assert _FAKE_PASSWORD not in caplog.text
    assert "redis-host:6379" in caplog.text


async def test_eviction_refusal_never_leaks_the_raw_password(caplog):
    """The eviction-refusal message is constructed as an exception, not a
    direct ``logger.X()`` call -- but callers (``initialize()``) re-log it
    verbatim via ``{e}``, so the password must be redacted at the source or it
    reappears the moment a caller logs the caught exception."""
    fake = FakeRedis()
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "allkeys-lru"}
    uri = f"redis://admin:{_FAKE_PASSWORD}@redis-host:6379/0"

    with pytest.raises(StorageControlPlaneError) as exc_info:
        await _ensure_no_eviction_policy(fake, uri, "evict-ws")

    assert _FAKE_PASSWORD not in str(exc_info.value)
    assert "redis-host:6379" in str(exc_info.value)

    # Pin the actual regression: a caller that logs the caught exception (as
    # initialize() does) must not resurrect the password via {e}.
    logger.propagate = True
    try:
        with caplog.at_level(logging.ERROR, logger=logger.name):
            logger.error(f"[evict-ws] Failed to connect to Redis: {exc_info.value}")
    finally:
        logger.propagate = False
    assert _FAKE_PASSWORD not in caplog.text
