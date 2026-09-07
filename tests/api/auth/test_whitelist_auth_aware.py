"""Auth-aware WHITELIST_PATHS (Medium finding).

The shipped default ``WHITELIST_PATHS=/health,/api/*`` exists so a zero-config
deployment (no ``AUTH_ACCOUNTS``, no ``LIGHTRAG_API_KEY``) keeps Ollama-client
compatibility out of the box -- harmless there, since every request is already
unauthenticated in that state regardless of the whitelist. But the whitelist
used to be consulted identically once an operator actually configured
authentication: the default's ``/api/*`` entry kept exempting the
Ollama-compatible router (invokes the LLM, can read the whole knowledge base)
from every auth check even though the operator had just turned auth on
specifically to protect the server. A startup banner warned about this, but a
warning is easy to miss in a daemonized/containerized deployment where stdout
isn't monitored.

The fix narrows -- never removes -- the whitelist: once EITHER AUTH_ACCOUNTS
or LIGHTRAG_API_KEY is configured, a whitelist match that falls inside the
``/api`` route tree no longer exempts the request; ``/health`` (and any other
non-``/api`` entry an operator added themselves) is unaffected. This file
pins both halves: the zero-config default staying exactly as open as before,
and the newly-protected behavior once either auth mode is turned on.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

API_KEY = "the-operators-secret-api-key"

# The shipped default: WHITELIST_PATHS=/health,/api/*
_DEFAULT_PATTERNS = [("/health", False), ("/api", True)]


def _scope(path: str) -> dict:
    return {"type": "http", "path": path, "root_path": ""}


@pytest.fixture(autouse=True)
def _default_whitelist(monkeypatch):
    monkeypatch.setattr(_utils_api, "whitelist_patterns", list(_DEFAULT_PATTERNS))


# --------------------------------------------------------------------------- #
# Unit-level: path_is_whitelisted directly
# --------------------------------------------------------------------------- #


def test_zero_config_api_route_stays_whitelisted(monkeypatch):
    """No AUTH_ACCOUNTS, no API key: the default whitelist's behavior for
    /api/* is completely unchanged by this fix."""
    monkeypatch.setattr(_utils_api, "auth_configured", False)

    assert _utils_api.path_is_whitelisted(_scope("/api/chat")) is True
    assert _utils_api.path_is_whitelisted(_scope("/health")) is True


def test_auth_accounts_configured_revokes_the_api_exemption(monkeypatch):
    """AUTH_ACCOUNTS configured (module-level auth_configured=True), no API
    key: /api/* is no longer exempt, but /health still is."""
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    assert _utils_api.path_is_whitelisted(_scope("/api/chat")) is False
    assert _utils_api.path_is_whitelisted(_scope("/health")) is True


def test_api_key_configured_revokes_the_api_exemption(monkeypatch):
    """LIGHTRAG_API_KEY configured, AUTH_ACCOUNTS not set: /api/* is no
    longer exempt, but /health still is.

    ``api_key_configured`` is call-site-specific (it comes from the
    ``api_key`` argument to ``get_combined_auth_dependency`` /
    ``AdmissionMiddleware``), unlike ``auth_configured`` which is a fixed
    module value -- this pins that it is honored too, not just AUTH_ACCOUNTS.
    """
    monkeypatch.setattr(_utils_api, "auth_configured", False)

    assert (
        _utils_api.path_is_whitelisted(_scope("/api/chat"), api_key_configured=True)
        is False
    )
    assert (
        _utils_api.path_is_whitelisted(_scope("/health"), api_key_configured=True)
        is True
    )


def test_non_api_whitelist_entries_are_unaffected_by_auth(monkeypatch):
    """The fix narrows the /api route tree specifically -- it must not
    disable a whitelist entry an operator added themselves for something
    unrelated to Ollama."""
    monkeypatch.setattr(
        _utils_api, "whitelist_patterns", [("/health", False), ("/webhook", True)]
    )
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    assert _utils_api.path_is_whitelisted(_scope("/webhook/incoming")) is True


def test_catch_all_whitelist_also_loses_the_api_exemption_under_auth(monkeypatch):
    """A catch-all ``/*`` entry compiles to an empty prefix and matches every
    route, including /api -- so it must lose the /api exemption exactly like
    an explicit /api/* entry once auth is configured, not just the shipped
    default literal string."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", [("", True)])
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    assert _utils_api.path_is_whitelisted(_scope("/api/chat")) is False
    # Everything else the catch-all covers is untouched.
    assert _utils_api.path_is_whitelisted(_scope("/documents")) is True


# --------------------------------------------------------------------------- #
# End to end over real HTTP, through the real dependency
# --------------------------------------------------------------------------- #


def _app(monkeypatch, *, auth_configured: bool, api_key: str | None) -> TestClient:
    monkeypatch.setattr(_utils_api, "auth_configured", auth_configured)
    dependency = _utils_api.get_combined_auth_dependency(api_key)
    app = FastAPI()

    @app.get("/health", dependencies=[Depends(dependency)])
    async def health():
        return {"status": "healthy"}

    # Method mirrors the production registration (see test_whitelist_path_prefix.py).
    @app.get("/api/tags", dependencies=[Depends(dependency)])
    async def ollama_tags():
        return {"models": []}

    @app.post("/api/chat", dependencies=[Depends(dependency)])
    async def ollama_chat():
        return {"message": "ok"}

    return TestClient(app)


def test_zero_config_api_routes_stay_unauthenticated(monkeypatch):
    """No AUTH_ACCOUNTS, no LIGHTRAG_API_KEY: the shipped default's
    fully-open behavior for /health and /api/* is unchanged by this fix."""
    client = _app(monkeypatch, auth_configured=False, api_key=None)

    assert client.get("/health").status_code == 200
    assert client.get("/api/tags").status_code == 200
    assert client.post("/api/chat").status_code == 200


def test_auth_accounts_configured_protects_the_ollama_routes(monkeypatch):
    """AUTH_ACCOUNTS configured: a request to /api/* without valid auth is
    now rejected -- before this fix it would pass straight through. /health
    stays exempt regardless."""
    client = _app(monkeypatch, auth_configured=True, api_key=None)

    assert client.get("/api/tags").status_code == 401
    assert client.post("/api/chat").status_code == 401
    assert client.get("/health").status_code == 200


def test_api_key_only_configured_protects_the_ollama_routes(monkeypatch):
    """LIGHTRAG_API_KEY configured (no AUTH_ACCOUNTS): a request to /api/*
    without a valid key is now rejected; a valid key still reaches it;
    /health stays exempt regardless.

    The rejection is 403, not 401: get_combined_auth_dependency's final
    fallback (utils_api.py, the API-key-only branch with no token/account
    auth configured) has always raised 403 Forbidden rather than 401 for a
    missing/invalid key in this mode -- pre-existing, unrelated to this fix.
    """
    client = _app(monkeypatch, auth_configured=False, api_key=API_KEY)

    assert client.get("/api/tags").status_code == 403
    assert (
        client.get("/api/tags", headers={"X-API-Key": API_KEY}).status_code == 200
    )
    assert client.get("/health").status_code == 200
