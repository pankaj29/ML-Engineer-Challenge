"""Token issuance.

The API accepted JWTs long before anything issued them, so the bearer path was
documented but unreachable. These cover the properties that make the endpoint
safe to expose, rather than re-testing `create_access_token`, which already has
its own tests.

The one that matters is escalation. This endpoint takes a credential and
returns a different credential, so the interesting question is never "does it
work" but "can a caller come out of it holding more than they went in with".
"""

from __future__ import annotations

import jwt
import pytest

from api.config import settings

TOKEN_URL = "/api/v1/auth/token"


def _issue(client, key: str, **body):
    return client.post(TOKEN_URL, json={"api_key": key, **body})


def _claims(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


class TestIssuing:
    def test_a_valid_key_returns_a_usable_token(self, api_client) -> None:
        response = _issue(api_client, "test-pro-key")
        assert response.status_code == 200

        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == settings.jwt_expiry_minutes * 60

        follow_up = api_client.get(
            "/api/v1/models", headers={"Authorization": f"Bearer {body['access_token']}"}
        )
        assert follow_up.status_code == 200, "the token the endpoint just issued was rejected"

    def test_the_token_is_reachable_without_credentials(self, api_client) -> None:
        """Otherwise you would need a token to get a token."""
        response = _issue(api_client, "test-pro-key")
        assert response.status_code != 401

    @pytest.mark.parametrize("key", ["not-a-real-key", "test-pro-key-almost", ""])
    def test_a_key_the_server_does_not_know_is_refused(self, api_client, key: str) -> None:
        response = _issue(api_client, key)
        assert response.status_code in (401, 422)
        assert "access_token" not in response.text


class TestNoEscalation:
    """A token must not be able to do more than the key that bought it."""

    def test_the_token_carries_the_key_s_own_tier(self, api_client) -> None:
        for key, expected in [
            ("test-free-key", "free"),
            ("test-pro-key", "pro"),
            ("test-ent-key", "enterprise"),
        ]:
            body = _issue(api_client, key).json()
            assert body["tier"] == expected
            assert _claims(body["access_token"])["tier"] == expected, (
                "the signed claim disagrees with the response body, so the tier "
                "shown to the caller is not the tier the API will enforce"
            )

    def test_a_free_key_cannot_buy_a_pro_token(self, api_client) -> None:
        """The tier is taken from the verified key, never from the request."""
        response = api_client.post(
            TOKEN_URL, json={"api_key": "test-free-key", "tier": "enterprise"}
        )
        # extra="forbid" rejects the unknown field outright. If that ever
        # relaxes, the tier must still come from the key.
        if response.status_code == 200:
            assert response.json()["tier"] == "free"
        else:
            assert response.status_code == 422

    def test_scopes_restrict_the_token(self, api_client) -> None:
        """A scoped token loses access the unscoped one has.

        `/models/reload` is the only scope-guarded route. An API key reaches it
        because keys are full access; a token scoped to something else must
        not.
        """
        unscoped = _issue(api_client, "test-ent-key").json()["access_token"]
        scoped = _issue(api_client, "test-ent-key", scopes=["read"]).json()["access_token"]

        allowed = api_client.post(
            "/api/v1/models/reload", headers={"Authorization": f"Bearer {unscoped}"}
        )
        denied = api_client.post(
            "/api/v1/models/reload", headers={"Authorization": f"Bearer {scoped}"}
        )

        assert allowed.status_code != 403
        assert (
            denied.status_code == 403
        ), "a token scoped to 'read' reached an admin route, so scopes are not enforced"


class TestClaims:
    def test_the_token_expires(self, api_client) -> None:
        claims = _claims(_issue(api_client, "test-pro-key").json()["access_token"])
        assert claims["exp"] > claims["iat"]
        assert claims["exp"] - claims["iat"] == settings.jwt_expiry_minutes * 60

    def test_the_subject_is_the_key_fingerprint_not_the_key(self, api_client) -> None:
        """The key itself must never end up inside a token that gets logged."""
        claims = _claims(_issue(api_client, "test-pro-key").json()["access_token"])
        assert claims["sub"].startswith("key_")
        assert "test-pro-key" not in claims["sub"]

    def test_the_response_never_echoes_the_key(self, api_client) -> None:
        assert "test-pro-key" not in _issue(api_client, "test-pro-key").text
