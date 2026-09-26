"""Production start-up guard in api.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.config import Settings

REAL_SECRET = "Zq3v8Lk1Nw5Rt7Yp2Hs9Dm4Fb6Jc0Xg-real-looking"


def _prod(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "jwt_secret": REAL_SECRET,
        "api_keys": "k7Hq2mZp9:pro",
        "debug": False,
        "auth_enabled": True,
    }
    values.update(overrides)
    return Settings(**values)


def test_real_values_are_accepted() -> None:
    assert _prod().jwt_secret == REAL_SECRET


def test_short_secret_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        _prod(jwt_secret="too-short")


@pytest.mark.parametrize(
    "secret",
    [
        # docker-compose.yml development default
        "dev-only-insecure-jwt-secret-do-not-deploy",
        # k8s/base/config.yaml placeholder
        "replace-me-with-at-least-32-characters-of-random",
    ],
)
def test_checked_in_placeholder_secrets_are_refused(secret: str) -> None:
    assert len(secret) >= 32, "the point is that length alone does not catch these"
    with pytest.raises(ValidationError, match="placeholder"):
        _prod(jwt_secret=secret)


@pytest.mark.parametrize(
    "keys", ["dev-key-free:free,dev-key-pro:pro", "replace-me-free:free,realkey123:pro"]
)
def test_placeholder_api_keys_are_refused(keys: str) -> None:
    with pytest.raises(ValidationError, match="API_KEYS"):
        _prod(api_keys=keys)


def test_debug_is_refused() -> None:
    with pytest.raises(ValidationError, match="DEBUG"):
        _prod(debug=True)


def test_disabled_auth_is_refused() -> None:
    with pytest.raises(ValidationError, match="AUTH_ENABLED"):
        _prod(auth_enabled=False)


def test_local_keeps_its_defaults() -> None:
    settings = Settings(
        environment="local",
        jwt_secret="dev-only-insecure-jwt-secret-do-not-deploy",
        api_keys="dev-key-free:free",
    )
    assert settings.parsed_api_keys() == {"dev-key-free": "free"}
