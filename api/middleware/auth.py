"""Authentication and user tiers.

Plain English:
    Two ways to prove who you are:

    * **API key** in the ``X-API-Key`` header — simple, long-lived, good for
      server-to-server callers.
    * **JWT bearer token** in ``Authorization: Bearer <token>`` — short-lived
      and self-describing, good for user-facing apps.

    Whichever is used, the result is the same: a :class:`Principal` describing
    who is calling and which tier they are on. The tier is what the rate
    limiter and batch-size checks act on.

Security choices worth calling out:

* **No default credentials.** If ``API_KEYS`` is empty and auth is enabled,
  every request is rejected. There is no built-in "admin/admin".
* **Constant-time comparison.** API keys are checked with
  ``secrets.compare_digest``. A plain ``==`` returns as soon as it finds a
  differing byte, and that timing difference can be measured to recover a key
  one character at a time.
* **Keys are never logged.** Only a short prefix goes into logs, so a log
  dump does not become a credential dump.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Request

from api.config import Settings, settings
from api.exceptions import AuthenticationError, AuthorizationError
from api.logging_config import get_logger, user_id_var
from api.models.schemas import UserTier

logger = get_logger(__name__)

# Paths that must work without credentials: health checks for the load
# balancer, metrics for Prometheus, and the docs so a new user can read them.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)
# /auth/token is public because it is how a caller obtains credentials. It
# is not unauthenticated: the body carries an API key, which the handler
# verifies before minting anything.
PUBLIC_SUFFIXES: tuple[str, ...] = (
    "/health",
    "/health/live",
    "/health/ready",
    "/metrics",
    "/auth/token",
)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller.

    Attributes:
        user_id: Stable identifier used in logs and inference records.
        tier: Subscription tier, which drives rate and batch limits.
        auth_method: ``"api_key"``, ``"jwt"`` or ``"anonymous"``.
        scopes: Permissions carried by a JWT, if any.
    """

    user_id: str
    tier: UserTier
    auth_method: str
    scopes: tuple[str, ...] = ()

    @property
    def is_anonymous(self) -> bool:
        return self.auth_method == "anonymous"

    def require_scope(self, scope: str) -> None:
        """Raise unless the caller holds ``scope``.

        API keys are treated as full-access, so scope checks only constrain
        JWT callers. Enterprise-tier keys bypass scope checks entirely.
        """
        if self.auth_method == "api_key" or not self.scopes:
            return
        if scope not in self.scopes:
            raise AuthorizationError(
                f"This action requires the '{scope}' permission.",
                details={"required_scope": scope, "granted_scopes": list(self.scopes)},
            )


def _key_fingerprint(api_key: str) -> str:
    """Short, non-reversible identifier for an API key.

    Used as the user id and in logs so that activity can be attributed to a
    key without the key itself ever being written down.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def verify_api_key(api_key: str, config: Settings | None = None) -> Principal:
    """Check an API key and return the caller it identifies.

    Raises:
        AuthenticationError: The key is unknown.
    """
    cfg = config or settings
    known = cfg.parsed_api_keys()

    if not known:
        logger.error("no_api_keys_configured")
        raise AuthenticationError(
            "Authentication is enabled but no API keys are configured on the server.",
            internal_message="API_KEYS env var is empty; set it or disable AUTH_ENABLED",
        )

    # Compare against every key so that the time taken does not reveal
    # whether a prefix matched.
    matched_tier: str | None = None
    for candidate, tier in known.items():
        if secrets.compare_digest(api_key, candidate):
            matched_tier = tier

    if matched_tier is None:
        logger.warning("invalid_api_key", extra={"key_prefix": api_key[:6]})
        raise AuthenticationError("The supplied API key is not valid.")

    try:
        tier = UserTier(matched_tier)
    except ValueError:
        logger.warning("unknown_tier_in_config", extra={"tier": matched_tier})
        tier = UserTier.FREE

    return Principal(
        user_id=f"key_{_key_fingerprint(api_key)}",
        tier=tier,
        auth_method="api_key",
    )


def create_access_token(
    user_id: str,
    tier: UserTier = UserTier.FREE,
    scopes: list[str] | None = None,
    config: Settings | None = None,
) -> str:
    """Mint a signed JWT for a user.

    Included so the authentication flow is demonstrable end to end; a real
    deployment would issue these from an identity provider.

    Raises:
        AuthenticationError: No signing secret is configured.
    """
    import jwt

    cfg = config or settings
    if not cfg.jwt_secret:
        raise AuthenticationError(
            "Token issuance is not available.",
            internal_message="JWT_SECRET is not configured",
        )

    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "tier": tier.value,
        "scopes": scopes or [],
        "iat": now,
        "exp": now + timedelta(minutes=cfg.jwt_expiry_minutes),
        "iss": cfg.app_name,
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def verify_jwt(token: str, config: Settings | None = None) -> Principal:
    """Validate a JWT and return the caller it describes.

    Raises:
        AuthenticationError: The token is expired, tampered with, or unusable.
    """
    import jwt

    cfg = config or settings
    if not cfg.jwt_secret:
        raise AuthenticationError(
            "Token authentication is not available.",
            internal_message="JWT_SECRET is not configured",
        )

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            cfg.jwt_secret,
            # SECURITY: the accepted algorithm is pinned. Without this, a
            # forged token declaring "alg": "none" would be accepted as valid.
            algorithms=[cfg.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError(
            "Your access token has expired. Request a new one.",
            details={"reason": "expired"},
        ) from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("invalid_jwt", extra={"error": str(exc)})
        raise AuthenticationError(
            "The access token could not be validated.",
            details={"reason": "invalid"},
            internal_message=str(exc),
        ) from exc

    try:
        tier = UserTier(payload.get("tier", "free"))
    except ValueError:
        tier = UserTier.FREE

    return Principal(
        user_id=str(payload["sub"]),
        tier=tier,
        auth_method="jwt",
        scopes=tuple(payload.get("scopes", [])),
    )


def is_public_path(path: str) -> bool:
    """True when a path may be reached without credentials."""
    if path in PUBLIC_PATHS:
        return True
    # Exact match only. A plain endswith() also made /models/health and
    # /models/metrics public, because /models/{name} accepts any name.
    return any(path in (suffix, f"{settings.api_prefix}{suffix}") for suffix in PUBLIC_SUFFIXES)


def authenticate_request(request: Request, config: Settings | None = None) -> Principal:
    """Work out who is calling, from the request headers.

    Resolution order: API key header, then bearer token. When authentication
    is disabled (local development) every caller becomes an anonymous
    enterprise-tier principal so that limits do not get in the way.

    Raises:
        AuthenticationError: Credentials are missing or invalid.
    """
    cfg = config or settings

    if not cfg.auth_enabled:
        return Principal(user_id="anonymous", tier=UserTier.ENTERPRISE, auth_method="anonymous")

    api_key = request.headers.get("X-API-Key")
    if api_key:
        return verify_api_key(api_key, cfg)

    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return verify_jwt(authorization[7:].strip(), cfg)

    raise AuthenticationError(
        "No credentials supplied. Send an 'X-API-Key' header or an "
        "'Authorization: Bearer <token>' header.",
        details={"accepted_methods": ["X-API-Key", "Bearer token"]},
    )


class AuthMiddleware:
    """ASGI middleware that authenticates every non-public request.

    Written as raw ASGI rather than ``BaseHTTPMiddleware`` deliberately:
    ``BaseHTTPMiddleware`` wraps each request in an extra task and breaks
    ``contextvars`` propagation, which would lose the correlation id.
    """

    def __init__(self, app: Any, config: Settings | None = None) -> None:
        self.app = app
        self.settings = config or settings

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)

        if is_public_path(request.url.path):
            scope["state"]["principal"] = Principal(
                user_id="public", tier=UserTier.FREE, auth_method="anonymous"
            )
            await self.app(scope, receive, send)
            return

        try:
            principal = authenticate_request(request, self.settings)
        except AuthenticationError as exc:
            from api.exceptions import app_error_handler

            response = await app_error_handler(request, exc)
            await response(scope, receive, send)
            return

        scope["state"]["principal"] = principal
        user_id_var.set(principal.user_id)
        await self.app(scope, receive, send)


def get_principal(request: Request) -> Principal:
    """FastAPI dependency returning the authenticated caller.

    The middleware has already run by the time a route handler executes, so
    this only reads what it stored.
    """
    principal = getattr(request.state, "principal", None)
    if principal is None:
        # Reachable only if a route is mounted outside the middleware stack.
        return authenticate_request(request)
    return principal


__all__ = [
    "PUBLIC_PATHS",
    "AuthMiddleware",
    "Principal",
    "authenticate_request",
    "create_access_token",
    "get_principal",
    "is_public_path",
    "verify_api_key",
    "verify_jwt",
]
