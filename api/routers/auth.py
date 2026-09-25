"""Token issuance.

Plain English:
    Trade an API key for a short-lived bearer token. The key is the long-lived
    secret and should stay on a server; the token expires, so it is the safer
    thing to hand to a browser or a mobile client.

The rest of the API already accepted JWTs, but nothing issued them, so the
bearer path could not be exercised end to end. This closes that.

A real deployment puts an identity provider here (Auth0, Cognito, Keycloak) and
this endpoint goes away. What stays is the contract: the caller presents proof
of identity and receives a signed token whose claims the API trusts.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from api.config import settings
from api.dependencies import CorrelationId
from api.logging_config import get_logger
from api.middleware.auth import create_access_token, verify_api_key
from api.models.responses import ErrorResponse, TokenResponse
from api.models.schemas import TokenRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange an API key for a bearer token",
    responses={
        401: {"model": ErrorResponse, "description": "The API key was not recognised"},
        503: {"model": ErrorResponse, "description": "No signing secret is configured"},
    },
)
async def issue_token(request: TokenRequest, correlation_id: CorrelationId) -> TokenResponse:
    """Mint a JWT for the caller identified by an API key.

    The key is verified the same way the auth middleware verifies it, so a key
    that cannot reach the API cannot mint a token either. The token carries the
    key's own tier, so a free-tier key issues a free-tier token and the rate
    limits that go with it.
    """
    principal = verify_api_key(request.api_key)

    # An API key is full access, and so is a JWT with no scopes, so the only
    # thing a scope list can do here is restrict the token. Granting exactly
    # what was asked for is therefore safe; there is nothing to escalate to.
    granted = list(request.scopes or [])

    token = create_access_token(
        user_id=principal.user_id,
        tier=principal.tier,
        scopes=granted,
    )

    logger.info(
        "token_issued",
        extra={
            "user_id": principal.user_id,
            "tier": principal.tier.value,
            "scopes": granted,
            "correlation_id": correlation_id,
        },
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",  # noqa: S106 - the RFC 6750 scheme name, not a secret
        expires_in=settings.jwt_expiry_minutes * 60,
        tier=principal.tier,
        scopes=granted,
    )


__all__ = ["router"]
