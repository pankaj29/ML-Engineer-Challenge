"""Central application configuration.

Everything the system needs to know about *where* things live and *how* it
should behave is defined here, and every value can be overridden with an
environment variable. Nothing is hardcoded in the business logic.

Why this matters (plain English):
    The exact same container image has to run on a laptop, in CI, and in
    production. The only thing that changes between those environments is the
    configuration. So we read configuration from environment variables (the
    "12-factor app" rule) instead of baking it into the code. That also means
    no password ever has to live in the source tree.

Usage::

    from api.config import settings
    settings.redis_url       # -> "redis://redis:6379/0"
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, resolved relative to this file so it works no matter what
# the current working directory is (a very common source of "file not found"
# bugs when the same code runs under uvicorn, pytest and celery).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Environment(str, Enum):
    """Which deployment environment we are running in."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


# Substrings that mark a secret as a checked-in default rather than a real one.
_PLACEHOLDER_MARKERS: tuple[str, ...] = ("replace-me", "do-not-deploy", "insecure", "changeme")
_PLACEHOLDER_KEY_PREFIXES: tuple[str, ...] = ("dev-key-", "replace-me")


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / `.env`.

    Field names map to env vars in upper case, e.g. ``redis_url`` is read from
    ``REDIS_URL``. See ``.env.example`` for the full list with comments.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Our settings include fields starting with `model_`; pydantic reserves
        # that prefix for its own methods, so we clear the protection.
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- app ---
    environment: Environment = Environment.LOCAL
    app_name: str = "Multi-Model Computer Vision API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # -------------------------------------------------------------- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ----------------------------------------------------------- datastores ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    database_url: str = "postgresql+asyncpg://mluser:mlpass@localhost:5432/mldb"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False

    # -------------------------------------------------------------- models ---
    model_registry_path: Path = REPO_ROOT / "models" / "registry.json"
    model_artifacts_dir: Path = REPO_ROOT / "models" / "artifacts"
    data_dir: Path = REPO_ROOT / "data"

    # Which runtime to prefer for inference. "onnx" is the production default
    # because it is meaningfully faster than eager PyTorch on CPU; "torch" is
    # the fallback used when an ONNX artifact is missing.
    preferred_runtime: Literal["onnx", "torch", "tensorrt"] = "onnx"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Where similarity vectors live. "memory" keeps them in this process,
    # which is fast and needs nothing, but gives N replicas N unrelated
    # indexes: an image indexed on one is not findable on another. "pgvector"
    # puts them in Postgres, which is already running, so every replica reads
    # and writes the same index.
    similarity_backend: Literal["memory", "pgvector"] = "memory"
    similarity_dimension: int = 2048

    # Load models when the process starts (production) instead of on the first
    # request (faster local iteration, but the first user pays the cost).
    eager_model_load: bool = True
    # Hard ceiling on concurrent inference calls, protecting the box from
    # being OOM-killed under a traffic spike.
    max_concurrent_inferences: int = 8
    inference_timeout_seconds: float = 30.0

    # ------------------------------------------------------------- security ---
    # SECURITY: there is no default secret. The app refuses to start in
    # staging/production unless this is set explicitly, so a deployment can
    # never accidentally ship with a well-known signing key.
    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    # Comma-separated `key:tier` pairs, e.g. "abc123:pro,def456:free".
    # In a real deployment these live in the database or a secrets manager;
    # this env-var form keeps local development frictionless.
    api_keys: str = Field(default="", repr=False)
    auth_enabled: bool = True
    cors_origins: str = "*"

    # --------------------------------------------------------- rate limiting ---
    rate_limit_enabled: bool = True
    # Requests allowed per minute, per tier. Tuned so that the free tier can
    # comfortably explore the docs while paid tiers get real throughput.
    rate_limit_free_rpm: int = 10
    rate_limit_basic_rpm: int = 60
    rate_limit_pro_rpm: int = 300
    rate_limit_enterprise_rpm: int = 3000
    # Burst allowance as a multiple of the per-minute rate (token bucket).
    rate_limit_burst_multiplier: float = 1.5

    # ------------------------------------------------------------- uploads ---
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_image_pixels: int = 50_000_000  # decompression-bomb guard
    min_image_dimension: int = 16
    max_image_dimension: int = 8192
    allowed_image_formats: str = "JPEG,PNG,WEBP,BMP"
    max_batch_size: int = 64

    # -------------------------------------------------------- observability ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True
    # Requests slower than this are logged at WARNING level as an early alert.
    slow_request_threshold_seconds: float = 1.0

    # ------------------------------------------------------------- celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_time_limit: int = 900
    celery_task_soft_time_limit: int = 840

    # ---------------------------------------------------------- validators ---
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> Settings:
        """Refuse to boot with insecure settings outside local/test.

        This is a deliberate fail-fast: a misconfigured production deployment
        should crash loudly at startup, not silently serve traffic with
        authentication disabled.
        """
        if self.environment in (Environment.STAGING, Environment.PRODUCTION):
            if not self.jwt_secret or len(self.jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be set to at least 32 characters in "
                    f"{self.environment.value}. Generate one with: "
                    'python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
            # The dev compose default and the k8s placeholder are both over 32
            # characters, so length alone let them through.
            if any(m in self.jwt_secret.lower() for m in _PLACEHOLDER_MARKERS):
                raise ValueError(
                    f"JWT_SECRET is a placeholder value; set a real secret in "
                    f"{self.environment.value}."
                )
            placeholder_keys = [
                k for k in self.parsed_api_keys() if k.lower().startswith(_PLACEHOLDER_KEY_PREFIXES)
            ]
            if placeholder_keys:
                raise ValueError(
                    f"API_KEYS contains {len(placeholder_keys)} development or placeholder "
                    f"key(s); set real keys in {self.environment.value}."
                )
            if self.debug:
                raise ValueError("DEBUG must be false in staging/production")
            if not self.auth_enabled:
                raise ValueError("AUTH_ENABLED must be true in staging/production")
        return self

    # ------------------------------------------------------------- helpers ---
    @property
    def allowed_formats_set(self) -> set[str]:
        """Image formats we accept, upper-cased, as a set."""
        return {f.strip().upper() for f in self.allowed_image_formats.split(",") if f.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list. ``"*"`` means "allow any origin"."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        """True when running with ``ENVIRONMENT=production``."""
        return self.environment == Environment.PRODUCTION

    def parsed_api_keys(self) -> dict[str, str]:
        """Parse ``API_KEYS`` into a ``{key: tier}`` mapping.

        Malformed entries are skipped rather than crashing the app, because a
        single typo in an env var should not take the whole service down.
        """
        mapping: dict[str, str] = {}
        for entry in self.api_keys.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            key, _, tier = entry.partition(":")
            key, tier = key.strip(), tier.strip().lower()
            if key and tier:
                mapping[key] = tier
        return mapping

    def rate_limit_for_tier(self, tier: str) -> int:
        """Requests-per-minute allowance for a user tier."""
        return {
            "free": self.rate_limit_free_rpm,
            "basic": self.rate_limit_basic_rpm,
            "pro": self.rate_limit_pro_rpm,
            "enterprise": self.rate_limit_enterprise_rpm,
        }.get(tier.lower(), self.rate_limit_free_rpm)

    def resolve_device(self) -> str:
        """Turn ``device="auto"`` into a concrete ``"cuda"`` or ``"cpu"``.

        torch is imported lazily so that modules which only need configuration
        (for example the Alembic migration runner) do not pay the multi-second
        cost of importing it.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dependency
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that the `.env` file is parsed once per process. Tests clear the
    cache (``get_settings.cache_clear()``) after patching environment variables.
    """
    return Settings()


settings: Settings = get_settings()

__all__ = ["REPO_ROOT", "Environment", "Settings", "get_settings", "settings"]
