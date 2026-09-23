"""Database access layer.

Plain English:
    This wraps PostgreSQL behind a small set of methods. Two rules shape the
    whole design:

    1. **Writes must never slow down or break a prediction.** Logging an
       inference is useful, but it is not worth failing a user's request over.
       Every write here is best-effort: if the database is down, we log a
       warning and carry on. The user still gets their answer.

    2. **The connection pool is created once.** Opening a PostgreSQL
       connection takes milliseconds and a chunk of server memory; doing it
       per request would dominate our latency budget.

The async engine (``asyncpg``) is used by the API. The Celery worker uses a
separate synchronous engine, because Celery tasks are not async.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import case, func, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.config import Settings, settings
from api.logging_config import get_logger
from db.models import Base, BatchJob, InferenceLog

logger = get_logger(__name__)


class DatabaseService:
    """Async access to the inference log and batch-job tables."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._available = False
        self._last_error: str | None = None

    # ------------------------------------------------------------ lifecycle --
    async def connect(self, *, create_tables: bool = False) -> bool:
        """Create the engine and verify the connection.

        Args:
            create_tables: Create tables if missing. Convenient for local
                development and tests; production uses Alembic migrations so
                that schema changes are versioned and reviewable.
        """
        try:
            engine_kwargs: dict[str, Any] = {"echo": self.settings.database_echo}

            # SQLite (used by the test suite) is served by StaticPool, which
            # has no concept of pool size or overflow and raises TypeError if
            # they are passed. Only PostgreSQL gets the pool configuration.
            if not self.settings.database_url.startswith("sqlite"):
                engine_kwargs.update(
                    pool_size=self.settings.database_pool_size,
                    max_overflow=self.settings.database_max_overflow,
                    pool_pre_ping=True,  # silently replace connections killed by a restart
                    pool_recycle=1800,  # avoid stale connections behind an idle-timeout proxy
                )
            else:
                from sqlalchemy.pool import StaticPool

                # An in-memory SQLite database lives inside one connection, so
                # every session must share that connection or each would see
                # its own empty database.
                engine_kwargs["poolclass"] = StaticPool
                engine_kwargs["connect_args"] = {"check_same_thread": False}

            self._engine = create_async_engine(self.settings.database_url, **engine_kwargs)
            self._session_factory = async_sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )

            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                if create_tables:
                    await conn.run_sync(Base.metadata.create_all)

            self._available = True
            self._last_error = None
            logger.info("database_connected", extra={"pool_size": self.settings.database_pool_size})
            return True
        except Exception as exc:
            self._available = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("database_unavailable_at_startup", extra={"error": self._last_error})
            return False

    async def close(self) -> None:
        """Dispose of the connection pool during shutdown."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
        self._available = False

    @property
    def available(self) -> bool:
        """True when the database is believed to be usable."""
        return self._available and self._session_factory is not None

    async def ping(self) -> bool:
        """Cheap health probe."""
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self._available = True
            return True
        except Exception as exc:
            self._available = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            return False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session, committing on success and rolling back on error."""
        if self._session_factory is None:
            raise RuntimeError("DatabaseService.connect() has not been called")
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------- writes --
    async def log_inference(self, record: dict[str, Any]) -> int | None:
        """Write one inference record. Never raises.

        Returns the new row id, or ``None`` if the write did not happen.
        A failure here is logged and swallowed on purpose: the user's
        prediction has already been computed and must still be returned.
        """
        if not self.available:
            return None
        try:
            async with self.session() as session:
                row = InferenceLog(**record)
                session.add(row)
                await session.flush()
                return row.id
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("inference_log_write_failed", extra={"error": self._last_error})
            return None

    async def create_batch_job(self, record: dict[str, Any]) -> bool:
        """Insert a batch job row."""
        if not self.available:
            return False
        try:
            async with self.session() as session:
                session.add(BatchJob(**record))
            return True
        except Exception as exc:
            logger.warning("batch_job_create_failed", extra={"error": str(exc)})
            return False

    async def get_batch_job(self, job_id: str) -> BatchJob | None:
        """Fetch one batch job by id."""
        if not self.available:
            return None
        try:
            async with self.session() as session:
                result = await session.execute(select(BatchJob).where(BatchJob.id == job_id))
                return result.scalar_one_or_none()
        except Exception as exc:
            logger.warning("batch_job_read_failed", extra={"error": str(exc)})
            return None

    async def update_batch_job(self, job_id: str, **fields: Any) -> bool:
        """Update fields on a batch job row."""
        if not self.available:
            return False
        try:
            async with self.session() as session:
                await session.execute(
                    update(BatchJob).where(BatchJob.id == job_id).values(**fields)
                )
            return True
        except Exception as exc:
            logger.warning("batch_job_update_failed", extra={"error": str(exc)})
            return False

    # -------------------------------------------------------------- reads --
    async def recent_predictions(
        self,
        model_name: str,
        model_version: str | None = None,
        *,
        since: datetime | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Fetch recent predictions for drift analysis.

        Returns the label and confidence of each prediction, which is what the
        statistical tests in :mod:`models.validation.drift` compare.
        """
        if not self.available:
            return []
        since = since or datetime.now(UTC) - timedelta(days=7)
        try:
            async with self.session() as session:
                stmt = (
                    select(
                        InferenceLog.top_label,
                        InferenceLog.top_confidence,
                        InferenceLog.created_at,
                        InferenceLog.inference_ms,
                    )
                    .where(
                        InferenceLog.model_name == model_name,
                        InferenceLog.created_at >= since,
                        InferenceLog.success.is_(True),
                    )
                    .order_by(InferenceLog.created_at.desc())
                    .limit(limit)
                )
                if model_version:
                    stmt = stmt.where(InferenceLog.model_version == model_version)
                result = await session.execute(stmt)
                return [
                    {
                        "label": r.top_label,
                        "confidence": r.top_confidence,
                        "created_at": r.created_at,
                        "inference_ms": r.inference_ms,
                    }
                    for r in result
                ]
        except Exception as exc:
            logger.warning("recent_predictions_failed", extra={"error": str(exc)})
            return []

    async def inference_stats(self, *, hours: int = 24) -> dict[str, Any]:
        """Aggregate statistics over a recent window, for dashboards."""
        if not self.available:
            return {}
        since = datetime.now(UTC) - timedelta(hours=hours)
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(
                        InferenceLog.model_name,
                        InferenceLog.model_version,
                        func.count().label("total"),
                        # `case` rather than casting the boolean: SQLite has no
                        # native boolean type, so a cast is dialect-dependent
                        # and fails. Summing 1/0 works on every backend.
                        func.sum(case((InferenceLog.success.is_(True), 1), else_=0)).label(
                            "successes"
                        ),
                        func.avg(InferenceLog.total_ms).label("avg_ms"),
                        func.max(InferenceLog.total_ms).label("max_ms"),
                    )
                    .where(InferenceLog.created_at >= since)
                    .group_by(InferenceLog.model_name, InferenceLog.model_version)
                )
                return {
                    "window_hours": hours,
                    "models": [
                        {
                            "model": f"{r.model_name}:{r.model_version}",
                            "total": r.total,
                            "successes": int(r.successes or 0),
                            "success_rate": (
                                round((r.successes or 0) / r.total, 4) if r.total else 0.0
                            ),
                            "avg_ms": round(float(r.avg_ms), 2) if r.avg_ms else None,
                            "max_ms": round(float(r.max_ms), 2) if r.max_ms else None,
                        }
                        for r in result
                    ],
                }
        except Exception as exc:
            logger.warning("inference_stats_failed", extra={"error": str(exc)})
            return {}

    async def health(self) -> dict[str, Any]:
        """Health report for the health endpoint."""
        started = time.perf_counter()
        ok = await self.ping()
        return {
            "status": "healthy" if ok else "unavailable",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2) if ok else None,
            "error": None if ok else self._last_error,
        }


# Process-wide singleton, created in the application lifespan handler.
_db_service: DatabaseService | None = None


def get_db_service() -> DatabaseService:
    """FastAPI dependency returning the shared :class:`DatabaseService`."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def set_db_service(service: DatabaseService | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _db_service
    _db_service = service


__all__ = ["DatabaseService", "get_db_service", "set_db_service"]
