"""Database tables.

Plain English:
    Two things get written to PostgreSQL:

    * **Inference logs** — one row per prediction. This is the audit trail.
      It answers "what did the model say about this image, on which version,
      for which user, and how long did it take?" It is also the raw material
      for drift detection: comparing the distribution of today's predictions
      against last month's is how you notice a model going stale.

    * **Batch jobs** — the state of every background job, so that progress
      survives a worker restart.

Design notes:

* We store the image **hash**, never the image. Storing user images would be
  a privacy and storage problem; a hash still lets us detect duplicates and
  correlate repeat requests.
* Timestamps are ``TIMESTAMPTZ`` (timezone-aware). Naive timestamps are a
  reliable source of off-by-one-hour bugs the first time a server crosses a
  daylight-saving boundary.
* Indexes are chosen for the queries we actually run: "recent inferences for
  this model" (drift analysis) and "this user's recent activity" (support).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as a Python-side default."""
    return datetime.now(UTC)


class InferenceLog(Base):
    """One row per prediction served.

    This is the system's memory. Every analysis that happens later — accuracy
    monitoring, drift detection, cost attribution, incident investigation —
    reads this table.
    """

    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Request identity --------------------------------------------------
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # --- What ran ----------------------------------------------------------
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    runtime: Mapped[str] = mapped_column(String(32), nullable=False)
    device: Mapped[str] = mapped_column(String(16), nullable=False, default="cpu")

    # --- Input description (never the image itself) ------------------------
    image_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_format: Mapped[str | None] = mapped_column(String(16), nullable=True)
    image_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Result ------------------------------------------------------------
    # The top prediction is pulled out into its own columns because drift
    # queries group by it constantly, and digging into JSON for every row is
    # far slower than reading an indexed column.
    top_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    top_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_results: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # --- Performance -------------------------------------------------------
    preprocess_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    postprocess_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Outcome -----------------------------------------------------------
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )

    __table_args__ = (
        # "Recent predictions for model X" — the drift-detection query.
        Index("ix_inference_model_time", "model_name", "model_version", "created_at"),
        # "What did this user do recently?" — the support query.
        Index("ix_inference_user_time", "user_id", "created_at"),
        # "Show me the failures" — partial index keeps it small.
        Index("ix_inference_errors", "created_at", postgresql_where=(success.is_(False))),
        Index("ix_inference_correlation", "correlation_id"),
        Index("ix_inference_hash", "image_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"<InferenceLog id={self.id} task={self.task} "
            f"model={self.model_name}:{self.model_version} success={self.success}>"
        )


class BatchJob(Base):
    """State of one background batch job."""

    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Celery task id

    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)

    task: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    callback_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_batch_user_time", "user_id", "submitted_at"),
        Index("ix_batch_status", "status", "submitted_at"),
    )

    @property
    def progress_percent(self) -> float:
        """Completion percentage, counting both successes and failures."""
        if not self.total_items:
            return 0.0
        done = self.completed_items + self.failed_items
        return round(min(100.0, done / self.total_items * 100.0), 2)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock runtime, or None if it has not finished."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def __repr__(self) -> str:
        return f"<BatchJob id={self.id} status={self.status} {self.completed_items}/{self.total_items}>"


class ModelVersionRecord(Base):
    """Audit trail of model registrations and promotions.

    The JSON registry file says what is live *now*. This table says what was
    live *then*, which is what you need when asking "which weights produced
    this six-week-old prediction?"
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    artifact_paths: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_model_version", "name", "version", unique=True),)

    def __repr__(self) -> str:
        return f"<ModelVersionRecord {self.name}:{self.version} status={self.status}>"


__all__ = ["Base", "BatchJob", "InferenceLog", "ModelVersionRecord"]
