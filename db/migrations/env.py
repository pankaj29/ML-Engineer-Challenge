"""Alembic environment.

The URL comes from DATABASE_URL, not from alembic.ini, so the same migrations
run against a developer's SQLite and against production Postgres without a
second config file holding a password.

The async driver is swapped for a sync one here. Alembic runs migrations
synchronously, and `postgresql+asyncpg://` fails with "the asyncio extension
requires an async driver" if handed to a normal engine.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Migrations take their target from the "
            "environment so no connection string is committed."
        )
    # asyncpg -> psycopg2, aiosqlite -> sqlite. Alembic is synchronous.
    return url.replace("+asyncpg", "").replace("+aiosqlite", "")


#: Tables Alembic must leave alone.
#:
#: `similarity_vectors` is created by api/services/pgvector_index.py, not by
#: the ORM, because its embedding column is a pgvector `vector(N)` whose width
#: comes from the model. Autogenerate does not know about it, sees a table
#: with no model behind it, and writes a DROP. Without this the first
#: migration after any schema change deletes the similarity index.
UNMANAGED_TABLES = {"similarity_vectors"}


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "table" and name in UNMANAGED_TABLES)


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            # Without this a column type change is invisible to autogenerate,
            # and a migration silently omits it.
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
