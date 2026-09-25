-- Enable pgvector on first boot.
--
-- The API creates this extension itself when SIMILARITY_BACKEND=pgvector, but
-- only if its database user has rights to. On a managed Postgres it usually
-- does not, so doing it here, as the superuser the init scripts run as, means
-- the application never needs that privilege.
--
-- Harmless when the similarity index runs in memory: an unused extension
-- costs nothing.
CREATE EXTENSION IF NOT EXISTS vector;
