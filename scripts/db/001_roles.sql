-- =============================================================================
-- 001_roles.sql: least-privilege roles and schema for simple-rag-db
-- =============================================================================
--
-- WHAT THIS CREATES
-- -----------------
--   simple_rag_owner   Owns schema `rag`. Runs DDL (CREATE/ALTER/DROP).
--                      Used ONLY by Alembic migrations (MIGRATIONS_DATABASE_URL).
--
--   simple_rag_app     SELECT / INSERT / UPDATE / DELETE on tables in `rag`.
--                      No DDL, cannot create objects anywhere.
--                      Used by the API and the ingestion worker (DATABASE_URL).
--
--   schema rag         All application tables live here (not in `public`).
--
-- Tables that future migrations create (as simple_rag_owner) are granted to
-- simple_rag_app automatically through ALTER DEFAULT PRIVILEGES. You never
-- need to re-run this script after a migration.
--
-- HOW TO RUN (once, as a superuser)
-- ---------------------------------
--   psql -h <db-host> -U postgres -d simple-rag-db -f scripts/db/001_roles.sql
--
--   At the end you are prompted (input hidden) for each role's password.
--   psql's \password hashes the password client-side (SCRAM-SHA-256), so the
--   plain-text password never appears in server logs or shell history.
--
--   Re-running is safe: every statement is idempotent. To re-run without
--   being prompted for passwords again:
--     psql -h <db-host> -U postgres -d simple-rag-db -v skip_passwords=1 \
--          -f scripts/db/001_roles.sql
--
-- AFTER RUNNING
-- -------------
--   1. Put both passwords into `.env` (URL-encode special characters):
--        DATABASE_URL=postgresql+asyncpg://simple_rag_app:<pw>@<db-host>:5432/simple-rag-db
--        MIGRATIONS_DATABASE_URL=postgresql+asyncpg://simple_rag_owner:<pw>@<db-host>:5432/simple-rag-db
--   2. Check that pg_hba.conf on the server allows both roles from your machine,
--      e.g.:  host  simple-rag-db  simple_rag_owner,simple_rag_app  <your-ip>/32  scram-sha-256
--      (If an existing `host all all ...` rule covers you, nothing to do.)
--   3. Verify the connections:
--        psql "postgresql://simple_rag_app@<db-host>:5432/simple-rag-db"   -c "SHOW search_path;"
--        psql "postgresql://simple_rag_owner@<db-host>:5432/simple-rag-db" -c "SHOW search_path;"
--
-- NOTE ON SHARED SERVERS
-- ----------------------
--   This script revokes the default PUBLIC privileges on simple-rag-db, so
--   other roles on the server (which can connect to any database by default)
--   can no longer connect to it. Superusers are unaffected.
--
-- TO UNDO (manual; destroys all application data)
-- -----------------------------------------------
--   DROP SCHEMA rag CASCADE;
--   DROP OWNED BY simple_rag_app;   DROP ROLE simple_rag_app;
--   DROP OWNED BY simple_rag_owner; DROP ROLE simple_rag_owner;
--   GRANT CONNECT, TEMPORARY ON DATABASE "simple-rag-db" TO PUBLIC;
-- =============================================================================

\set ON_ERROR_STOP on

-- -----------------------------------------------------------------------------
-- 0. Safety checks: right database, sufficient privileges
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF current_database() <> 'simple-rag-db' THEN
        RAISE EXCEPTION 'Connected to "%", expected "simple-rag-db" (use -d simple-rag-db)',
            current_database();
    END IF;
    IF NOT (SELECT rolsuper FROM pg_roles WHERE rolname = current_user) THEN
        RAISE EXCEPTION 'Run this script as a superuser (current user: %)', current_user;
    END IF;
END
$$;

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. Roles: created if missing, then attributes are (re)applied on every run
-- -----------------------------------------------------------------------------
SELECT 'CREATE ROLE simple_rag_owner'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'simple_rag_owner') \gexec

SELECT 'CREATE ROLE simple_rag_app'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'simple_rag_app') \gexec

ALTER ROLE simple_rag_owner WITH
    LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 5;

-- API pool (5 + 10 overflow) + worker pool (same) = 30.
ALTER ROLE simple_rag_app WITH
    LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 30;

COMMENT ON ROLE simple_rag_owner IS 'simple-rag: schema owner, migrations only';
COMMENT ON ROLE simple_rag_app   IS 'simple-rag: application runtime, DML only';

-- -----------------------------------------------------------------------------
-- 2. Database-level access: only these two roles may connect
-- -----------------------------------------------------------------------------
REVOKE ALL ON DATABASE "simple-rag-db" FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE "simple-rag-db" TO simple_rag_owner;
GRANT CONNECT            ON DATABASE "simple-rag-db" TO simple_rag_app;

-- -----------------------------------------------------------------------------
-- 3. Schema `public`: no one creates objects here.
--    USAGE stays granted to PUBLIC (the PostgreSQL default) because the
--    pgvector extension, i.e. the `vector` type and its operators, lives in
--    `public` and both roles need to reference it.
-- -----------------------------------------------------------------------------
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- -----------------------------------------------------------------------------
-- 4. Extensions (superuser-only; already installed, kept for completeness)
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------------
-- 5. Application schema `rag`, owned by simple_rag_owner
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS rag AUTHORIZATION simple_rag_owner;
ALTER SCHEMA rag OWNER TO simple_rag_owner;
COMMENT ON SCHEMA rag IS 'agentic-rag application schema';

REVOKE ALL   ON SCHEMA rag FROM PUBLIC;
GRANT  USAGE ON SCHEMA rag TO simple_rag_app;

-- -----------------------------------------------------------------------------
-- 6. Privileges for the app role
--    a) on objects that already exist (makes re-runs converge)
--    b) on objects that simple_rag_owner creates in the future (migrations)
-- -----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES    IN SCHEMA rag TO simple_rag_app;
GRANT USAGE, SELECT                  ON ALL SEQUENCES IN SCHEMA rag TO simple_rag_app;
GRANT EXECUTE                        ON ALL FUNCTIONS IN SCHEMA rag TO simple_rag_app;

ALTER DEFAULT PRIVILEGES FOR ROLE simple_rag_owner IN SCHEMA rag
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO simple_rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE simple_rag_owner IN SCHEMA rag
    GRANT USAGE, SELECT ON SEQUENCES TO simple_rag_app;

-- Functions are executable by PUBLIC by default; restrict them to the app role.
ALTER DEFAULT PRIVILEGES FOR ROLE simple_rag_owner
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE simple_rag_owner IN SCHEMA rag
    GRANT EXECUTE ON FUNCTIONS TO simple_rag_app;

-- -----------------------------------------------------------------------------
-- 7. Per-role session defaults in this database
-- -----------------------------------------------------------------------------
-- Unqualified names resolve to `rag` first, then `public` (for pgvector).
ALTER ROLE simple_rag_owner IN DATABASE "simple-rag-db" SET search_path = rag, public;
ALTER ROLE simple_rag_app   IN DATABASE "simple-rag-db" SET search_path = rag, public;

-- Runtime guard rails: no runaway queries or sessions stuck in a transaction.
ALTER ROLE simple_rag_app IN DATABASE "simple-rag-db" SET statement_timeout = '30s';
ALTER ROLE simple_rag_app IN DATABASE "simple-rag-db" SET idle_in_transaction_session_timeout = '60s';

-- Migrations should fail fast instead of queueing behind long-held locks.
ALTER ROLE simple_rag_owner IN DATABASE "simple-rag-db" SET lock_timeout = '10s';

COMMIT;

-- -----------------------------------------------------------------------------
-- 8. Passwords (prompted, hidden, hashed client-side)
-- -----------------------------------------------------------------------------
\if :{?skip_passwords}
    \echo 'Skipping password prompts (skip_passwords is set).'
\else
    \echo 'Set password for simple_rag_owner:'
    \password simple_rag_owner
    \echo 'Set password for simple_rag_app:'
    \password simple_rag_app
\endif

-- -----------------------------------------------------------------------------
-- 9. Summary
-- -----------------------------------------------------------------------------
\echo ''
\echo '== Roles =='
SELECT rolname, rolcanlogin AS login, rolsuper AS superuser, rolcreatedb AS createdb,
       rolconnlimit AS conn_limit, rolpassword IS NOT NULL AS has_password
FROM pg_authid
WHERE rolname IN ('simple_rag_owner', 'simple_rag_app')
ORDER BY rolname;

\echo '== Role settings in simple-rag-db =='
SELECT r.rolname, unnest(s.setconfig) AS setting
FROM pg_db_role_setting s
JOIN pg_roles r ON r.oid = s.setrole
JOIN pg_database d ON d.oid = s.setdatabase
WHERE d.datname = 'simple-rag-db'
ORDER BY r.rolname, setting;

\echo '== Schema rag =='
\dn+ rag

\echo '== Default privileges =='
\ddp
