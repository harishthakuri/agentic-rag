#!/bin/sh
# Runs once, on first start of the docker-compose Postgres container (an empty volume).
# Provisions the same least-privilege roles as production, via 001_roles.sql,
# then sets their passwords from environment variables.
set -eu

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector"
psql -v ON_ERROR_STOP=1 -v skip_passwords=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -f /scripts/001_roles.sql
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -v owner_pw="$OWNER_DB_PASSWORD" -v app_pw="$APP_DB_PASSWORD" <<'SQL'
ALTER ROLE simple_rag_owner PASSWORD :'owner_pw';
ALTER ROLE simple_rag_app PASSWORD :'app_pw';
SQL
