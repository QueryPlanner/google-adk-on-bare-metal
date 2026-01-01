#!/bin/sh
set -e

# Wait for the database if DATABASE_URL is set and looks like a postgres url
if echo "$DATABASE_URL" | grep -q "postgresql://"; then
    echo "Waiting for database..."
    # Extract host and port from DATABASE_URL
    # Assumes format postgresql://user:pass@host:port/dbname
    # This is a basic extraction and might need adjustment for complex URLs
    DB_HOST=$(echo $DATABASE_URL | sed -e 's|^.*@||' -e 's|/.*$||' -e 's|:.*$||')
    DB_PORT=$(echo $DATABASE_URL | sed -e 's|^.*@||' -e 's|/.*$||' -e 's|^.*:||')
    
    # Default port if not specified
    if [ "$DB_HOST" = "$DB_PORT" ]; then
        DB_PORT=5432
    fi

    # Loop until the database is ready
    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 1
    done
    echo "Database started"
fi

exec "$@"