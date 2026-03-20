Database migrations managed by Alembic.

Common commands (run from project root):

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1

# Roll back everything
alembic downgrade base

# Create a new migration (autogenerate from model changes)
alembic revision --autogenerate -m "description"

# Show current migration state
alembic current

# Show migration history
alembic history
```

Migrations run automatically on app startup via `run_migrations()` in `database.py`.
