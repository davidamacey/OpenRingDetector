-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Tables are created by SQLAlchemy on app startup.
-- This script just ensures the pgvector extension is ready.
