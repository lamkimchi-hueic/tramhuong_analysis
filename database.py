"""
Database connection module for AI Analysis service.
Connects to the same PostgreSQL (Neon) database used by the Node.js backend.
"""

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """Create and return a new database connection."""
    return psycopg2.connect(DATABASE_URL)


def query(sql: str, params: tuple = None) -> list[dict]:
    """
    Execute a SQL query and return results as a list of dicts.
    Uses RealDictCursor for dict-style row access.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()
