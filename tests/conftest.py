"""Test fixtures.

Tests run against a real Postgres database (not SQLite) so the check
constraints, the Uuid column type, and the Alembic-managed schema all behave
exactly as they do in production. Each test runs inside its own transaction
that is rolled back afterward (via a SAVEPOINT-per-session pattern), so
tests never see each other's data and never need manual cleanup.

TEST_DATABASE_URL defaults to a local Postgres instance and is auto-created
if it doesn't exist yet; see README.md for how to start one locally, and
.github/workflows/ci.yml for how CI provides one.
"""

import os

import asyncpg
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import Base, get_db
from app.dependencies import get_llm_client
from app.main import app
from tests.fakes import FakeLLMClient

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_support_test",
)


async def _ensure_database_exists(url: str) -> None:
    parsed = make_url(url)
    dbname = parsed.database
    admin_dsn = parsed.set(database="postgres").render_as_string(hide_password=False)
    admin_dsn = admin_dsn.replace("postgresql+asyncpg", "postgresql")

    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", dbname)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        await conn.close()


@pytest_asyncio.fixture(scope="session")
async def engine():
    await _ensure_database_exists(TEST_DATABASE_URL)
    eng = create_async_engine(TEST_DATABASE_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine):
    connection = await engine.connect()
    trans = await connection.begin()
    session_factory = async_sessionmaker(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )
    async with session_factory() as session:
        yield session
    await trans.rollback()
    await connection.close()


@pytest_asyncio.fixture
def fake_llm():
    return FakeLLMClient()


@pytest_asyncio.fixture
async def client(db_session, fake_llm):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_llm_client, None)
