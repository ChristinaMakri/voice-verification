"""Integration test for the Postgres store. Runs only when VV_TEST_DATABASE_URL is set
(CI sets it; locally: `docker compose up -d db` and export the URL from .env.example)."""

import os
import uuid

import pytest

DB_URL = os.environ.get("VV_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="VV_TEST_DATABASE_URL not set")


@pytest.fixture
def store():
    from voice_verification.storage import PostgresStore

    s = PostgresStore(DB_URL)
    yield s
    s.close()


def test_add_list_count_delete(store):
    user = f"test-{uuid.uuid4().hex}"
    store.add_many(user, "m1", [b"a", b"b"])
    store.add_many(user, "m2", [b"c"])
    assert store.list(user, "m1") == [b"a", b"b"]
    assert store.count(user, "m2") == 1
    assert store.delete(user) == 3
    assert store.count(user, "m1") == 0
