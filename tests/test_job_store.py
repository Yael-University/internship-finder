"""
Tests for src.job_store._job_id.

Only the pure id helper is exercised here so the suite stays offline (no
ChromaDB client is constructed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.job_store import _job_id

_BOILERPLATE = (
    "About us: we are a fast-growing company that values diversity and "
    "inclusion. " * 4
)


def test_same_posting_is_stable():
    a = _job_id("Acme", "Data Engineer", _BOILERPLATE + "Build pipelines.", location="Remote")
    b = _job_id("Acme", "Data Engineer", _BOILERPLATE + "Build pipelines.", location="Remote")
    assert a == b


def test_shared_boilerplate_does_not_collide():
    # Same first 80 chars (the old hashed prefix) but different real content.
    a = _job_id("Acme", "Data Engineer", _BOILERPLATE + "Own the Spark stack.", location="Remote")
    b = _job_id("Acme", "Data Engineer", _BOILERPLATE + "Own the Kafka stack.", location="Remote")
    assert a != b


def test_location_distinguishes_postings():
    a = _job_id("Acme", "Data Engineer", _BOILERPLATE, location="Remote")
    b = _job_id("Acme", "Data Engineer", _BOILERPLATE, location="NYC")
    assert a != b


def test_link_takes_precedence_and_dedupes():
    # When a link is present, identity is the link regardless of description text.
    a = _job_id("Acme", "Data Engineer", "desc one", link="https://jobs.acme.com/123")
    b = _job_id("Acme", "Data Engineer", "desc two — totally different", link="https://jobs.acme.com/123")
    assert a == b


def test_distinct_links_do_not_collide():
    a = _job_id("Acme", "Data Engineer", _BOILERPLATE, link="https://jobs.acme.com/123")
    b = _job_id("Acme", "Data Engineer", _BOILERPLATE, link="https://jobs.acme.com/456")
    assert a != b
