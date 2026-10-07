import time
from datetime import UTC, datetime, timedelta

import pytest

from app.domain.exceptions import DomainValidationError, InvalidStateTransitionError
from app.domain.models import ApiKey, Collection, Document, DocumentStatus, DocumentType
from app.domain.models.api_key import KEY_PREFIX, hash_api_key
from app.domain.value_objects import CollectionName, ContentHash, EmbeddingSpec, new_id

EMBEDDING = EmbeddingSpec(model="qwen3-embedding:8b", dimensions=1024)


# --- Identifiers --------------------------------------------------------------
def test_new_id_is_uuid_version_7_and_time_ordered() -> None:
    first = new_id()
    time.sleep(0.002)
    second = new_id()

    assert first.version == 7
    assert first.variant == "specified in RFC 4122"
    assert first < second  # millisecond timestamp prefix sorts by creation time


# --- Value objects ------------------------------------------------------------
@pytest.mark.parametrize("raw", ["kubernetes-docs", "my_notes", "abc", "  Recipes  "])
def test_collection_name_accepts_and_normalises(raw: str) -> None:
    assert CollectionName.parse(raw).value == raw.strip().lower()


@pytest.mark.parametrize("raw", ["ab", "-leading-dash", "has space", "x" * 65, "ümlaut"])
def test_collection_name_rejects_invalid(raw: str) -> None:
    with pytest.raises(DomainValidationError):
        CollectionName.parse(raw)


def test_content_hash_of_bytes() -> None:
    assert ContentHash.of(b"hello").value == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_content_hash_rejects_non_sha256() -> None:
    with pytest.raises(DomainValidationError):
        ContentHash("not-a-hash")


def test_embedding_spec_validates() -> None:
    with pytest.raises(DomainValidationError):
        EmbeddingSpec(model="m", dimensions=0)


# --- Collection ---------------------------------------------------------------
def test_collection_create_blank_description_becomes_none() -> None:
    collection = Collection.create(
        name=CollectionName("docs"), embedding=EMBEDDING, description="   "
    )
    assert collection.description is None


def test_entities_compare_by_identity() -> None:
    a = Collection.create(name=CollectionName("docs"), embedding=EMBEDDING)
    b = Collection.create(name=CollectionName("docs"), embedding=EMBEDDING)
    assert a != b
    a_renamed = Collection(id=a.id, name=CollectionName("other"), embedding=EMBEDDING)
    assert a == a_renamed


# --- Document lifecycle -------------------------------------------------------
def _document() -> Document:
    return Document(
        collection_id=new_id(),
        title="Guide",
        source_filename="guide.md",
        document_type=DocumentType.MARKDOWN,
        content_hash=ContentHash.of(b"# Guide"),
        size_bytes=7,
        storage_key="abc/guide.md",
    )


def test_document_happy_path() -> None:
    doc = _document()
    doc.start_processing()
    doc.mark_ready(chunk_count=12)
    assert doc.status is DocumentStatus.READY
    assert doc.chunk_count == 12


def test_document_failure_then_retry_clears_error() -> None:
    doc = _document()
    doc.start_processing()
    doc.mark_failed("embedding service unavailable")
    assert doc.error == "embedding service unavailable"

    doc.requeue()
    doc.start_processing()
    assert doc.status is DocumentStatus.PROCESSING
    assert doc.error is None


@pytest.mark.parametrize(
    "illegal",
    [
        lambda d: d.mark_ready(1),  # pending -> ready skips processing
        lambda d: d.mark_failed("x"),  # pending -> failed
    ],
)
def test_document_rejects_illegal_transitions(illegal: object) -> None:
    doc = _document()
    with pytest.raises(InvalidStateTransitionError):
        illegal(doc)  # type: ignore[operator]


# --- API keys -----------------------------------------------------------------
def test_issue_api_key_stores_only_the_hash() -> None:
    key, raw = ApiKey.issue("dev")

    assert raw.startswith(KEY_PREFIX)
    assert len(raw) > 40
    assert key.key_hash == hash_api_key(raw)
    assert raw not in repr(key)
    assert raw.startswith(key.display_prefix)


def test_api_key_record_use_is_throttled() -> None:
    key, _ = ApiKey.issue("dev")
    now = datetime.now(UTC)

    assert key.record_use(now) is True
    assert key.record_use(now + timedelta(seconds=10)) is False
    assert key.record_use(now + timedelta(minutes=2)) is True


def test_revoked_key_is_inactive() -> None:
    key, _ = ApiKey.issue("dev")
    key.revoke()
    assert not key.is_active
