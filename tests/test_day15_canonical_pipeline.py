import pytest

from evidencemm.canonical_pipeline import (
    validate_document_mode,
)


def test_validate_document_mode_hybrid():
    assert (
        validate_document_mode("hybrid")
        == "hybrid"
    )


def test_validate_document_mode_bm25():
    assert (
        validate_document_mode("BM25")
        == "bm25"
    )


def test_validate_document_mode_rejects_unknown():
    with pytest.raises(
        ValueError,
        match="document_mode",
    ):
        validate_document_mode(
            "colqwen"
        )
