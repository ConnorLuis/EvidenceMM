from __future__ import annotations

from evidencemm.text_retrieval import (
    BM25Index,
    PageDocument,
    normalize_text,
    tokenize_mixed,
)


def test_normalize_text_nfkc_and_whitespace():
    assert normalize_text(
        " Voltage\n  ７.４V "
    ) == "Voltage 7.4V"


def test_mixed_tokenizer_contains_english_and_cjk_bigrams():
    tokens = tokenize_mixed(
        "Feedback 反馈状态"
    )

    assert "feedback" in tokens
    assert "反馈" in tokens
    assert "状态" in tokens


def test_bm25_ranks_matching_page_first():
    documents = [
        PageDocument.from_text(
            source_id="manual",
            page_number=1,
            text="包装与尺寸说明",
        ),
        PageDocument.from_text(
            source_id="manual",
            page_number=2,
            text=(
                "典型工作电压 Operating Voltage "
                "6V 7.4V"
            ),
        ),
    ]

    hits = BM25Index(
        documents
    ).search(
        "典型工作电压有哪些",
        top_k=2,
    )

    assert hits[0].page_number == 2


def test_page_document_uses_one_based_page_number():
    document = PageDocument.from_text(
        source_id="manual",
        page_number=3,
        text="Feedback",
    )

    assert document.page_number == 3
    assert document.char_count > 0
    assert len(document.text_sha256) == 64
