from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN = re.compile(
    r"[a-z0-9]+(?:[._/+\-][a-z0-9]+)*",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def tokenize_mixed(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    tokens = _LATIN_TOKEN.findall(normalized)

    for match in _CJK_RUN.finditer(normalized):
        run = match.group(0)
        chars = list(run)
        tokens.extend(chars)
        tokens.extend(
            run[i : i + 2]
            for i in range(len(run) - 1)
        )

    return tokens


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PageDocument:
    source_id: str
    page_number: int
    text: str
    text_sha256: str
    char_count: int

    @classmethod
    def from_text(
        cls,
        *,
        source_id: str,
        page_number: int,
        text: str,
    ) -> "PageDocument":
        normalized = normalize_text(text)
        return cls(
            source_id=source_id,
            page_number=page_number,
            text=normalized,
            text_sha256=sha256_text(normalized),
            char_count=len(normalized),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    source_id: str
    page_number: int
    text_preview: str

    def to_dict(self) -> dict:
        return asdict(self)


class BM25Index:
    def __init__(
        self,
        documents: list[PageDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25Index requires at least one document")

        self.documents = documents
        self.k1 = k1
        self.b = b

        self._tokens = [
            tokenize_mixed(doc.text)
            for doc in documents
        ]
        self._term_freqs = [
            Counter(tokens)
            for tokens in self._tokens
        ]
        self._doc_lengths = [
            len(tokens)
            for tokens in self._tokens
        ]
        self._avgdl = (
            sum(self._doc_lengths)
            / len(self._doc_lengths)
        )

        doc_freq: Counter[str] = Counter()
        for tokens in self._tokens:
            doc_freq.update(set(tokens))
        self._doc_freq = doc_freq

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._doc_freq.get(term, 0)
        return math.log(
            1.0
            + (n - df + 0.5)
            / (df + 0.5)
        )

    def score(self, query: str) -> list[float]:
        query_terms = tokenize_mixed(query)
        scores: list[float] = []

        for tf, dl in zip(
            self._term_freqs,
            self._doc_lengths,
        ):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue

                numerator = (
                    freq
                    * (self.k1 + 1.0)
                )
                denominator = (
                    freq
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * dl
                        / self._avgdl
                    )
                )
                score += (
                    self._idf(term)
                    * numerator
                    / denominator
                )
            scores.append(score)

        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        scores = self.score(query)
        order = sorted(
            range(len(scores)),
            key=lambda i: (
                -scores[i],
                self.documents[i].source_id,
                self.documents[i].page_number,
            ),
        )[:top_k]

        hits: list[RetrievalHit] = []
        for rank, idx in enumerate(order, start=1):
            doc = self.documents[idx]
            preview = doc.text[:240]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=round(scores[idx], 6),
                    source_id=doc.source_id,
                    page_number=doc.page_number,
                    text_preview=preview,
                )
            )

        return hits


def save_corpus(
    documents: Iterable[PageDocument],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        json.dumps(
            doc.to_dict(),
            ensure_ascii=False,
        )
        for doc in documents
    ]
    output.write_text(
        "\n".join(rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
        newline="\n",
    )
    return output


def load_corpus(
    path: str | Path,
) -> list[PageDocument]:
    input_path = Path(path)
    return [
        PageDocument(**json.loads(line))
        for line in input_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
