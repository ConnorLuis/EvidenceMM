from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class EvidenceCandidate:
    evidence_id: str
    domain: str
    score: float
    metadata: dict[str, Any]

class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceCandidate]:
        ...
