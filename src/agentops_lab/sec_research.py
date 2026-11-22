"""Build the existing ResearchRequest from bounded, topical SEC retrieval."""

from dataclasses import dataclass

from agentops_lab.filing_store import FilingStore
from agentops_lab.research_models import DocumentExcerpt, ResearchRequest
from agentops_lab.sec import COMPANY, TICKER


@dataclass(frozen=True)
class EvidenceTopic:
    query: str
    sections: tuple[str, ...]
    terms: tuple[str, ...]

    def matches(self, excerpt: DocumentExcerpt) -> bool:
        if excerpt.filing is None or excerpt.filing.section not in self.sections:
            return False
        text = excerpt.text.casefold()
        return any(term in text for term in self.terms)


EVIDENCE_TOPICS = (
    EvidenceTopic(
        "Business segments and revenue growth",
        ("Business", "MD&A"),
        ("segment", "revenue growth", "data center revenue", "compute & networking"),
    ),
    EvidenceTopic(
        "Supply chain suppliers or customer concentration",
        ("Business", "Risk Factors", "MD&A"),
        ("supply chain", "supplier", "customer concentration", "customers accounted"),
    ),
    EvidenceTopic(
        "Material business risks including export restrictions and competition",
        ("Risk Factors",),
        ("export", "competition", "competitive", "restriction"),
    ),
)


def build_sec_request(store: FilingStore, ticker: str) -> ResearchRequest:
    ticker = ticker.strip().upper()
    if ticker != TICKER:
        raise ValueError("The SEC pilot supports NVDA only.")
    excerpts: dict[str, DocumentExcerpt] = {}
    for topic in EVIDENCE_TOPICS:
        hits = store.search_filings(topic.query, ticker, top_k=3)
        matches = [hit.excerpt for hit in hits if topic.matches(hit.excerpt)]
        for excerpt in matches[:2]:
            excerpts.setdefault(excerpt.source_id, excerpt)
    return ResearchRequest(company=COMPANY, ticker=ticker, excerpts=list(excerpts.values()))
