"""Build the existing ResearchRequest from bounded, topical SEC retrieval."""

from dataclasses import dataclass
from itertools import zip_longest

from agentops_lab.models.research import DocumentExcerpt, ResearchRequest
from agentops_lab.retrieval.filing_store import FilingStore
from agentops_lab.retrieval.sec import COMPANY, TICKER


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
        "Supply chain constraints, manufacturing lead times, foundries and production capacity",
        ("Business", "Risk Factors"),
        ("supply chain", "supplier", "manufacturing lead times", "capacity commitments"),
    ),
    EvidenceTopic(
        "Customer concentration: share of revenue from individual direct and indirect customers",
        ("Risk Factors", "MD&A"),
        (
            "concentration of revenue",
            "revenue is concentrated",
            "concentration of sales",
            "customer concentration",
            "sales to one direct customer",
            "limited number of customers",
        ),
    ),
    EvidenceTopic(
        "Material export control risks: trade restrictions, product licensing and sanctions",
        ("Risk Factors",),
        ("export controls", "export restrictions", "export license", "trade restrictions"),
    ),
)


def build_sec_request(store: FilingStore, ticker: str) -> ResearchRequest:
    ticker = ticker.strip().upper()
    if ticker != TICKER:
        raise ValueError("The SEC pilot supports NVDA only.")

    queries = [topic.query for topic in EVIDENCE_TOPICS]
    hits_by_query = store.search_filings_batch(queries, ticker, top_k=3)
    ranked_topics = []
    for topic, hits in zip(EVIDENCE_TOPICS, hits_by_query, strict=True):
        matches = [hit.excerpt for hit in hits if topic.matches(hit.excerpt)]
        ranked_topics.append(matches)

    excerpts = select_distinct_excerpts(ranked_topics, limit=6)
    return ResearchRequest(company=COMPANY, ticker=ticker, excerpts=excerpts)


def select_distinct_excerpts(
    ranked_topics: list[list[DocumentExcerpt]], *, limit: int, excluded: set[str] | None = None
) -> list[DocumentExcerpt]:
    """Interleave query ranks so a single topic cannot crowd out other topics."""
    if limit <= 0:
        return []
    excerpts: dict[str, DocumentExcerpt] = {}
    for rank in zip_longest(*ranked_topics):
        for excerpt in rank:
            if excerpt is None or (excluded is not None and excerpt.source_id in excluded):
                continue
            excerpts.setdefault(excerpt.source_id, excerpt)
            if len(excerpts) == limit:
                return list(excerpts.values())
    return list(excerpts.values())
