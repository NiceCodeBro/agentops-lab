"""Manual ingestion and a bounded retrieval smoke test for the NVIDIA SEC pilot."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.error import HTTPError

from botocore.exceptions import BotoCoreError, ClientError

from agentops_lab.filing_store import load_filing_store
from agentops_lab.sec import (
    MAX_DOCUMENT_BYTES,
    PILOT_FILING,
    SUBMISSIONS_URL,
    TICKER,
    FilingCorpus,
    create_chunks,
    discover_filing,
    extract_sections,
    sec_get,
    verify_pilot_html,
)
from agentops_lab.sec_research import EVIDENCE_TOPICS


def ingest() -> int:
    parser = argparse.ArgumentParser(description="Ingest one verified NVIDIA Form 10-K.")
    parser.add_argument("--html", type=Path, help="Local copy of the pilot primary filing HTML")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count chunks locally")
    args = parser.parse_args()

    try:
        if args.html:
            with args.html.open("rb") as source:
                html = source.read(MAX_DOCUMENT_BYTES + 1)
            if len(html) > MAX_DOCUMENT_BYTES:
                raise ValueError("Local filing HTML exceeds 16 MiB.")
            filing = PILOT_FILING
        else:
            user_agent = os.getenv("SEC_USER_AGENT", "")
            submissions = sec_get(SUBMISSIONS_URL, user_agent)
            filing = discover_filing(submissions)
            html = sec_get(filing.url, user_agent)

        verify_pilot_html(html)
        sections = extract_sections(html)
        chunks = create_chunks(filing, sections)
        corpus = FilingCorpus(html_sha256=hashlib.sha256(html).hexdigest(), chunks=chunks)
        print(f"{filing.accession_number} filed {filing.filing_date} | {filing.url}")
        print(f"Prepared {len(chunks)} chunks | Filing SHA-256 {corpus.html_sha256}")
        if args.dry_run:
            return 0

        store = load_filing_store()
        count = store.ingest(corpus)
        print(f"Upserted {count} vectors.")
    except HTTPError as error:
        parser.exit(
            1, f"SEC HTTP {error.code}. Retry later or use --html with the primary filing.\n"
        )
    except (BotoCoreError, ClientError, OSError, ValueError) as error:
        parser.exit(1, f"Ingestion failed: {error}\n")
    return 0


def smoke() -> int:
    parser = argparse.ArgumentParser(description="Check NVIDIA SEC retrieval topics.")
    parser.add_argument("--show-text", action="store_true", help="Print complete retrieved chunks")
    args = parser.parse_args()

    passed = 0
    try:
        store = load_filing_store()
        for topic in EVIDENCE_TOPICS:
            hits = store.search_filings(topic.query, TICKER, top_k=3)
            matched = any(topic.matches(hit.excerpt) for hit in hits)
            passed += int(matched)
            print(f"\n{'PASS' if matched else 'MISS'} | {topic.query}")
            for hit in hits:
                excerpt = hit.excerpt
                print(
                    f"  {excerpt.source_id} | {excerpt.document_date} | distance {hit.distance:.3f}"
                )
                print(f"  {excerpt.title} | {excerpt.url}")
                print(f"  {excerpt.text if args.show_text else excerpt.text[:260]}")
    except (BotoCoreError, ClientError, OSError, ValueError) as error:
        parser.exit(1, f"Retrieval failed: {error}\n")
    print(f"\n{passed}/{len(EVIDENCE_TOPICS)} queries returned a topical excerpt.")
    return int(passed != len(EVIDENCE_TOPICS))
