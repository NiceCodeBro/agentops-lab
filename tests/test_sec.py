import hashlib
import json
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from agentops_lab.retrieval import sec

# Fictional parser fixture. Filing identifiers exercise the pilot identity checks.
HTML = b"""<html><body>
<ix:header><ix:hidden>
  <ix:nonNumeric name="dei:EntityCentralIndexKey">0001045810</ix:nonNumeric>
  <ix:nonNumeric name="dei:DocumentPeriodEndDate">
    January&nbsp;25 <span>,</span> 2026
  </ix:nonNumeric>
  <p>Hidden XBRL noise.</p>
</ix:hidden></ix:header>
<ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>
<table><tr><td><div>Item 1. Business</div></td></tr></table>
<div><a href="#business">Item 1. Business</a></div>
<div id="business"><span>Item 1.</span> <span>Business</span></div>
<p>Our reportable segments serve different markets.</p>
<div>15</div><p>Table of Contents</p><script>script noise</script>
<div><span>Item 1A.</span> Risk Factors</div>
<p>A supplier outage could delay production. Export restrictions could reduce sales.</p>
<div>Item 1B. Unresolved Staff Comments</div><p>Outside the selected sections.</p>
<div>Item 7. Management's Discussion and Analysis</div>
<p>Revenue growth reflects demand for several products.</p>
<div>Item 7A. Market Risk</div><p>Also outside the selected sections.</p>
</body></html>"""


@pytest.fixture
def corpus():
    sections = sec.extract_sections(HTML)
    chunks = sec.create_chunks(sec.PILOT_FILING, sections)
    return sec.FilingCorpus(html_sha256=hashlib.sha256(HTML).hexdigest(), chunks=chunks)


def test_discovers_exact_filing_from_sec_submissions():
    submissions = {
        "cik": 1045810,
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-26-000022", sec.PILOT_ACCESSION],
                "form": ["10-K/A", "10-K"],
                "filingDate": ["2026-02-26", "2026-02-25"],
                "primaryDocument": ["amended.htm", "nvda-20260125.htm"],
            }
        },
    }
    raw = json.dumps(submissions).encode()
    assert sec.discover_filing(raw) == sec.PILOT_FILING
    with pytest.raises(ValueError, match="was not found"):
        sec.discover_filing(raw, "0001045810-26-000022")


def test_parses_narrative_and_preserves_provenance(corpus):
    sec.verify_pilot_html(HTML)
    with pytest.raises(ValueError, match="does not identify"):
        sec.verify_pilot_html(HTML.replace(b"0001045810", b"0000000001"))

    assert [chunk.text for chunk in corpus.chunks] == [
        "Our reportable segments serve different markets.",
        "A supplier outage could delay production. Export restrictions could reduce sales.",
        "Revenue growth reflects demand for several products.",
    ]
    sections = sec.extract_sections(HTML)
    assert sec.create_chunks(sec.PILOT_FILING, sections) == corpus.chunks
    assert len({chunk.source_id for chunk in corpus.chunks}) == 3

    risk = corpus.chunks[1]
    assert risk.source_id == "nvda-0001045810-26-000021-1a-001"
    assert risk.document_date.isoformat() == "2026-02-25"
    assert str(risk.url) == sec.PILOT_FILING.url
    assert risk.filing.model_dump() == {
        "ticker": "NVDA",
        "accession_number": sec.PILOT_ACCESSION,
        "document_type": "10-K",
        "section": "Risk Factors",
        "chunk_number": 1,
    }


def test_chunking_is_bounded_without_losing_words():
    paragraphs = ["A short opening. " + "long words " * 170, "Last paragraph."]
    chunks = sec.chunk_paragraphs(paragraphs, max_chars=100)
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(" ".join(paragraphs).split())


def test_sec_retries_rate_limits_with_identifying_header(monkeypatch):
    user_agent = "Example Research test@example.com"
    delays = []
    headers = Message()
    headers["Retry-After"] = "8"
    errors = [HTTPError(sec.SUBMISSIONS_URL, 429, "rate-limited", headers, None)]

    def request(req, *, timeout):
        assert timeout == 20
        assert req.get_header("User-agent") == user_agent
        if errors:
            raise errors.pop()
        return BytesIO(b"SEC payload")

    monkeypatch.setattr(sec.time, "sleep", delays.append)
    monkeypatch.setattr(sec, "urlopen", request)
    assert sec.sec_get(sec.SUBMISSIONS_URL, user_agent) == b"SEC payload"
    assert delays == [1, 8]


def test_sec_does_not_retry_forbidden_response(monkeypatch):
    delays = []

    def request(req, *, timeout):
        raise HTTPError(sec.SUBMISSIONS_URL, 403, "denied", Message(), None)

    monkeypatch.setattr(sec.time, "sleep", delays.append)
    monkeypatch.setattr(sec, "urlopen", request)
    with pytest.raises(HTTPError, match="403"):
        sec.sec_get(sec.SUBMISSIONS_URL, "Example Research test@example.com")
    assert delays == [1]
