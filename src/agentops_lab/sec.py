"""Download and chunk the narrative sections of an NVIDIA Form 10-K."""

import json
import re
import textwrap
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from agentops_lab.research_models import DocumentExcerpt, FilingProvenance

TICKER = "NVDA"
COMPANY = "NVIDIA Corporation"
CIK = "0001045810"
PILOT_ACCESSION = "0001045810-26-000021"
PILOT_FILING_DATE = date(2026, 2, 25)
PILOT_REPORT_DATE = date(2026, 1, 25)
PILOT_DOCUMENT = "nvda-20260125.htm"
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
CHUNK_CHARACTERS = 2400
SECTIONS = {"1": "Business", "1A": "Risk Factors", "7": "MD&A"}
ITEM_HEADING = re.compile(r"^item\s+(\d+[a-z]?)\s*(?:[.:\-]\s*|$)", re.IGNORECASE)
BLOCKS = ("div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6")


@dataclass(frozen=True)
class Filing:
    accession_number: str
    filing_date: date
    document_name: str

    @property
    def url(self) -> str:
        accession = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{accession}/{self.document_name}"
        )


PILOT_FILING = Filing(PILOT_ACCESSION, PILOT_FILING_DATE, PILOT_DOCUMENT)


class FilingCorpus(BaseModel):
    """Full extracted chunk text and provenance, stored together as one S3 object."""

    html_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: list[DocumentExcerpt] = Field(min_length=1)


def sec_get(url: str, user_agent: str) -> bytes:
    user_agent = user_agent.strip()
    if "@" not in user_agent or "your-" in user_agent.casefold():
        raise ValueError("Set SEC_USER_AGENT to your project or organization and contact email.")

    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    delay = 1.0
    for attempt in range(3):
        time.sleep(delay)
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read(MAX_DOCUMENT_BYTES + 1)
            if not body or len(body) > MAX_DOCUMENT_BYTES:
                raise ValueError("SEC response was empty or exceeded 16 MiB.")
            return body
        except HTTPError as error:
            if attempt == 2 or error.code not in {429, 500, 502, 503, 504}:
                raise
            delay = 2.0 ** (attempt + 1)
            retry_after = error.headers.get("Retry-After")
            if retry_after:
                if retry_after.isdecimal():
                    wait = float(retry_after)
                else:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                    except ValueError:
                        retry_at = datetime.now(UTC)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    wait = (retry_at - datetime.now(UTC)).total_seconds()
                if wait > 60:
                    raise
                delay = max(delay, wait)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            delay = 2.0 ** (attempt + 1)
    raise RuntimeError("SEC request exhausted its retry attempts.")


def discover_filing(submissions: bytes, accession_number: str = PILOT_ACCESSION) -> Filing:
    data = json.loads(submissions)
    if str(data.get("cik", "")).zfill(10) != CIK or TICKER not in data.get("tickers", []):
        raise ValueError("SEC submissions did not identify NVIDIA (NVDA).")

    recent = data["filings"]["recent"]
    rows = zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["primaryDocument"],
        strict=True,
    )
    for accession, form, filed, document in rows:
        if accession != accession_number or form != "10-K":
            continue
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(
            r"[A-Za-z0-9_-]+\.html?", document
        ):
            raise ValueError("SEC returned an invalid accession number or primary document name.")
        return Filing(accession, date.fromisoformat(filed), document)
    raise ValueError(
        f"NVIDIA Form 10-K {accession_number} was not found in recent SEC submissions."
    )


def extract_sections(html: bytes) -> dict[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("head, script, style, nav, table, ix\\:header, ix\\:hidden, [hidden]"):
        tag.decompose()

    sections: dict[str, list[str]] = {}
    item = ""
    paragraphs: list[str] = []
    for block in soup.find_all(BLOCKS):
        if block.find(BLOCKS):
            continue
        style = str(block.get("style", "")).replace(" ", "").lower()
        if "display:none" in style:
            continue
        text = " ".join(block.get_text(" ", strip=True).split())
        if not text or text.isdecimal() or text.casefold() == "table of contents":
            continue

        heading = ITEM_HEADING.match(text) if len(text) <= 120 else None
        if heading and not block.find("a", href=True):
            if item in SECTIONS and sum(map(len, paragraphs)) > sum(
                map(len, sections.get(item, []))
            ):
                sections[item] = paragraphs
            item = heading[1].upper()
            paragraphs = []
        elif item in SECTIONS:
            paragraphs.append(text)

    if item in SECTIONS and sum(map(len, paragraphs)) > sum(map(len, sections.get(item, []))):
        sections[item] = paragraphs
    missing = SECTIONS.keys() - sections.keys()
    if missing:
        labels = ", ".join(SECTIONS[item] for item in SECTIONS if item in missing)
        raise ValueError(f"Expected 10-K narrative sections could not be extracted: {labels}.")
    return sections


def verify_pilot_html(html: bytes) -> None:
    soup = BeautifulSoup(html, "html.parser")
    fields = {
        str(tag.get("name", "")).casefold(): " ".join(tag.get_text(" ", strip=True).split())
        for tag in soup.find_all("ix:nonnumeric")
    }
    cik = fields.get("dei:entitycentralindexkey", "").zfill(10)
    document_type = fields.get("dei:documenttype", "")
    period = fields.get("dei:documentperiodenddate", "")
    expected_periods = {
        PILOT_REPORT_DATE.isoformat(),
        PILOT_REPORT_DATE.strftime("%B %d, %Y"),
        PILOT_REPORT_DATE.strftime("%b %d, %Y"),
    }
    if cik != CIK or document_type != "10-K" or period not in expected_periods:
        raise ValueError("HTML does not identify the selected NVIDIA Form 10-K and report period.")


def chunk_paragraphs(paragraphs: list[str], max_chars: int = CHUNK_CHARACTERS) -> list[str]:
    chunks = []
    current = ""
    for paragraph in paragraphs:
        parts = [paragraph]
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
            parts = [part for sentence in sentences for part in textwrap.wrap(sentence, max_chars)]
        for part in parts:
            if current and len(current) + len(part) + 2 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        chunks.append(current)
    return chunks


def create_chunks(filing: Filing, sections: dict[str, list[str]]) -> list[DocumentExcerpt]:
    excerpts = []
    for item, section in SECTIONS.items():
        texts = chunk_paragraphs(sections[item])
        for number, text in enumerate(texts, start=1):
            provenance = FilingProvenance(
                ticker=TICKER,
                accession_number=filing.accession_number,
                document_type="10-K",
                section=section,
                chunk_number=number,
            )
            excerpts.append(
                DocumentExcerpt(
                    source_id=f"nvda-{filing.accession_number}-{item.lower()}-{number:03}",
                    title=f"{COMPANY} Form 10-K, Item {item}. {section}",
                    document_date=filing.filing_date,
                    url=filing.url,
                    text=text,
                    filing=provenance,
                )
            )
    return excerpts
