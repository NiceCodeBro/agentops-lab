"""Inputs and outputs for evidence-based company research."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

type NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# Identifies the filing and position of an excerpt selected by SEC retrieval.
class FilingProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    ticker: NonEmptyText
    accession_number: NonEmptyText
    document_type: Literal["10-K"]
    section: NonEmptyText
    chunk_number: int = Field(gt=0)


# Identifies a source in the final brief by ID, title, date, and optional URL.
class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    source_id: NonEmptyText
    title: NonEmptyText
    document_date: date
    url: HttpUrl | None = None
    filing: FilingProvenance | None = Field(default=None, exclude_if=lambda value: value is None)


# Adds the actual evidence text to the source metadata for the research input.
class DocumentExcerpt(SourceReference):
    text: NonEmptyText


# Holds the company, ticker, and excerpts supplied for one research run.
class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    company: NonEmptyText
    ticker: NonEmptyText
    excerpts: list[DocumentExcerpt]
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_unique_sources(self) -> Self:
        source_ids = {excerpt.source_id for excerpt in self.excerpts}
        if len(source_ids) != len(self.excerpts):
            raise ValueError("Each document excerpt must have a unique source_id.")
        return self


# Represents one finding, opportunity, or risk with at least one source ID.
class CitedClaim(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    text: NonEmptyText = Field(description="One claim or inference supported by the excerpts.")
    source_ids: list[NonEmptyText] = Field(min_length=1, description="Supporting input source IDs.")


# A potential opportunity must be anchored to a passage in its cited evidence.
class Opportunity(CitedClaim):
    evidence_quote: NonEmptyText | None = Field(
        default=None,
        description="A verbatim passage from a cited excerpt that directly supports the inference.",
    )


# Defines the structured response that the Bedrock model is asked to produce.
class ResearchAnalysis(BaseModel):
    """Evidence-based findings and gaps from supplied financial excerpts."""

    model_config = ConfigDict(hide_input_in_errors=True)

    key_findings: list[CitedClaim] = Field(
        description="Material reported results and business drivers stated in the excerpts."
    )
    opportunities: list[Opportunity] = Field(
        description="Conditional opportunities directly grounded in a cited verbatim passage."
    )
    risks: list[CitedClaim] = Field(
        description="Business or investment exposures explicitly stated or supported by evidence."
    )
    missing_information: list[NonEmptyText] = Field(
        description="Questions or evidence gaps only, without financial assertions."
    )


# Combines the validated analysis with company details and cited source metadata.
class ResearchBrief(ResearchAnalysis):
    company: NonEmptyText
    ticker: NonEmptyText
    synthetic: bool
    sources: list[SourceReference]
    review_notes: list[str] = Field(default_factory=list, exclude_if=lambda notes: not notes)
