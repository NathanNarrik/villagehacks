"""Pydantic models — both the public response shapes (frozen by frontend/src/types/api.ts)
and the internal types that flow between pipeline layers.

The internal types (ScribeWord, ScribeResult, WordWithConfidence, VerifyResult, Correction)
are imported by Person A's modules and Person B's modules so the contract is enforced at
type-check time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Public response shapes — must match frontend/src/types/api.ts EXACTLY
# ---------------------------------------------------------------------------

Speaker = Literal["Doctor", "Patient"]
ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]


class RawWord(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    speaker: Speaker
    confidence: ConfidenceLevel
    uncertainty_signals: list[str] | None = None


class CorrectedWord(BaseModel):
    word: str
    changed: bool
    tavily_verified: bool
    unverified: bool
    speaker: Speaker


class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    route: str
    tavily_verified: bool = False


class ClinicalSummary(BaseModel):
    medications: list[Medication] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    appointment_needed: bool = False


class PipelineLatency(BaseModel):
    preprocessing: int
    scribe: int
    uncertainty: int
    tavily: int
    claude: int
    total: int


class TranscribeResponse(BaseModel):
    raw_transcript: list[RawWord]
    corrected_transcript: list[CorrectedWord]
    clinical_summary: ClinicalSummary
    pipeline_latency_ms: PipelineLatency


class HealthResponse(BaseModel):
    status: str
    redis: str
    scribe: str
    tavily: str
    claude: str


# Benchmark response — passed through from cached JSON, validated loosely
class AblationRow(BaseModel):
    stage: str
    wer: float
    delta: float
    description: str


class BenchmarkClipResult(BaseModel):
    clip_id: str
    category: str
    difficulty: Literal["Standard", "Adversarial"]
    raw_wer: float
    corrected_wer: float
    improvement_pct: float


class BenchmarkMetrics(BaseModel):
    verification_rate: float
    unsafe_guess_rate: float
    uncertainty_coverage: float
    phonetic_hit_rate: float


class BenchmarkAggregate(BaseModel):
    avg_raw_wer: float
    avg_corrected_wer: float
    avg_improvement_pct: float
    keyterm_impact_pct: float


class BenchmarkResponse(BaseModel):
    results: list[BenchmarkClipResult]
    ablation: list[AblationRow]
    metrics: BenchmarkMetrics
    aggregate: BenchmarkAggregate


# ---------------------------------------------------------------------------
# Internal pipeline dataclasses — flow between layers, not serialized to client
# ---------------------------------------------------------------------------


@dataclass
class ScribeWord:
    """One word from ElevenLabs Scribe v2 batch transcription. Person A returns these."""

    text: str
    start_ms: int
    end_ms: int
    speaker_id: str  # "speaker_0" / "speaker_1" — Person C maps to Doctor/Patient


@dataclass
class ScribeResult:
    """Output of Person A's `scribe.transcribe_batch()`."""

    words: list[ScribeWord]
    audio_events: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class WordWithConfidence:
    """A scored word emitted by Person A's `uncertainty.score_words()`."""

    word: str
    start_ms: int
    end_ms: int
    speaker_id: str
    confidence: ConfidenceLevel
    uncertainty_signals: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    """Result of a single Tavily verification call."""

    original: str
    status: Literal["VERIFIED", "UNVERIFIED"]
    canonical: str | None = None
    source_url: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class Correction:
    """Internal correction record produced by Claude + applied by the pipeline."""

    index: int
    original: str
    corrected: str
    tavily_verified: bool
    unverified: bool
