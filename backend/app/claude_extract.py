"""Layer 6 — Clinical entity extraction with Claude.

Second Claude call on the corrected transcript. Returns a structured ClinicalSummary
matching the frontend schema. Each medication's `tavily_verified` flag is overwritten
post-hoc using the verifications dict from Layer 4 — Claude isn't trusted to set it.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from .config import settings
from .schemas import ClinicalSummary, CorrectedWord, Medication, VerifyResult

log = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You extract clinical entities from a medical phone-call transcript.

Return JSON ONLY (no prose, no markdown fences) matching this schema exactly:
{
  "medications": [{"name": str, "dosage": str, "frequency": str, "route": str, "tavily_verified": false}],
  "symptoms": [str],
  "allergies": [str],
  "follow_up_actions": [str],
  "appointment_needed": bool
}

Rules:
- Use "Unknown" (literal string) for missing dosage/frequency/route fields.
- Always set tavily_verified=false here; the backend overwrites it.
- "appointment_needed" is true if the patient mentions new or worsening symptoms, or the doctor schedules a follow-up.
- Symptoms are short noun phrases (e.g. "headaches", "dizziness"). No sentences.
- Allergies must be explicitly stated as allergies — do not infer.
"""


class ClaudeExtractor:
    def __init__(self, api_key: str, model: str = settings.CLAUDE_MODEL):
        self._client = AsyncAnthropic(api_key=api_key) if api_key else None
        self._model = model

    async def extract(
        self,
        corrected: list[CorrectedWord],
        verifications: dict[str, VerifyResult],
    ) -> ClinicalSummary:
        text = self._render_transcript(corrected)

        if self._client is None:
            log.warning("Claude client not configured — returning empty ClinicalSummary")
            return ClinicalSummary()

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            payload = self._parse_json(resp.content[0].text)
            summary = ClinicalSummary.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("Claude extraction failed: %s — returning empty summary", exc)
            return ClinicalSummary()

        verified_canonicals = {
            v.canonical.lower() for v in verifications.values() if v.canonical
        }
        for med in summary.medications:
            med.tavily_verified = med.name.lower() in verified_canonicals

        return summary

    @staticmethod
    def _render_transcript(corrected: list[CorrectedWord]) -> str:
        lines: list[str] = []
        current_speaker: str | None = None
        buffer: list[str] = []
        for cw in corrected:
            if cw.speaker != current_speaker:
                if buffer:
                    lines.append(f"{current_speaker}: {' '.join(buffer)}")
                buffer = []
                current_speaker = cw.speaker
            buffer.append(cw.word)
        if buffer:
            lines.append(f"{current_speaker}: {' '.join(buffer)}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)


_extractor: ClaudeExtractor | None = None


def get_extractor() -> ClaudeExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClaudeExtractor(
            api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
            model=settings.CLAUDE_MODEL,
        )
    return _extractor


def reset_extractor() -> None:
    global _extractor
    _extractor = None
