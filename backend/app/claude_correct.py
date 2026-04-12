"""Layer 5 — Safe correction with Claude.

Hard rules (enforced by both the system prompt and a post-hoc hallucination guard):
1. Only correct flagged words that have a Tavily VERIFIED match.
2. Never guess. Never invent medications.
3. For flagged words without verification, keep the original and set unverified=True.
4. Doctor speech is drug-dense; patient speech is symptom-dense.
5. Use surrounding dosage context (e.g. "500 mg twice daily") to confirm drug names.

The hallucination guard runs after Claude returns: any "changed" word that isn't
backed by a tavily_verified=True flag is reverted to the original. This is what
powers the "0% unsafe guess rate" headline metric.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any

from anthropic import AsyncAnthropic

from .config import settings
from .medical_patterns import normalize
from .schemas import CorrectedWord, VerifyResult, WordWithConfidence

log = logging.getLogger(__name__)

CORRECTION_SYSTEM_PROMPT = """You are a medical transcript safety reviewer. You receive a Speech-To-Text transcript with words flagged as uncertain, plus a dictionary of Tavily-verified canonical drug names.

ABSOLUTE RULES — violating any of these is a critical patient-safety failure:
1. ONLY change a flagged word if Tavily confirmed a canonical replacement for it in the verifications dictionary below.
2. NEVER guess a correction. NEVER invent or suggest a medication name that is not in the verifications dictionary.
3. For flagged words with no Tavily confirmation, KEEP the original text exactly and mark unverified=true.
4. Doctor speech is drug-dense; patient speech is symptom-dense. Use this context to disambiguate.
5. Use surrounding dosage context (e.g. "500 mg twice daily" after a candidate drug name) to confirm.

Return JSON ONLY (no prose, no markdown fences) matching this schema exactly:
{"corrections": [{"index": int, "corrected": str, "tavily_verified": bool, "unverified": bool}]}

The "index" is the integer index into the input words array. Include an entry for every word that was flagged in the input — even if you decide to keep it unchanged. For unchanged words, set corrected to the original word and tavily_verified=false, unverified=false.
"""


class ClaudeCorrector:
    def __init__(self, api_key: str, model: str = settings.CLAUDE_MODEL):
        self._client = AsyncAnthropic(api_key=api_key) if api_key else None
        self._model = model

    async def correct(
        self,
        raw_words: list[WordWithConfidence],
        verifications: dict[str, VerifyResult],
        speakers: list[str],
    ) -> list[CorrectedWord]:
        """Run safe correction. `speakers[i]` is the resolved Doctor/Patient label for raw_words[i]."""
        if self._client is None:
            log.warning("Claude client not configured — returning identity corrections")
            return self._identity_corrections(raw_words, speakers)

        flagged_indices = [
            i for i, w in enumerate(raw_words) if w.confidence in ("LOW", "MEDIUM")
        ]

        # If nothing was flagged, skip Claude entirely.
        if not flagged_indices:
            return self._identity_corrections(raw_words, speakers)

        prompt = self._build_prompt(raw_words, verifications, speakers, flagged_indices)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                temperature=0.0,
                system=CORRECTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            corrections_payload = self._parse_json(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Claude correction failed: %s — falling back to identity", exc)
            return self._identity_corrections(raw_words, speakers)

        return self._apply_and_guard(
            raw_words, speakers, corrections_payload, verifications
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        raw_words: list[WordWithConfidence],
        verifications: dict[str, VerifyResult],
        speakers: list[str],
        flagged_indices: list[int],
    ) -> str:
        annotated_words = []
        for i, w in enumerate(raw_words):
            tag = "FLAGGED" if i in flagged_indices else "ok"
            annotated_words.append(
                {
                    "index": i,
                    "text": w.word,
                    "speaker": speakers[i],
                    "status": tag,
                }
            )

        verifications_payload = {
            term: {
                "status": v.status,
                "canonical": v.canonical,
                "source_url": v.source_url,
            }
            for term, v in verifications.items()
        }

        return (
            "WORDS:\n"
            + json.dumps(annotated_words, indent=2)
            + "\n\nVERIFICATIONS (only these canonical replacements may be used):\n"
            + json.dumps(verifications_payload, indent=2)
            + "\n\nReturn corrections JSON for the FLAGGED indices."
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        # Strip markdown fences if Claude added any despite the prompt.
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)

    def _identity_corrections(
        self, raw_words: list[WordWithConfidence], speakers: list[str]
    ) -> list[CorrectedWord]:
        return [
            CorrectedWord(
                word=w.word,
                changed=False,
                tavily_verified=False,
                unverified=False,
                speaker=speakers[i],  # type: ignore[arg-type]
            )
            for i, w in enumerate(raw_words)
        ]

    def _apply_and_guard(
        self,
        raw_words: list[WordWithConfidence],
        speakers: list[str],
        payload: dict[str, Any],
        verifications: dict[str, VerifyResult],
    ) -> list[CorrectedWord]:
        """Apply Claude's corrections, then enforce the hallucination guard."""
        # Index Claude's output by the index it gave us
        by_index: dict[int, dict[str, Any]] = {}
        for entry in payload.get("corrections", []):
            try:
                by_index[int(entry["index"])] = entry
            except (KeyError, ValueError, TypeError):
                continue

        verified_canonicals = {
            v.canonical.lower() for v in verifications.values() if v.canonical
        }

        result: list[CorrectedWord] = []
        for i, raw in enumerate(raw_words):
            speaker: Any = speakers[i]
            entry = by_index.get(i)
            if entry is None:
                # Word wasn't flagged — pass through unchanged
                result.append(
                    CorrectedWord(
                        word=raw.word,
                        changed=False,
                        tavily_verified=False,
                        unverified=False,
                        speaker=speaker,
                    )
                )
                continue

            corrected_text = str(entry.get("corrected", raw.word))
            tavily_verified = bool(entry.get("tavily_verified", False))
            unverified = bool(entry.get("unverified", False))
            changed = normalize(corrected_text) != normalize(raw.word)

            # --- HALLUCINATION GUARD ---
            # Any "changed" entry must (a) claim tavily_verified AND (b) actually map
            # to a canonical that came back from Tavily. If not, revert to original.
            if changed:
                canonical_match = normalize(corrected_text) in verified_canonicals
                if not (tavily_verified and canonical_match):
                    log.warning(
                        "Hallucination guard reverted '%s' -> '%s' (verified=%s, in_canonicals=%s)",
                        raw.word,
                        corrected_text,
                        tavily_verified,
                        canonical_match,
                    )
                    result.append(
                        CorrectedWord(
                            word=raw.word,
                            changed=False,
                            tavily_verified=False,
                            unverified=True,
                            speaker=speaker,
                        )
                    )
                    continue

            result.append(
                CorrectedWord(
                    word=corrected_text,
                    changed=changed,
                    tavily_verified=tavily_verified and changed,
                    unverified=unverified and not changed,
                    speaker=speaker,
                )
            )

        return result


_corrector: ClaudeCorrector | None = None


def get_corrector() -> ClaudeCorrector:
    global _corrector
    if _corrector is None:
        _corrector = ClaudeCorrector(
            api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
            model=settings.CLAUDE_MODEL,
        )
    return _corrector


def reset_corrector() -> None:
    global _corrector
    _corrector = None
