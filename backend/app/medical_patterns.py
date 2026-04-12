"""Medical-pattern regex gate for the Tavily verifier.

A word only proceeds to Tavily if (a) it was flagged LOW-confidence by Layer 3 AND
(b) it matches one of these medical patterns. This is what makes the verification
"confidence-gated" rather than blasting Tavily on every uncertain word.
"""
from __future__ import annotations

import re

# Common drug-name endings. Pattern matches whole words ending in any of these.
_DRUG_SUFFIX_GROUP = (
    r"in|ol|mab|nib|pril|statin|mycin|azole|cillin|pine|sartan|prazole|caine|"
    r"oxetine|setron|tinib|olol|formin|profen|dronate"
)
DRUG_SUFFIXES = re.compile(rf"(?i)^[a-z]{{4,}}({_DRUG_SUFFIX_GROUP})$")

DOSAGE = re.compile(r"^\d+(\.\d+)?\s?(mg|mcg|ml|g|iu|units?)$", re.I)

SYMPTOM_ANCHORS = {
    "feeling",
    "pain",
    "ache",
    "dizzy",
    "dizziness",
    "nausea",
    "tired",
    "sore",
    "headache",
    "headaches",
    "fever",
    "cough",
}


def normalize(word: str) -> str:
    """Lowercase and strip surrounding punctuation."""
    return word.lower().strip(".,;:!?\"'()[]{}")


def matches_medical(word: str) -> bool:
    """True if `word` looks like a drug name or dosage and is worth verifying."""
    w = normalize(word)
    if not w:
        return False
    return bool(DRUG_SUFFIXES.match(w) or DOSAGE.match(w))
