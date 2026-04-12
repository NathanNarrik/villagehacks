"""Learning loop — verifies in-memory persistence across pipeline calls."""
from __future__ import annotations

from app import learning_loop
from app.schemas import CorrectedWord, VerifyResult, WordWithConfidence
from app.storage import store


def _raw(word: str) -> WordWithConfidence:
    return WordWithConfidence(
        word=word, start_ms=0, end_ms=100, speaker_id="speaker_0", confidence="LOW"
    )


def _corr(word: str, changed: bool, verified: bool) -> CorrectedWord:
    return CorrectedWord(
        word=word,
        changed=changed,
        tavily_verified=verified,
        unverified=False,
        speaker="Doctor",
    )


def test_record_call_grows_phonetic_map_and_history():
    raw = [_raw("metoformin"), _raw("lisinipril")]
    corrected = [
        _corr("metformin", changed=True, verified=True),
        _corr("lisinopril", changed=True, verified=True),
    ]
    verifications = {
        "metoformin": VerifyResult(
            original="metoformin", status="VERIFIED", canonical="metformin"
        ),
        "lisinipril": VerifyResult(
            original="lisinipril", status="VERIFIED", canonical="lisinopril"
        ),
    }

    learning_loop.record_call(raw, corrected, verifications)

    pmap = learning_loop.get_phonetic_map()
    assert pmap.get("metoformin") == "metformin"
    assert pmap.get("lisinipril") == "lisinopril"

    history = learning_loop.get_correction_history()
    assert history.get("metoformin") == 1
    assert history.get("lisinipril") == 1


def test_keyterms_accumulate_across_calls():
    raw = [_raw("metoformin")]
    corrected = [_corr("metformin", changed=True, verified=True)]
    verifications = {
        "metoformin": VerifyResult(
            original="metoformin", status="VERIFIED", canonical="metformin"
        )
    }

    learning_loop.record_call(raw, corrected, verifications)
    learning_loop.record_call(raw, corrected, verifications)
    learning_loop.record_call(raw, corrected, verifications)

    keyterms = learning_loop.get_keyterms(top_n=10)
    assert "metformin" in keyterms
    # The sorted-set frequency should reflect 3 calls (each call: +1 for changed
    # correction +1 for VERIFIED canonical = +2 per call)
    assert store._zsets["cc:keyterms"]["metformin"] >= 3.0


def test_unverified_corrections_dont_pollute_keyterms():
    raw = [_raw("xyzabc")]
    corrected = [_corr("xyzabc", changed=False, verified=False)]
    verifications = {
        "xyzabc": VerifyResult(original="xyzabc", status="UNVERIFIED")
    }

    learning_loop.record_call(raw, corrected, verifications)

    # Learned sorted-set stays empty — nothing verified to persist.
    assert learning_loop.keyterm_count() == 0
    assert learning_loop.get_phonetic_map() == {}
    # `get_keyterms` still returns Person A's seed list when the store is empty (Scribe keywords).
    seeded = learning_loop.get_keyterms(top_n=500)
    assert "xyzabc" not in {k.lower() for k in seeded}
    assert len(seeded) >= 10
