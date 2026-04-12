# Person A — Handoff

You own **Layer 1 (audio preprocessing)**, **Layer 2 (ElevenLabs Scribe v2)**,
**Layer 3 (uncertainty scoring)**, and the **benchmark generation script**.
Person B + C have already shipped everything else — the FastAPI backend boots
right now and `/transcribe` will start working as soon as the 5 stub functions
below are filled in.

## What you implement

5 Python functions + 1 JSON file. Every type you need is in `app/schemas.py` —
do not redefine them.

| File | Function | Returns |
|---|---|---|
| `app/preprocessing.py` | `async def preprocess(input_path: str) -> str` | Path to cleaned WAV |
| `app/scribe.py` | `async def transcribe_batch(wav_path: str, keyterms: list[str]) -> ScribeResult` | `ScribeResult` from schemas.py |
| `app/uncertainty.py` | `def score_words(words, keyterms, phonetic_map, correction_history) -> list[WordWithConfidence]` | List parallel to `words` |
| `app/phonetic.py` | `def normalized_levenshtein(a: str, b: str) -> float` | float in `[0, 1]` |
| `app/keyterms.py` | `def load_initial_keyterms() -> list[str]` | ~50 medical terms |
| `data/benchmark_results.json` | (file, not function) | Matches `BenchmarkResponse` schema |

When all five raise `NotImplementedError` (the current state), `POST /transcribe`
returns `501 Person A module not implemented: <function name>`. As you fill them
in, the error message will name the next missing one.

---

## 1. `app/preprocessing.py` — Layer 1

```python
async def preprocess(input_path: str) -> str
```

Run an ffmpeg subprocess to normalize loudness, denoise, and resample to 16 kHz
mono PCM.

```
ffmpeg -i {input} -af loudnorm=I=-16:LRA=11:TP=-1.5,afftdn=nf=-25 \
       -ar 16000 -ac 1 -c:a pcm_s16le {output.wav}
```

- Use `asyncio.create_subprocess_exec` so the event loop stays responsive.
- Return the absolute path to the cleaned WAV (write to `tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name`).
- Raise `RuntimeError("preprocessing failed: <stderr tail>")` if ffmpeg exits non-zero.

---

## 2. `app/scribe.py` — Layer 2

```python
async def transcribe_batch(wav_path: str, keyterms: list[str]) -> ScribeResult
```

Submit the cleaned audio to ElevenLabs Scribe v2 batch.

Required SDK call parameters:
```python
model_id="scribe_v2"
diarize=True
tag_audio_events=True
timestamps_granularity="word"
keywords=keyterms[:100]   # SDK caps at 100 per request
```

Return a `ScribeResult` (from `app/schemas.py`):
```python
ScribeResult(
    words=[ScribeWord(text=..., start_ms=..., end_ms=..., speaker_id="speaker_0"), ...],
    audio_events=["[coughing]", "[silence]"],
    duration_ms=int(...),
)
```

**Speaker labels**: leave them as raw `"speaker_0"` / `"speaker_1"`. The pipeline
in `app/pipeline.py` runs a heuristic (first speaker to utter a drug-suffix word
becomes Doctor) — you don't need to map them yourself.

API key comes from `settings.ELEVENLABS_API_KEY` (already wired in `app/config.py`).

---

## 3. `app/uncertainty.py` — Layer 3

```python
def score_words(
    words: list[ScribeWord],
    keyterms: list[str],
    phonetic_map: dict[str, str],
    correction_history: dict[str, int],
) -> list[WordWithConfidence]
```

Combine four signals into a confidence score in `[0, 1]`:

| Signal | Weight | Trigger |
|---|---|---|
| Timing irregularity | +0.30 | Word duration vs rolling 5-word median, z-score > 1.5 |
| Keyterm mismatch | +0.25 | `medical_patterns.matches_medical(word)` true but word not in `keyterms` |
| Phonetic distance | +0.30 | Min Levenshtein to keyterms is 1 or 2 |
| Correction likelihood | +0.15 | Word appears in `correction_history` |

Buckets:
- score < 0.25 → `"HIGH"`
- 0.25 ≤ score < 0.5 → `"MEDIUM"`
- score ≥ 0.5 → `"LOW"`

For LOW/MEDIUM words, populate `uncertainty_signals` with the names of the
contributing signals so the frontend tooltip can show them:
```python
["timing_irregularity", "phonetic_distance: 1", "keyterm_mismatch"]
```

Return a list **parallel** to `words` (same length, same order, same `speaker_id`).

---

## 4. `app/phonetic.py`

```python
def normalized_levenshtein(a: str, b: str) -> float
```

Edit distance from `a` to `b`, normalized to `[0, 1]` by `max(len(a), len(b))`.
Used by `uncertainty.score_words` for the phonetic-distance signal. You may
optionally add Double Metaphone matching, but only this one function is required.

---

## 5. `app/keyterms.py`

```python
def load_initial_keyterms() -> list[str]
```

Return ~50 common medical terms. Suggested categories:

- **Cardiovascular**: lisinopril, atorvastatin, amlodipine, metoprolol, losartan
- **Diabetes**: metformin, insulin, glipizide, sitagliptin
- **Mental health**: sertraline, escitalopram, bupropion, fluoxetine
- **Pain**: ibuprofen, acetaminophen, tramadol, gabapentin
- **Antibiotics**: amoxicillin, azithromycin, doxycycline, ciprofloxacin
- **GI**: omeprazole, pantoprazole, ranitidine, famotidine

The learning loop calls this only when its sorted set is empty (fresh process).
After the first successful pipeline run, learned keyterms take over.

---

## 6. `data/benchmark_results.json`

Run your TTS + WER benchmark script and write the results to
`backend/data/benchmark_results.json`. The shape **must** match
`BenchmarkResponse` in `app/schemas.py`. A reference file is committed at
`backend/data/benchmark_results.json.example` — copy it and overwrite the
numbers.

Until this file exists, `GET /benchmark` returns 503 with the message
`"Benchmark results not generated yet — Person A: run scripts/run_benchmark.py"`.

---

## Environment

Add `ELEVENLABS_API_KEY=...` to `backend/.env` (the example file already lists
the variable, just paste the key in).

## Running locally with your code

```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

Hit `POST http://localhost:8000/transcribe` with a multipart `file` field. As
each stub gets filled in, the 501 message will move to the next missing one.

## What I will NOT touch

Person B + C own everything outside the 5 files above. Don't modify
`pipeline.py`, `tavily_verify.py`, `claude_*.py`, `learning_loop.py`,
`storage.py`, `main.py`, `config.py`, or `schemas.py` — if you think one of
those needs to change, ping me first.
