# Backend Audio Preprocessing

This backend module adds a Scribe-focused preprocessing contract that always emits:

- 16kHz sample rate
- mono channel
- PCM signed 16-bit little-endian (`pcm_s16le`)
- WAV container

## Conda setup (Python 3.11 + pytest)

From repository root:

```bash
conda env create -f backend/environment.yml
conda activate villagehacks-audio
```

`ffmpeg` and `ffprobe` must be available in `PATH`.

Examples:

- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`

## Public entrypoint

```python
from backend.audio_preprocess import preprocess_for_scribe

result = preprocess_for_scribe(
    input_path="/path/to/input_audio.wav",
    output_dir="/path/to/tmp",
    job_id="call_123",
)
```

## ffmpeg chain (fixed order)

`loudnorm=I=-16:LRA=11:TP=-1.5,afftdn=nf=-25,aresample=16000:resampler=soxr`

If `soxr` is unavailable in your ffmpeg build, the pipeline automatically retries with:
`loudnorm=I=-16:LRA=11:TP=-1.5,afftdn=nf=-25,aresample=16000`.

## Future /transcribe adapter

```python
from backend.audio_preprocess import prepare_transcribe_audio

payload = prepare_transcribe_audio(input_path="/path/to/input.wav", working_dir="/tmp")
# payload["preprocessed_wav_path"]
# payload["preprocessing_metrics"]
```

## ElevenLabs dataset generation

Build 5x scenario variants from a separator-delimited text file (`---` lines are ignored):

```bash
python -m backend.audio_gen.build_variants \
  --clips-file backend/audio_gen/clips.txt \
  --output backend/audio_gen/input/clips_5x_variants.csv
```

Run from repository root:

```bash
python -m backend.audio_gen.run \
  --input /path/to/clips.csv \
  --out-dir /path/to/output \
  --concurrency 3 \
  --model-id eleven_multilingual_v2 \
  --resume
```

Required environment variable:

- `ELEVEN_LABS_API_KEY`

Input requirements:

- Input file must be `.csv` or `.jsonl`
- Preflight validation is strict and fails before generation on schema/row errors
- Required grouping and numerics fields include:
  - `script_family_id`
  - `scenario_group`
  - `numeric_confusion_type` (`digit_vs_digit|dose_confusion|duration_confusion|none`)
- `voice_type` is validated against enums:
  - `neutral|telephony|accented|clinical`
- Medical/category consistency is enforced:
  - `medical_domain=true` requires category in `medical_conversation|clinical_triage|adverse_event_followup`
  - `medical_domain=false` rejects those medical-only categories
- Scenario rules are validated:
  - `clean_speech` requires `noise_profile=clean`
  - `clean_speech` requires `scenario_group=baseline`
  - `noisy_environment` requires `noise_profile` in `medium|high`
  - `noisy_environment` requires `scenario_group=noisy`
  - `accented_speech` requires non-empty `accent_profile`
  - `accented_speech` requires `scenario_group=accented`
  - `medical_conversation` requires `medical_domain=true` and non-empty `medical_subtype`
  - `medical_conversation` requires `scenario_group=medical`
- For `noisy_environment` and `medical_conversation`, the generator injects synthetic
  background noise into clean audio before telephony conversion. `noise_profile=high`
  is stronger than `noise_profile=medium`.

Primary artifacts:

- `clips.jsonl`
- `generation_errors.jsonl`
- `run_metadata.json`
- `word_features.template.jsonl`
- `numeric_features.template.jsonl`
- `medical_entities.template.jsonl`

Resume determinism:

- `manifest_version` and `input_hash` are written to outputs
- resume mode hard-fails if current input hash differs from existing run metadata

## Run tests

From repository root:

```bash
pytest backend/tests
```

Or from inside `backend/`:

```bash
pytest tests
```

Integration tests auto-skip if `ffmpeg`/`ffprobe` are unavailable.
