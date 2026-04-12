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
