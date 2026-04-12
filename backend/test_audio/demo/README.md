# Demo Audio

This folder holds the shipped six-clip demo set.

Workflow:

1. Keep the canonical synthesis spec in `backend/audio_gen/input/demo_cards_20260412.csv`.
2. Keep each script text in `scripts/*.txt` exactly in sync with that CSV.
3. Regenerate and export the demo clips with:

```bash
python -m backend.audio_gen.build_demo_audio
```

Use `--resume` only if you are retrying a partially failed build; the wrapper does a clean rebuild by default.

4. The build writes generated artifacts to `backend/audio_gen/output/demo_cards_20260412/`.
5. The shipped telephony WAVs are copied into `audio/` using dated canonical filenames.
6. The same WAVs are copied into `frontend/public/demo-audio/` using the friendly public names used by the UI.
7. `manifest.csv` is rewritten as the checked-in mapping between canonical clip ids, backend demo WAVs, scripts, and frontend public assets.

Suggested qualities:

- 10-25 seconds
- clear medication and dose mentions
- at least one mild-noise clip and one accented-speaker clip
