# Demo Audio

This folder holds the shipped demo catalog: six situations with four takes each.

Workflow:

1. Keep the canonical synthesis spec in `backend/audio_gen/input/demo_cards_20260412.csv`.
2. Keep each script text in `scripts/*.txt` exactly in sync with that CSV.
3. Regenerate and export the demo clips with:

```bash
conda run -n village-hacks python -m backend.audio_gen.build_demo_audio
```

Use `--resume` only if you are retrying a partially failed build; the wrapper does a clean rebuild by default.

4. The build writes generated artifacts to `backend/audio_gen/output/demo_cards_20260412/`.
5. Each of the six situations is expanded into four takes: `clear_call`, `ambient_noise`, `heavy_accent`, and `clinical_handoff`.
6. The shipped telephony WAVs are copied into `audio/` using canonical per-take filenames.
7. The same WAVs are copied into `frontend/public/demo-audio/<situation>/`, and a few top-level friendly aliases are preserved for backwards compatibility.
8. `manifest.csv` is rewritten as the checked-in mapping between canonical clip ids, backend demo WAVs, scripts, and frontend public assets.

Suggested qualities:

- 10-25 seconds
- clear medication and dose mentions
- ambient variants should sound like conversation, TV, music, or room tone rather than static hiss
- every situation should have at least one accented take
