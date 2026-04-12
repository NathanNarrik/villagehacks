# Telephony STT Fine-Tuning

This folder contains the sponsor-facing STT fine-tuning handoff:

- a Colab-ready telephony dataset staging script
- a Whisper-small Colab notebook
- accent coverage checks so the train split is not overly narrow
- a drop-in runtime path for the exported Hugging Face model

## Runtime drop-in path

Once Colab training finishes, copy the `save_pretrained()` export into:

`backend/stt/models/fine_tuned_telephony/`

`STT_PROVIDER=auto` also recognizes a backend-root drop-in for convenience:

`backend/whisper_small_telephony_final/`

The backend batch transcription path will then use that model automatically when:

- `STT_PROVIDER=auto` and the folder is valid
- or `STT_PROVIDER=fine_tuned_telephony`

If the folder is missing or invalid and `STT_PROVIDER=auto`, the app falls back to
ElevenLabs Scribe v2.

For lower local latency:

- use `FINE_TUNED_STT_DEVICE=auto` or `mps` on Apple Silicon
- keep `FINE_TUNED_STT_WORD_TIMESTAMPS=false` unless you specifically need real word timestamps

## Recommended dataset

Use:

`backend/audio_gen/output/run_5x_v2/clips_rich_noise_rebalanced.csv`

Why:

- it already points at telephony audio via `audio_telephony_path`
- it already includes the transcript text in `text`
- it already includes `train` / `val` / `test`
- the telephony clips are narrowband 8 kHz audio, which matches the sponsor requirement

Keep these out of training:

- `backend/test_audio/benchmark/v1/`
- `backend/test_audio/demo/`

Those should remain held-out evaluation and demo assets.

## Build a Colab-ready dataset

From the repo root:

```bash
python3 -m backend.stt.build_telephony_manifest
```

That writes:

- `backend/stt/colab_dataset/telephony_manifest.csv`
- `backend/stt/colab_dataset/telephony_manifest_summary.json`
- `backend/stt/colab_dataset/audio/*.wav`

The manifest format is intentionally Colab-friendly:

- `audio_path`
- `text`
- `split`

## Accent coverage

The builder enforces accent coverage on the training split.

Default policy:

- require at least `2` distinct train accent buckets
- require at least `40` train rows per accent bucket after balancing
- oversample underrepresented train accents up to `50%` of the dominant train accent count

With the current `run_5x_v2` source data, that means the minority
`south_asian_english` bucket is duplicated in the train manifest so it is better
represented during fine-tuning.

You can tune the policy:

```bash
python3 -m backend.stt.build_telephony_manifest \
  --train-accent-target-ratio 0.4 \
  --min-train-samples-per-accent 30
```

## Copy to Drive for Colab

Copy the contents of `backend/stt/colab_dataset/` into:

`/content/drive/MyDrive/carecaller/`

Expected layout in Drive:

```text
carecaller/
  telephony_manifest.csv
  telephony_manifest_summary.json
  audio/
    clip_0001_clean.wav
    ...
```

## Run the notebook

Open:

`backend/stt/whisper_small_telephony_colab.ipynb`

in Colab and run the cells top to bottom.
