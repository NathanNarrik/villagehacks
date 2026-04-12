# Fine-Tuned Telephony Model Drop-In Folder

Place the Colab-exported Hugging Face Whisper model artifacts directly in this folder.

Expected export style:

- produced by `save_pretrained()`
- not Faster-Whisper / CTranslate2

Required files:

- `config.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `tokenizer.json` or `vocab.json`
- `model.safetensors` or `pytorch_model.bin`

Optional but commonly present:

- `generation_config.json`
- `special_tokens_map.json`
- `merges.txt`
- `normalizer.json`

Runtime behavior:

- `STT_PROVIDER=auto`: use this model if the folder is valid, otherwise fall back to Scribe v2
- `STT_PROVIDER=fine_tuned_telephony`: app startup should fail if this folder is missing or invalid
- `STT_PROVIDER=scribe_v2`: ignore this folder and use ElevenLabs Scribe
