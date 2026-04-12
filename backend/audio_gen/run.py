"""CLI entrypoint for ElevenLabs clip generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .elevenlabs import ElevenLabsClient
from .generator import GenerationConfig, run_generation


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _load_dotenv_if_present(Path.cwd() / ".env")
    _load_dotenv_if_present(Path.cwd() / "backend" / ".env")

    api_key = os.getenv("ELEVEN_LABS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ELEVEN_LABS_API_KEY is required in environment or .env file")

    config = GenerationConfig(
        input_path=Path(args.input).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        concurrency=args.concurrency,
        model_id=args.model_id,
        resume=args.resume,
        timeout_s=args.timeout_s,
    )

    client = ElevenLabsClient(api_key=api_key, timeout_s=args.timeout_s)
    metadata = run_generation(config, client)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate clip-level audio dataset using ElevenLabs")
    parser.add_argument("--input", required=True, help="Path to clip metadata file (.csv or .jsonl)")
    parser.add_argument("--out-dir", required=True, help="Output directory for audio and manifests")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent clip generations")
    parser.add_argument("--model-id", default="eleven_multilingual_v2", help="ElevenLabs model id")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="ffmpeg/ffprobe and request timeout")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def _load_dotenv_if_present(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
