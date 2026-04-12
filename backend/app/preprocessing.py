"""Layer 1 — Audio Preprocessing.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations


async def preprocess(input_path: str) -> str:
    """Run loudnorm → afftdn → 16kHz mono PCM WAV. Return path to cleaned WAV.

    Implementation notes:
        ffmpeg -y -i {input} \\
          -af "loudnorm=I=-16:LRA=11:TP=-1.5,afftdn=nf=-25" \\
          -ar 16000 -ac 1 -c:a pcm_s16le {output}

    Use asyncio.create_subprocess_exec so the event loop stays responsive.
    Raise an exception (any) on non-zero exit — the pipeline route turns it into a 500.
    """
    raise NotImplementedError("preprocessing.preprocess — Person A: see HANDOFF_PERSON_A.md")
