from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports like `backend.audio_preprocess...` work regardless of
# running pytest from repo root or from the backend/ directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
