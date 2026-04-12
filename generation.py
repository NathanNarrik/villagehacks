import json, csv
from pathlib import Path

run_dir = Path("backend/audio_gen/output/run_5x_v3")
src = run_dir / "clips_rich_noise.jsonl"
out = run_dir / "telephony_manifest.csv"

rows = []
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    rows.append({
        "audio_path": str((run_dir / row["audio_telephony_path"]).resolve()),
        "text": row["text"],
        "split": row["split"],
    })

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "text", "split"])
    writer.writeheader()
    writer.writerows(rows)

print(out)
