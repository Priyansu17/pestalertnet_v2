"""Phase 1b helper: write a size-sorted inventory of all project files.

Read-only. Used to verify which artifacts exist after ingestion/validation.
Output: outputs/inventory.txt
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "outputs" / "inventory.txt"


def main() -> int:
    entries = []
    for root, _dirs, files in os.walk(PROJECT_ROOT):
        for name in files:
            if name == ".gitkeep":
                continue
            full = Path(root) / name
            rel = full.relative_to(PROJECT_ROOT).as_posix()
            entries.append((rel, full.stat().st_size))
    entries.sort()

    lines = [f"TOTAL-FILES: {len(entries)}"]
    for rel, size in entries:
        lines.append(f"{rel:<64} {size:>10}")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {OUT_PATH} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())