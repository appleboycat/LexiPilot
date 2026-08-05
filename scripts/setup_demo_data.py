#!/usr/bin/env python3
"""Create a deterministic synthetic learner profile for the public demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vocab_trainer as vt
from scripts.validate_vocab_index import validate_vocab_index


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def demo_state(profile: str, today: date) -> dict[str, Any]:
    due_offsets = {
        1: -2,
        2: 0,
        3: -1,
        4: 0,
        5: -3,
        6: 0,
        7: 0,
        8: 1,
        9: 2,
        10: 4,
        11: 7,
        12: 14,
    }
    missed_counts = {2: 4, 5: 3, 6: 5, 9: 2}
    cards: dict[str, dict[str, Any]] = {}
    for seq in range(1, 13):
        misses = missed_counts.get(seq, 0)
        correct = 2 + (seq % 3)
        cards[str(seq)] = {
            "stage": min(4, 1 + (seq % 4)),
            "due": (today + timedelta(days=due_offsets[seq])).isoformat(),
            "seen": correct + misses,
            "correct": correct,
        }

    daily_stats: dict[str, dict[str, int]] = {}
    daily_seen: dict[str, list[int]] = {}
    daily_misses: dict[str, list[int]] = {}
    daily_miss_counts: dict[str, dict[str, int]] = {}
    synthetic_days = [
        (6, [1, 2, 3], {2: 2}),
        (4, [4, 5, 6], {5: 2, 6: 1}),
        (3, [2, 7, 8], {2: 1}),
        (1, [5, 6, 9], {5: 1, 6: 3, 9: 1}),
        (0, [2, 6, 9, 10], {2: 1, 6: 1, 9: 1}),
    ]
    for days_ago, seen, misses in synthetic_days:
        day = (today - timedelta(days=days_ago)).isoformat()
        missed_total = sum(misses.values())
        daily_stats[day] = {
            "studied": len(seen),
            "new": 1 if days_ago >= 3 else 0,
            "review": len(seen) - (1 if days_ago >= 3 else 0),
            "remembered": max(0, len(seen) - len(misses)),
            "missed": missed_total,
        }
        daily_seen[day] = seen
        daily_misses[day] = list(misses)
        daily_miss_counts[day] = {str(seq): count for seq, count in misses.items()}

    return {
        "profile": vt.normalize_profile_name(profile),
        "start_page": 1,
        "start_seq": 1,
        "last_new_seq": 12,
        "cards": cards,
        "daily_stats": daily_stats,
        "daily_misses": daily_misses,
        "daily_miss_counts": daily_miss_counts,
        "daily_seen": daily_seen,
        "demo_data": True,
    }


def setup_demo_data(
    *,
    index_path: Path | str = REPO_ROOT / "examples" / "sample_vocab_index.json",
    progress_root: Path | str = REPO_ROOT / ".demo_data" / "profiles",
    profile: str = "demo",
    force: bool = False,
    today: date | None = None,
) -> Path:
    index = Path(index_path)
    validate_vocab_index(index)
    output = Path(progress_root) / vt.normalize_profile_name(profile) / "progress.json"
    if output.exists() and not force:
        return output
    atomic_write_json(output, demo_state(profile, today or date.today()))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create reproducible synthetic LexiPilot demo data.")
    parser.add_argument("--index-file", type=Path, default=REPO_ROOT / "examples" / "sample_vocab_index.json")
    parser.add_argument("--progress-root", type=Path, default=REPO_ROOT / ".demo_data" / "profiles")
    parser.add_argument("--profile", default="demo")
    parser.add_argument("--force", action="store_true", help="Replace an existing synthetic demo profile")
    args = parser.parse_args()
    existed = (args.progress_root / vt.normalize_profile_name(args.profile) / "progress.json").exists()
    try:
        path = setup_demo_data(
            index_path=args.index_file,
            progress_root=args.progress_root,
            profile=args.profile,
            force=args.force,
        )
    except Exception as exc:
        print(f"FAIL demo data setup: {exc}")
        return 1
    action = "kept" if existed and not args.force else "created"
    print(f"PASS demo profile {action}: {path}")
    print("Started words: 12")
    print("Due today: 7")
    print("Historically missed words: 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
