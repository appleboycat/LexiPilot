#!/usr/bin/env python3
"""Model-free smoke test for LexiPilot."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexipilot_core import build_session_plan
from lexipilot_tools import LexiPilotToolbox


def write_fixture(root: Path) -> LexiPilotToolbox:
    entries = [
        {"seq": 1, "word": "granular", "first_letter": "G", "page": 94, "phonetic": "英:/grænjələr/", "definition": "adj. 颗粒状的", "source_text": "granular adj. 颗粒状的"},
        {"seq": 2, "word": "redeem", "first_letter": "R", "page": 94, "phonetic": "英:/rɪdiːm/", "definition": "vt. 赎回；弥补", "source_text": "redeem vt. 赎回"},
        {"seq": 3, "word": "regiment", "first_letter": "R", "page": 95, "phonetic": "英:/redʒɪmənt/", "definition": "n. 团；严格管制", "source_text": "regiment n. 团"},
        {"seq": 4, "word": "impetus", "first_letter": "I", "page": 95, "phonetic": "英:/ɪmpɪtəs/", "definition": "n. 推动力", "source_text": "impetus n. 推动力"},
    ]
    index = root / ".vocab_index.json"
    index.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    progress = root / ".vocab_progress"
    state = {
        "profile": "smoke",
        "start_page": 1,
        "start_seq": 1,
        "last_new_seq": 3,
        "cards": {
            "1": {"stage": 1, "due": date.today().isoformat(), "seen": 3, "correct": 1},
            "2": {"stage": 2, "due": (date.today() - timedelta(days=1)).isoformat(), "seen": 4, "correct": 2},
            "3": {"stage": 1, "due": (date.today() + timedelta(days=1)).isoformat(), "seen": 1, "correct": 1},
        },
        "daily_stats": {},
        "daily_misses": {"2026-08-01": [1, 2]},
        "daily_miss_counts": {"2026-08-01": {"1": 3, "2": 1}},
        "daily_seen": {"2026-08-01": [1, 2, 3]},
    }
    path = progress / "smoke" / "progress.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return LexiPilotToolbox(index_path=index, progress_dir=progress, state_file=root / ".vocab_state.json", report_dir=root / "reports", material_dir=root / "materials")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="lexipilot_smoke_"))
    try:
        toolbox = write_fixture(root)
        summary = toolbox.get_profile_summary("smoke")
        check(summary["reviews_due_today"] == 2, "profile summary loaded")

        plan = build_session_plan(toolbox, "smoke", "I have 15 minutes. Focus on recently missed words.")
        planned = [word["word"] for word in plan["planned_words"]]
        check("granular" in planned and "redeem" in planned, "due words selected")
        check(planned[0] == "granular", "missed words prioritized")

        correct = toolbox.record_answer("smoke", "granular", True)
        check(correct["review_stage"] == 2, "correct answer advanced review stage")

        incorrect = toolbox.record_answer("smoke", "redeem", False)
        check(incorrect["review_stage"] == 0, "incorrect answer reset review stage")

        story = toolbox.generate_practice_story("smoke", ["redeem", "granular"], "academic", True)
        check("redeem" in story["english"] and story["chinese"], "personalized material generated")

        reloaded = toolbox.load_state("smoke")
        check(reloaded["cards"]["1"]["stage"] == 2 and reloaded["cards"]["2"]["stage"] == 0, "progress saved and reloaded")
    finally:
        shutil.rmtree(root)
    check(not root.exists(), "temporary files cleaned up")
    print("PASS LexiPilot smoke test")


if __name__ == "__main__":
    main()
