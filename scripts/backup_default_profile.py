#!/usr/bin/env python3
"""Back up the existing default LexiPilot/vocab_trainer profile."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


PROFILE_DIR = Path(".vocab_progress/default")
BACKUP_ROOT = Path(".vocab_progress_backups")


def backup_default_profile() -> Path:
    if not PROFILE_DIR.exists():
        raise SystemExit("Default profile does not exist: .vocab_progress/default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUP_ROOT / f"default_{timestamp}"
    if target.exists():
        raise SystemExit(f"Backup already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROFILE_DIR, target)
    return target


def main() -> int:
    path = backup_default_profile()
    print(f"Backup created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
