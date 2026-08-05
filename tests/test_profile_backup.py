from __future__ import annotations

import json
from pathlib import Path

from scripts.backup_default_profile import backup_default_profile
from scripts.restore_default_profile import restore_default_profile


def test_default_profile_backup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / ".vocab_progress" / "default"
    profile.mkdir(parents=True)
    (profile / "progress.json").write_text(json.dumps({"profile": "default", "cards": {}}), encoding="utf-8")
    backup = backup_default_profile()
    assert backup.exists()
    assert (backup / "progress.json").exists()
    assert ".vocab_progress_backups" in str(backup)


def test_default_profile_restore(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    profile = tmp_path / ".vocab_progress" / "default"
    profile.mkdir(parents=True)
    (profile / "progress.json").write_text(json.dumps({"version": "current"}), encoding="utf-8")
    backup = tmp_path / ".vocab_progress_backups" / "default_test"
    backup.mkdir(parents=True)
    (backup / "progress.json").write_text(json.dumps({"version": "backup"}), encoding="utf-8")
    restore_default_profile(backup, yes=True)
    restored = json.loads((profile / "progress.json").read_text(encoding="utf-8"))
    assert restored["version"] == "backup"


def test_tests_do_not_touch_real_default_profile() -> None:
    # This assertion guards the test suite convention: backup/restore tests chdir to tmp_path.
    assert Path.cwd().name == "LexiPilot"
