from __future__ import annotations

from datetime import date

from scripts.show_study_heatmap import generate_random_activity, render_heatmap


def test_random_activity_is_deterministic() -> None:
    first = generate_random_activity(14, 123, today=date(2026, 8, 5))
    second = generate_random_activity(14, 123, today=date(2026, 8, 5))
    assert first == second
    assert len(first) == 14


def test_heatmap_renders_demo_summary_without_color() -> None:
    data = generate_random_activity(28, 123, today=date(2026, 8, 5))
    output = render_heatmap(data, no_color=True)
    assert "LexiPilot Study Intensity" in output
    assert "Legend:" in output
    assert "Active days:" in output
    assert "Source: deterministic random demo data" in output
    assert "\033[" not in output
