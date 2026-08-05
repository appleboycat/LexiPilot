from __future__ import annotations

from datetime import date

from scripts.show_study_heatmap import (
    DayActivity,
    activity_from_daily_stats,
    generate_random_activity,
    render_heatmap,
)


def test_random_activity_is_deterministic() -> None:
    first = generate_random_activity(14, 123, today=date(2026, 8, 5))
    second = generate_random_activity(14, 123, today=date(2026, 8, 5))
    assert first == second
    assert len(first) == 14


def test_heatmap_renders_demo_summary_without_color() -> None:
    data = generate_random_activity(28, 123, today=date(2026, 8, 5))
    output = render_heatmap(data, no_color=True)
    assert "LexiPilot Study Activity" in output
    assert "Intensity Legend" in output
    assert "Level    Mark  Words" in output
    assert "Active days:" in output
    assert "Source: deterministic random demo data" in output
    assert output.startswith("╭")
    assert output.endswith("╯")
    assert "\033[" not in output


def test_profile_activity_uses_recent_aggregated_daily_stats() -> None:
    data = activity_from_daily_stats(
        {
            "2026-08-04": {"studied": 8, "remembered": 6, "missed": 2},
            "2026-08-05": {"studied": 20, "remembered": 15, "missed": 5},
        },
        3,
        today=date(2026, 8, 5),
    )
    assert [item.words for item in data] == [0, 8, 20]
    output = render_heatmap(
        data,
        no_color=True,
        source="aggregated learner activity for profile demo",
    )
    assert "Words reviewed: 28" in output
    assert "Missed: 7" in output
    assert "Source: aggregated learner activity for profile demo" in output


def test_activity_color_depth_matches_legend(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    data = [
        DayActivity(date(2026, 8, 1), 0, 0, 0),
        DayActivity(date(2026, 8, 2), 2, 2, 0),
        DayActivity(date(2026, 8, 3), 7, 6, 1),
        DayActivity(date(2026, 8, 4), 15, 12, 3),
        DayActivity(date(2026, 8, 5), 30, 25, 5),
    ]
    output = render_heatmap(data)
    for code in ("48;5;236;37", "48;5;22;97", "48;5;28;97", "48;5;34;30", "48;5;40;30"):
        assert f"\033[{code}m" in output
