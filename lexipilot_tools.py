#!/usr/bin/env python3
"""Structured vocabulary tools for LexiPilot.

This module is intentionally thin: it imports the existing trainer functions
and exposes them as deterministic, OpenAI-compatible tools.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import vocab_trainer as vt

REPO_ROOT = Path(__file__).resolve().parent
SIBLING_AIAGENT_ENV = REPO_ROOT.parent / "aiagent" / ".env"


class ConfigError(ValueError):
    pass


def parse_env_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def load_simple_env_file(path: Path | str, *, override: bool = False, protected_keys: set[str] | None = None) -> bool:
    protected = protected_keys or set()
    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = (REPO_ROOT / env_path).resolve()
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in protected:
            continue
        if override or os.environ.get(key, "") == "":
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return True


def _load_env_with_local_override(path: Path, protected_keys: set[str]) -> list[Path]:
    loaded: list[Path] = []
    if load_simple_env_file(path, override=False, protected_keys=protected_keys):
        loaded.append(path)
    local_override = path.with_name(".env.local") if path.name == ".env" else None
    if local_override is not None and load_simple_env_file(local_override, override=True, protected_keys=protected_keys):
        loaded.append(local_override)
    return loaded


def normalize_base_url(base_url: str) -> str:
    stripped = base_url.strip().rstrip("/")
    if not stripped:
        return ""
    parsed = urllib.parse.urlsplit(stripped)
    if not parsed.scheme or not parsed.netloc:
        raise ConfigError("RADEON_BASE_URL must be an absolute URL.")
    path = parsed.path.rstrip("/")
    normalized_path = path if path == "/v1" or path.endswith("/v1") else f"{path}/v1" if path else "/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def load_lexipilot_env(env_file: Path | str | None = None) -> list[Path]:
    loaded: list[Path] = []
    protected_keys = {key for key, value in os.environ.items() if value != ""}
    explicit = env_file or os.getenv("LEXIPILOT_ENV_FILE")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        loaded.extend(_load_env_with_local_override(path, protected_keys))
        return loaded

    local_env = REPO_ROOT / ".env"
    loaded.extend(_load_env_with_local_override(local_env, protected_keys))

    required = ("RADEON_API_KEY", "RADEON_BASE_URL", "RADEON_MODEL")
    if any(not os.getenv(name, "").strip() for name in required):
        loaded.extend(_load_env_with_local_override(SIBLING_AIAGENT_ENV, protected_keys))
    return loaded


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_entry(entry: dict[str, Any], card: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "word": str(entry.get("word", "")),
        "phonetic": vt.display_phonetic(entry),
        "definition": vt.plain_definition(entry.get("definition") or entry.get("source_text") or ""),
        "page": int(entry.get("page", 0)),
        "seq": int(entry.get("seq", 0)),
    }
    if card is not None:
        seen = int(card.get("seen", 0))
        correct = int(card.get("correct", 0))
        data.update(
            {
                "review_stage": int(card.get("stage", 0)),
                "due_date": str(card.get("due", "")),
                "correct_count": correct,
                "missed_count": max(0, seen - correct),
            }
        )
    return data


def chinese_phrases_for_entry(entry: dict[str, Any]) -> list[str]:
    definition = vt.plain_definition(entry.get("definition") or entry.get("source_text") or "")
    definition = re.sub(r"\b(?:abbr|prep|conj|pron|num|int|adj|adv|vt|vi|n|v)\.\s*", " ", definition)
    phrases: list[str] = []
    for part in re.split(r"[；;，,、（）()【】\[\]\s]+", definition):
        cleaned = part.strip()
        if cleaned and vt.CJK_RE.search(cleaned) and cleaned not in phrases:
            phrases.append(cleaned)
    return phrases[:4]


def parse_radeon_story_payload(
    text: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate model prose and exact Chinese phrases without retaining reasoning."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None

    english = vt.clean_spaces(str(parsed.get("english", "")))
    chinese = vt.clean_spaces(str(parsed.get("chinese", "")))
    raw_mappings = parsed.get("target_translations")
    if not english or not chinese or not isinstance(raw_mappings, dict):
        return None
    mappings_by_word = {
        str(word).strip().lower(): values
        for word, values in raw_mappings.items()
        if str(word).strip()
    }
    target_translations: dict[str, list[str]] = {}
    for entry in entries:
        word = str(entry.get("word", "")).strip()
        values = mappings_by_word.get(word.lower())
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return None
        phrases: list[str] = []
        for value in values:
            phrase = vt.clean_spaces(str(value)).strip()
            if (
                len(phrase) >= 2
                and "\x1b" not in phrase
                and phrase in chinese
                and phrase not in phrases
            ):
                phrases.append(phrase)
        if not phrases:
            return None
        target_translations[word] = phrases
    return {
        "english": english,
        "chinese": chinese,
        "target_translations": target_translations,
    }


def _entry_pos(entry: dict[str, Any]) -> str:
    definition = str(entry.get("definition") or entry.get("source_text") or "")
    match = re.search(r"\b(adj|adv|vt|vi|v|n)\.", definition)
    return match.group(1) if match else ""


def _join_words(words: list[str]) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"


FALLBACK_ENGLISH_SENTENCES = {
    "abandon": "The researchers refused to abandon the field study when the first set of measurements proved incomplete.",
    "abate": "To abate confusion about the unusual readings, the team recalibrated every sensor before collecting more evidence.",
    "abbey": "The study monitored humidity around a restored abbey whose stone walls were vulnerable to long-term weather damage.",
    "abbreviate": "Researchers agreed to abbreviate routine labels while preserving complete notes for every unusual observation.",
    "aberrant": "An aberrant temperature reading prompted the team to inspect the instrument instead of accepting the value immediately.",
    "abhor": "The lead scholar came to abhor careless shortcuts because one undocumented change could undermine the entire study.",
    "abiding": "Her abiding commitment to accurate records helped the group explain the anomaly with confidence.",
    "abrupt": "An abrupt drop in temperature led the team to compare the sensor data with the local weather station.",
    "farce": "A student newspaper first treated the delayed repair as a farce, but the researchers looked for evidence instead of ridicule.",
    "far-fetched": "They rejected a far-fetched rumor that the entire study had been invented to embarrass the administration.",
    "fatigue": "Survey data showed that fatigue rose sharply among students who had slept poorly for several nights.",
    "faucet": "A leaking faucet in the residence hall gave the team a concrete example of how small maintenance failures can affect daily life.",
    "feedback": "Residents' feedback helped the team separate personal complaints from patterns that appeared across the whole building.",
    "flake": "A flake of old paint near the sink supported the claim that the room had been neglected for a long time.",
    "expropriate": "When one proposal tried to expropriate a shared storage room for private offices, students objected at the hearing.",
    "forfeit": "The dean warned that the university would forfeit public trust if it ignored the evidence.",
    "frenzy": "Careful reporting prevented the meeting from turning into a frenzy.",
    "exotic": "The committee avoided exotic explanations and focused on ordinary causes that the data could support.",
    "flair": "One student with a flair for visual design turned the findings into a clear public poster.",
    "extol": "The final report did not simply extol the research team; it explained how the evidence had changed campus policy.",
    "faculty": "Faculty members reviewed the evidence before the recommendations were sent to the dean.",
    "ferment": "After several ignored complaints, frustration began to ferment among residents and pushed them to organize a formal petition.",
    "fertile": "The neglected courtyard became fertile ground for a discussion about how shared spaces influence health, safety, and community life.",
}


FALLBACK_CHINESE_SENTENCES = {
    "abandon": "第一组测量不完整时，研究者没有放弃这项实地研究。",
    "abate": "为了减轻异常读数造成的困惑，团队重新校准了每个传感器，然后再收集证据。",
    "abbey": "这项研究监测了一座修道院周围的湿度，因为它的石墙容易受到长期天气变化的影响。",
    "abbreviate": "研究者同意缩写常规标签，同时为每个异常观察保留完整记录。",
    "aberrant": "一个异常的温度读数促使团队检查仪器，而不是立即接受这个数值。",
    "abhor": "首席学者逐渐痛恨草率的捷径，因为一次没有记录的改动就可能破坏整项研究。",
    "abiding": "她持久的严谨记录帮助团队有把握地解释了这个异常现象。",
    "abrupt": "温度突然下降后，团队把传感器数据与当地气象站进行了比较。",
    "farce": "学生报纸起初把维修拖延写成一场笑剧，但研究者选择依据证据分析，而不是嘲笑。",
    "far-fetched": "他们排除了一个牵强的传言：整项研究是为了让校方难堪而编造的。",
    "fatigue": "调查数据显示，连续几晚睡眠不好后，学生的疲劳感明显上升。",
    "faucet": "宿舍里漏水的旋塞给团队提供了一个具体例子，说明小的维修问题也会影响日常生活。",
    "feedback": "住户的反馈帮助团队区分个人抱怨和整栋楼反复出现的共同问题。",
    "flake": "水槽旁一小片剥落的旧漆支持了房间长期缺乏维护的判断。",
    "expropriate": "当一项方案试图征用公共储物间改成私人办公室时，学生在听证会上提出反对。",
    "forfeit": "院长警告说，如果学校无视证据，就会丧失公众信任。",
    "frenzy": "谨慎的报告避免了会议变成一场狂乱的争吵。",
    "exotic": "委员会没有采用异国情调式的离奇解释，而是关注数据能够支持的普通原因。",
    "flair": "一名有设计天分的学生把研究结果做成了清晰的公共海报。",
    "extol": "最终报告并不是单纯颂扬研究团队，而是解释证据如何改变了校园政策。",
    "faculty": "教师成员在建议提交给院长前审阅了证据。",
    "ferment": "几次投诉被忽视后，住户之间的不满开始发酵，并促使他们组织正式请愿。",
    "fertile": "这片原本可以是肥沃的公共庭院，如今却成了讨论健康、安全和社区生活的有力例子。",
}


def _fallback_sentence(entry: dict[str, Any]) -> str:
    word = str(entry["word"])
    known = FALLBACK_ENGLISH_SENTENCES.get(word.lower())
    if known:
        return known
    pos = _entry_pos(entry)
    meaning = " / ".join(chinese_phrases_for_entry(entry)[:2]) or vt.short_meaning(entry)
    if pos == "adj":
        return f"The committee described one recurring condition as {word}, linking the label to residents' reports of {meaning}."
    if pos in {"v", "vt", "vi"}:
        return f"The revised policy had to {word} the problem in a way that residents could connect with {meaning}."
    return f"The survey treated {word} as a concrete issue after residents repeatedly described {meaning}."


def _fallback_chinese_sentence(entry: dict[str, Any]) -> str:
    word = str(entry["word"])
    known = FALLBACK_CHINESE_SENTENCES.get(word.lower())
    if known:
        return known
    phrases = "、".join(chinese_phrases_for_entry(entry)[:2]) or vt.short_meaning(entry)
    pos = _entry_pos(entry)
    if pos == "adj":
        return f"委员会用 {word} 描述一种反复出现的状态，并把它和“{phrases}”这样的住户反馈联系起来。"
    if pos in {"v", "vt", "vi"}:
        return f"修订后的政策需要用 {word} 处理这个问题，使住户能把措施和“{phrases}”联系起来。"
    return f"在住户反复提到“{phrases}”之后，调查把 {word} 视为一个具体问题。"


def local_academic_practice(entries: list[dict[str, Any]]) -> dict[str, str]:
    words = [str(entry["word"]) for entry in entries]
    lower_set = {word.lower() for word in words}
    if {"abate", "abbey", "abandon", "abbreviate", "aberrant", "abhor", "abiding"}.issubset(lower_set):
        return {
            "english": (
                "At an old abbey, a research team refused to abandon its climate survey when an aberrant "
                "temperature pattern appeared in the records. To abate confusion, the lead scholar asked "
                "students to abbreviate routine notes but preserve every unusual detail. The group came to "
                "abhor careless shortcuts, and their abiding patience finally turned a puzzling anomaly into "
                "a publishable academic case study."
            ),
            "chinese": (
                "在一座古老的大修道院里，一个研究团队在记录中发现脱离常轨的温度模式后，并没有离弃气候调查。"
                "为了减轻混乱，首席学者要求学生使常规笔记简短，但保留每一个异常细节。团队逐渐痛恨草率的捷径，"
                "而他们持久的耐心最终把一个令人困惑的异常现象变成了可以发表的学术案例。"
            ),
        }

    english = (
        "During a campus policy study, researchers examined how neglected facilities affected student life. "
        + " ".join(_fallback_sentence(entry) for entry in entries)
        + " The final recommendation asked the university to repair the building, publish maintenance data, and respond to residents before minor problems became public conflicts."
    )
    chinese = (
        "在一次校园政策研究中，研究者考察了被忽视的设施如何影响学生生活。"
        + "".join(_fallback_chinese_sentence(entry) for entry in entries)
        + "最终建议要求学校维修建筑、公开维护数据，并在小问题演变成公共冲突前回应住户。"
    )
    return {"english": english, "chinese": chinese}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class LexiPilotRuntime:
    model_name: str = field(default_factory=lambda: os.getenv("RADEON_MODEL", "Qwen/Qwen3-8B").strip() or "Qwen/Qwen3-8B")
    endpoint_type: str = field(default_factory=lambda: (os.getenv("ENDPOINT_TYPE", "dedicated").strip() or "dedicated").lower())
    base_url: str = field(default_factory=lambda: normalize_base_url(os.getenv("RADEON_BASE_URL", "")))
    api_key: str = field(default_factory=lambda: os.getenv("RADEON_API_KEY", ""))
    enable_thinking: bool = field(
        default_factory=lambda: parse_env_bool(os.getenv("QWEN_ENABLE_THINKING"), default=False)
    )
    performance_reports_enabled: bool = field(
        default_factory=lambda: parse_env_bool(os.getenv("PERFORMANCE_REPORTS_ENABLED"), default=True)
    )
    model_request_count: int = 0
    model_request_durations: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    story_generation_duration: float = 0.0
    radeon_planning_succeeded: bool = False
    radeon_story_succeeded: bool = False

    def dedicated_extra_body(self) -> dict[str, Any] | None:
        if self.endpoint_type == "dedicated" and not self.enable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None


class LexiPilotToolbox:
    def __init__(
        self,
        *,
        index_path: Path | str | None = None,
        progress_dir: Path | str | None = None,
        state_file: Path | str | None = None,
        report_dir: Path | str | None = None,
        material_dir: Path | str | None = None,
        runtime: LexiPilotRuntime | None = None,
    ) -> None:
        self.index_path = Path(index_path) if index_path else vt.INDEX_FILE
        self.progress_dir = Path(progress_dir) if progress_dir else vt.PROGRESS_DIR
        self.state_file = Path(state_file) if state_file else vt.STATE_FILE
        self.report_dir = Path(report_dir) if report_dir else Path("performance_reports")
        self.material_dir = Path(material_dir) if material_dir else Path(".lexipilot_materials")
        self.runtime = runtime or LexiPilotRuntime()
        self.tool_events: list[dict[str, Any]] = []
        self._path_lock = threading.RLock()

    @contextmanager
    def trainer_paths(self) -> Any:
        with self._path_lock:
            old_index, old_progress, old_state = vt.INDEX_FILE, vt.PROGRESS_DIR, vt.STATE_FILE
            old_config = vt.CONFIG
            vt.INDEX_FILE = self.index_path
            vt.PROGRESS_DIR = self.progress_dir
            vt.STATE_FILE = self.state_file
            vt.CONFIG = {}
            try:
                yield
            finally:
                vt.INDEX_FILE = old_index
                vt.PROGRESS_DIR = old_progress
                vt.STATE_FILE = old_state
                vt.CONFIG = old_config

    def _run(self, name: str, func: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        ok = False
        try:
            result = func()
            ok = True
            return result
        except Exception:
            raise
        finally:
            self.tool_events.append({"name": name, "duration": round(time.perf_counter() - start, 4), "ok": ok})

    def load_entries(self) -> list[dict[str, Any]]:
        with self.trainer_paths():
            return vt.load_index(vt.DEFAULT_PDF, rebuild=False)

    def load_state(self, profile: str) -> dict[str, Any]:
        with self.trainer_paths():
            return vt.load_state(profile)

    def save_state_atomic(self, state: dict[str, Any], profile: str) -> Path:
        with self.trainer_paths():
            path = vt.state_path(profile)
        state["profile"] = vt.normalize_profile_name(profile)
        _atomic_write_json(path, state)
        return path

    def find_entry(self, word: str) -> dict[str, Any] | None:
        target = word.strip().lower()
        for entry in self.load_entries():
            if str(entry.get("word", "")).lower() == target:
                return entry
        return None

    def get_profile_summary(self, profile: str) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = self.load_entries()
            state = self.load_state(profile)
            today = date.today().isoformat()
            start_seq = int(state.get("start_seq", 1))
            last_new_seq = int(state.get("last_new_seq", 0))
            scoped_cards = {
                seq: card
                for seq, card in state.get("cards", {}).items()
                if str(seq).isdigit() and start_seq <= int(seq) <= max(last_new_seq, start_seq - 1)
            }
            due_count = sum(1 for card in scoped_cards.values() if card.get("due", today) <= today)
            total_correct = sum(int(card.get("correct", 0)) for card in scoped_cards.values())
            total_seen = sum(int(card.get("seen", 0)) for card in scoped_cards.values())
            stages: dict[str, int] = {}
            for card in scoped_cards.values():
                stage = str(int(card.get("stage", 0)))
                stages[stage] = stages.get(stage, 0) + 1
            max_page = vt.total_pdf_pages(entries)
            last_entry = vt.entry_for_seq(entries, last_new_seq)
            next_entry = vt.first_entry_after_seq(entries, last_new_seq)
            current_page = int((next_entry or last_entry or {"page": state.get("start_page", 1)})["page"])
            return {
                "profile": vt.normalize_profile_name(profile),
                "total_vocabulary_count": len(entries),
                "started_word_count": len(scoped_cards),
                "current_new_word_position": last_new_seq,
                "current_page": current_page,
                "max_page": max_page,
                "reviews_due_today": due_count,
                "recent_study_statistics": {
                    day: state.get("daily_stats", {}).get(day, {})
                    for day in sorted(state.get("daily_stats", {}))[-7:]
                },
                "total_correct_answers": total_correct,
                "total_incorrect_answers": max(0, total_seen - total_correct),
                "spaced_repetition_distribution": stages,
                "progress_path": str(self.state_path(profile)),
            }

        return self._run("get_profile_summary", work)

    def state_path(self, profile: str) -> Path:
        with self.trainer_paths():
            return vt.state_path(profile)

    def get_due_words(self, profile: str, limit: int = 20) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = self.load_entries()
            state = self.load_state(profile)
            due = vt.due_entries(entries, state)[: max(0, int(limit))]
            return {"profile": vt.normalize_profile_name(profile), "words": [_clean_entry(e, state["cards"].get(str(e["seq"]))) for e in due]}

        return self._run("get_due_words", work)

    def get_missed_words(
        self,
        profile: str,
        limit: int = 20,
        date_value: str | None = None,
        highest_first: bool = True,
    ) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = self.load_entries()
            state = self.load_state(profile)
            counts = vt.daily_n_counts(state, date_value) if date_value else vt.all_miss_counts(state)
            entry_by_seq = {int(entry["seq"]): entry for entry in entries}
            rows = []
            for seq, count in counts.items():
                if seq in entry_by_seq and count > 0:
                    card = state.get("cards", {}).get(str(seq), {})
                    item = _clean_entry(entry_by_seq[seq], card)
                    item["missed_count"] = int(count)
                    rows.append(item)
            rows.sort(key=lambda item: ((-1 if highest_first else 1) * int(item["missed_count"]), int(item["seq"])))
            return {"profile": vt.normalize_profile_name(profile), "words": rows[: max(0, int(limit))]}

        return self._run("get_missed_words", work)

    def get_new_words(self, profile: str, limit: int = 10, from_page: int | None = None) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = self.load_entries()
            state = self.load_state(profile)
            if from_page is not None:
                start_seq = vt.page_start_seq(entries, int(from_page))
                probe_state = dict(state)
                probe_state["start_seq"] = start_seq
                probe_state["last_new_seq"] = start_seq - 1
                selected = vt.new_entries(entries, probe_state, int(limit))
            else:
                selected = vt.new_entries(entries, state, int(limit))
            return {"profile": vt.normalize_profile_name(profile), "words": [_clean_entry(entry) for entry in selected]}

        return self._run("get_new_words", work)

    def get_word_details(self, word: str, profile: str | None = None) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entry = self.find_entry(word)
            if entry is None:
                raise ValueError(f"Unknown vocabulary word: {word}")
            card = None
            if profile:
                card = self.load_state(profile).get("cards", {}).get(str(entry["seq"]))
            data = _clean_entry(entry, card)
            data["source_sequence"] = int(entry["seq"])
            data["source_text"] = str(entry.get("source_text", ""))[:500]
            return data

        return self._run("get_word_details", work)

    def record_answer(self, profile: str, word: str, remembered: bool) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entry = self.find_entry(word)
            if entry is None:
                raise ValueError(f"Unknown vocabulary word: {word}")
            state = self.load_state(profile)
            seq = int(entry["seq"])
            existed = str(seq) in state.get("cards", {})
            before = dict(state.get("cards", {}).get(str(seq), {}))
            mode = "review" if existed and seq <= int(state.get("last_new_seq", 0)) else "new"
            with self.trainer_paths():
                vt.mark_answer(state, seq, bool(remembered), mode)
            if mode == "new":
                state["last_new_seq"] = max(int(state.get("last_new_seq", 0)), seq)
            path = self.save_state_atomic(state, profile)
            after = state["cards"][str(seq)]
            return {
                "profile": vt.normalize_profile_name(profile),
                "word": str(entry["word"]),
                "remembered": bool(remembered),
                "mode": mode,
                "previous_stage": int(before.get("stage", 0)) if before else None,
                "review_stage": int(after.get("stage", 0)),
                "due_date": str(after.get("due", "")),
                "correct_count": int(after.get("correct", 0)),
                "missed_count": max(0, int(after.get("seen", 0)) - int(after.get("correct", 0))),
                "progress_path": str(path),
            }

        return self._run("record_answer", work)

    def lookup_etymology(self, word: str) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            try:
                cn_url, cn_lines = vt.fetch_etymology_cn(word)
                url, lines = vt.fetch_etymology(word)
            except SystemExit as exc:
                return {"word": word, "etymology": str(exc), "memory_hints": [], "source": "Etymonline"}
            chosen = cn_lines or lines[:6]
            hint = f"Connect {word} to its origin: {chosen[0]}" if chosen else f"No concise etymology found for {word}."
            return {
                "word": word,
                "etymology": " ".join(chosen)[:1200],
                "memory_hints": [hint[:300]],
                "source": "Etymonline",
            }

        return self._run("lookup_etymology", work)

    def _call_radeon_story(self, entries: list[dict[str, Any]], style: str, include_translation: bool) -> dict[str, Any] | None:
        if self.runtime.endpoint_type != "dedicated" or not self.runtime.base_url or not self.runtime.api_key:
            return None
        words = [{"word": e["word"], "definition": vt.plain_definition(e.get("definition") or "")} for e in entries]
        prompt = (
            f"Write a short {style} vocabulary passage using every target word exactly once or more. "
            "Return strict JSON with keys english, chinese, target_translations, notes. "
            "target_translations must map every target word to an array containing every complete Chinese phrase "
            "that translates its use in chinese. Copy each phrase verbatim from chinese, use at least two Chinese "
            "characters per phrase, and include distinct phrases when a target is translated more than once. "
            "Notes must be an array of short strings. "
            f"Chinese translation required: {include_translation}.\nTargets:\n{json.dumps(words, ensure_ascii=False)}"
        )
        payload: dict[str, Any] = {
            "model": self.runtime.model_name,
            "messages": [
                {"role": "system", "content": "You create concise academic vocabulary practice material. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
        }
        extra = self.runtime.dedicated_extra_body()
        if extra:
            payload["extra_body"] = extra
        request = urllib.request.Request(
            f"{self.runtime.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.runtime.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        self.runtime.model_request_count += 1
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self.runtime.model_request_durations.append(round(time.perf_counter() - start, 4))
            return None
        self.runtime.model_request_durations.append(round(time.perf_counter() - start, 4))
        usage = data.get("usage") if isinstance(data, dict) else {}
        if isinstance(usage, dict):
            self.runtime.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.runtime.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        parsed = parse_radeon_story_payload(vt.response_text(data), entries)
        if not parsed:
            return None
        self.runtime.radeon_story_succeeded = True
        return {
            "english": parsed["english"],
            "chinese": parsed["chinese"],
            "target_translations": parsed["target_translations"],
            "notes": [],
        }

    def generate_practice_story(
        self,
        profile: str,
        words: list[str],
        style: str = "academic",
        include_translation: bool = True,
    ) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = []
            for word in words:
                entry = self.find_entry(word)
                if entry is None:
                    raise ValueError(f"Unknown vocabulary word: {word}")
                entries.append(entry)
            start = time.perf_counter()
            generated = self._call_radeon_story(entries, style, include_translation)
            source = "radeon-dedicated" if generated else "local-fallback"
            if generated is None and parse_env_bool(os.getenv("LEXIPILOT_REMOTE_FALLBACK"), default=False):
                generated = vt.generate_snap_for_entries(entries)
            if generated is None:
                generated = local_academic_practice(entries)
            self.runtime.story_generation_duration += round(time.perf_counter() - start, 4)
            target_words = [str(entry["word"]) for entry in entries]
            target_phonetics = {
                str(entry["word"]): vt.display_phonetic(entry) for entry in entries
            }
            target_parts_of_speech = {
                str(entry["word"]): _entry_pos(entry) for entry in entries
            }
            state = self.load_state(profile)
            target_review_stages = {
                str(entry["word"]): int(state.get("cards", {}).get(str(entry["seq"]), {}).get("stage", 0))
                for entry in entries
            }
            fallback_target_translations = {
                str(entry["word"]): chinese_phrases_for_entry(entry) for entry in entries
            }
            generated_target_translations = generated.get("target_translations")
            target_translations = (
                {
                    str(entry["word"]): [
                        str(phrase)
                        for phrase in generated_target_translations.get(str(entry["word"]), [])
                    ]
                    for entry in entries
                }
                if isinstance(generated_target_translations, dict)
                else fallback_target_translations
            )
            material = {
                "profile": vt.normalize_profile_name(profile),
                "created_at": _iso_now(),
                "style": style,
                "target_words": target_words,
                "target_phonetics": target_phonetics,
                "target_parts_of_speech": target_parts_of_speech,
                "target_review_stages": target_review_stages,
                "target_translations": target_translations,
                "english_passage": generated.get("english", ""),
                "chinese_translation": generated.get("chinese", "") if include_translation else "",
                "english": generated.get("english", ""),
                "chinese": generated.get("chinese", "") if include_translation else "",
                "highlighted_target_words": target_words,
                "vocabulary_notes": [
                    f"{entry['word']}: {vt.short_meaning(entry)}" for entry in entries
                ],
                "source": source,
            }
            path = self.material_dir / vt.normalize_profile_name(profile) / f"practice_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
            _atomic_write_json(path, material)
            material["path"] = str(path)
            return material

        return self._run("generate_practice_story", work)

    def next_review_summary(self, profile: str) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            state = self.load_state(profile)
            today = date.today()
            due_dates = []
            for card in state.get("cards", {}).values():
                due = card.get("due")
                if isinstance(due, str):
                    try:
                        due_dates.append(date.fromisoformat(due))
                    except ValueError:
                        continue
            if not due_dates:
                return {"status": "none", "message": "no scheduled review", "due_date": None}
            earliest = min(due_dates)
            if earliest <= today:
                return {"status": "today", "message": "today - one or more words remain due today", "due_date": earliest.isoformat()}
            if earliest == today + timedelta(days=1):
                return {"status": "tomorrow", "message": "tomorrow", "due_date": earliest.isoformat()}
            return {"status": "future", "message": earliest.isoformat(), "due_date": earliest.isoformat()}

        return self._run("next_review_summary", work)

    def get_words_by_page(self, page: int) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            entries = self.load_entries()
            page_entries = vt.page_entries_for_arg(entries, int(page))
            return {"page": int(page), "words": [_clean_entry(entry) for entry in page_entries]}

        return self._run("get_words_by_page", work)

    def save_session_summary(self, **summary: Any) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            safe = {
                "profile": vt.normalize_profile_name(str(summary.get("profile", vt.DEFAULT_PROFILE))),
                "start_timestamp": summary.get("start_timestamp"),
                "end_timestamp": summary.get("end_timestamp") or _iso_now(),
                "planned_words": list(summary.get("planned_words", [])),
                "reviewed_words": list(summary.get("reviewed_words", [])),
                "new_words": list(summary.get("new_words", [])),
                "correct_count": int(summary.get("correct_count", 0)),
                "incorrect_count": int(summary.get("incorrect_count", 0)),
                "frequently_missed_words": list(summary.get("frequently_missed_words", [])),
                "generated_material_path": summary.get("generated_material_path"),
                "next_recommended_review_time": summary.get("next_recommended_review_time"),
            }
            path = self.progress_dir / safe["profile"] / "agent_sessions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(safe, ensure_ascii=False)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            return {"saved": True, "path": str(path), "summary": safe}

        return self._run("save_session_summary", work)

    def write_performance_report(
        self,
        final_session_state: dict[str, Any],
        started_at: float,
        timings: dict[str, float] | None = None,
    ) -> Path | None:
        if not self.runtime.performance_reports_enabled:
            return None
        timing_payload = timings or {}
        payload = {
            "created_at": _iso_now(),
            "model_name": self.runtime.model_name,
            "endpoint_type": self.runtime.endpoint_type,
            "timing_semantics": (
                "Timing fields are not all additive. Model request time is included in tool time "
                "when a tool invokes the model. Story generation time is a subset of tool time. "
                "Use non_overlapping_timing_breakdown for top-level additive components."
            ),
            "total_task_duration": round(time.perf_counter() - started_at, 4),
            "session_wall_seconds": timing_payload.get("session_wall_seconds"),
            "user_interaction_wait_seconds": timing_payload.get("user_interaction_wait_seconds"),
            "active_system_seconds": timing_payload.get("active_system_seconds"),
            "model_request_seconds": timing_payload.get("model_request_seconds"),
            "tool_execution_seconds": timing_payload.get("tool_execution_seconds"),
            "story_generation_seconds": timing_payload.get("story_generation_seconds"),
            "planning_seconds": timing_payload.get("planning_seconds"),
            "finalization_seconds": timing_payload.get("finalization_seconds"),
            "non_overlapping_timing_breakdown": {
                "user_interaction_wait_seconds": timing_payload.get("non_overlapping_user_interaction_wait_seconds"),
                "model_api_execution_seconds": timing_payload.get("non_overlapping_model_api_execution_seconds"),
                "local_non_model_processing_seconds": timing_payload.get("non_overlapping_local_processing_seconds"),
            },
            "model_request_count": self.runtime.model_request_count,
            "model_request_durations": self.runtime.model_request_durations,
            "prompt_tokens": self.runtime.prompt_tokens,
            "completion_tokens": self.runtime.completion_tokens,
            "tool_call_count": len(self.tool_events),
            "tool_durations": self.tool_events,
            "story_generation_duration": self.runtime.story_generation_duration,
            "final_session_state": final_session_state,
            "radeon_planning_succeeded": self.runtime.radeon_planning_succeeded,
            "radeon_inference_succeeded": self.runtime.radeon_story_succeeded,
        }
        path = self.report_dir / f"lexipilot_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
        _atomic_write_json(path, payload)
        return path


def openai_tool_schemas() -> list[dict[str, Any]]:
    def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    return [
        tool("get_profile_summary", "Summarize learner progress without returning the full progress file.", {"profile": {"type": "string"}}, ["profile"]),
        tool("get_due_words", "Return words due for review today.", {"profile": {"type": "string"}, "limit": {"type": "integer"}}, ["profile"]),
        tool("get_missed_words", "Return frequently missed words.", {"profile": {"type": "string"}, "limit": {"type": "integer"}, "date": {"type": ["string", "null"]}, "highest_first": {"type": "boolean"}}, ["profile"]),
        tool("get_new_words", "Return next unlearned words without changing progress.", {"profile": {"type": "string"}, "limit": {"type": "integer"}, "from_page": {"type": ["integer", "null"]}}, ["profile"]),
        tool("get_word_details", "Return details for one vocabulary word.", {"word": {"type": "string"}, "profile": {"type": ["string", "null"]}}, ["word"]),
        tool("record_answer", "Record an explicit learner answer using spaced repetition.", {"profile": {"type": "string"}, "word": {"type": "string"}, "remembered": {"type": "boolean"}}, ["profile", "word", "remembered"]),
        tool("lookup_etymology", "Look up concise Etymonline notes.", {"word": {"type": "string"}}, ["word"]),
        tool("generate_practice_story", "Generate personalized academic vocabulary practice material.", {"profile": {"type": "string"}, "words": {"type": "array", "items": {"type": "string"}}, "style": {"type": "string"}, "include_translation": {"type": "boolean"}}, ["profile", "words"]),
        tool("get_words_by_page", "Return vocabulary entries from a PDF page.", {"page": {"type": "integer"}}, ["page"]),
        tool("save_session_summary", "Persist a concise privacy-safe session summary.", {}, []),
    ]


PLANNING_TOOL_NAMES = frozenset(
    {
        "get_profile_summary",
        "get_due_words",
        "get_missed_words",
        "get_new_words",
        "get_word_details",
    }
)


def planning_tool_schemas() -> list[dict[str, Any]]:
    """Return the read-only tools permitted during model planning."""
    return [
        schema
        for schema in openai_tool_schemas()
        if schema.get("function", {}).get("name") in PLANNING_TOOL_NAMES
    ]


def execute_tool(toolbox: LexiPilotToolbox, name: str, arguments: dict[str, Any]) -> Any:
    mapping: dict[str, Callable[..., Any]] = {
        "get_profile_summary": toolbox.get_profile_summary,
        "get_due_words": toolbox.get_due_words,
        "get_missed_words": lambda profile, limit=20, date=None, highest_first=True: toolbox.get_missed_words(profile, limit, date, highest_first),
        "get_new_words": toolbox.get_new_words,
        "get_word_details": toolbox.get_word_details,
        "record_answer": toolbox.record_answer,
        "lookup_etymology": toolbox.lookup_etymology,
        "generate_practice_story": toolbox.generate_practice_story,
        "get_words_by_page": toolbox.get_words_by_page,
        "save_session_summary": toolbox.save_session_summary,
    }
    if name not in mapping:
        raise ValueError(f"Unknown tool: {name}")
    return mapping[name](**arguments)
