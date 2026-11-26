# ruff: noqa: RUF001  (zh-TW text uses full-width punctuation on purpose)
"""Short, plain refusal messages for people. `message` stays English and detailed, for models; `message_human` is
for a UI, a phone or a voice, in the configured language."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from .config import LANGS

__all__ = ["LANGS", "humanize", "refusal_text", "suggestion_text"]
Details = dict[str, Any]


def _num(details: Details, key: str) -> float | None:
    v = details.get(key)
    return float(v) if isinstance(v, int | float) and math.isfinite(v) else None


def _short(details: Details, value_key: str) -> float | None:
    """How many millimetres the plan was short of its limit."""
    value, limit = _num(details, value_key), _num(details, "limit_mm")
    if value is None or limit is None:
        return None
    return max(0.0, limit - value)


def _obstacle(details: Details) -> str:
    raw = str(details.get("obstacle", ""))
    m = re.fullmatch(r"obstacle '(.*)'", raw)
    return m.group(1) if m else raw


def _mm(v: float) -> str:
    return f"{max(1, math.ceil(v - 0.05)):d}"  # "short by 0 mm" would read as no problem


Template = Callable[[Details], str]


def _en_table(d: Details) -> str:
    short = _short(d, "clearance_mm")
    if short is None:
        short = _short(d, "z_mm")
    tail = f" (short by {_mm(short)} mm)" if short is not None else ""
    return f"Too close to the table{tail}. Not moving."


def _zh_table(d: Details) -> str:
    short = _short(d, "clearance_mm")
    if short is None:
        short = _short(d, "z_mm")
    tail = f"（還差 {_mm(short)} mm）" if short is not None else ""
    return f"會太靠近桌面{tail}，已停止規劃"


def _en_obstacle(d: Details) -> str:
    short = _short(d, "gap_mm")
    tail = f" (short by {_mm(short)} mm)" if short is not None else ""
    return f"Too close to {_obstacle(d) or 'an obstacle'}{tail}. Not moving."


def _zh_obstacle(d: Details) -> str:
    short = _short(d, "gap_mm")
    tail = f"（還差 {_mm(short)} mm）" if short is not None else ""
    name = _obstacle(d)
    return f"會太靠近{f' {name}' if name else '障礙物'}{tail}，已停止規劃"


TEMPLATES: dict[str, dict[str, Template]] = {
    "en": {
        "JOINT_LIMIT": lambda d: f"Joint {d.get('joint', '')} would go past its limit. Not moving.".replace("  ", " "),
        "TABLE_CLEARANCE": _en_table,
        "TABLE_COLLISION": lambda d: "The arm would hit the table. Not moving.",
        "COLLISION": lambda d: "The arm would bump into something. Not moving.",
        "OBSTACLE_CLEARANCE": _en_obstacle,
        "WORKSPACE": lambda d: f"That is outside the work area ({d.get('axis', '?')} axis). Not moving.",
        "UNREACHABLE": lambda d: "The arm cannot reach that spot.",
        "BAD_TARGET": lambda d: "That request is not valid. Check the values.",
        "BUSY": lambda d: "The arm is already moving. Wait, or press stop.",
        "STOPPED": lambda d: "Stopped.",
        "ENVELOPE_BREACH": lambda d: "The arm left its safe zone and was halted.",
        "INTERNAL_ERROR": lambda d: "Something went wrong inside the controller. You can try again.",
        "PAUSED": lambda d: "Paused. Resume or stop.",
        "NOTHING_TO_RESUME": lambda d: "Nothing is paused.",
        "PLAN_UNKNOWN": lambda d: "That preview is not known. Preview again.",
        "PLAN_EXPIRED": lambda d: "That preview is too old. Preview again.",
        "PLAN_USED": lambda d: "That preview already ran. Preview again.",
        "PLAN_STALE": lambda d: "The arm has moved since the preview. Preview again.",
        "POSE_UNKNOWN": lambda d: f"No saved pose called {d.get('name', '')}.".replace("  ", " "),
    },
    "zh-TW": {
        "JOINT_LIMIT": lambda d: f"關節 {d.get('joint', '')} 會超出可動範圍，已停止規劃",
        "TABLE_CLEARANCE": _zh_table,
        "TABLE_COLLISION": lambda d: "會撞到桌面，已停止規劃",
        "COLLISION": lambda d: "手臂會碰到東西，已停止規劃",
        "OBSTACLE_CLEARANCE": _zh_obstacle,
        "WORKSPACE": lambda d: f"超出工作範圍（{d.get('axis', '?')} 軸），已停止規劃",
        "UNREACHABLE": lambda d: "手臂搆不到那個位置",
        "BAD_TARGET": lambda d: "指令內容有誤，請檢查數值",
        "BUSY": lambda d: "手臂正在動，請等它停下或按停止",
        "STOPPED": lambda d: "已停止",
        "ENVELOPE_BREACH": lambda d: "手臂實際動作超出安全範圍，已停下",
        "INTERNAL_ERROR": lambda d: "控制器內部出錯，可以再試一次",
        "PAUSED": lambda d: "已暫停，可以繼續或停止",
        "NOTHING_TO_RESUME": lambda d: "目前沒有暫停中的動作",
        "PLAN_UNKNOWN": lambda d: "找不到這個預覽，請重新預覽",
        "PLAN_EXPIRED": lambda d: "預覽已過期，請重新預覽",
        "PLAN_USED": lambda d: "這個預覽已經執行過，請重新預覽",
        "PLAN_STALE": lambda d: "手臂已經不在預覽的起點，請重新預覽",
        "POSE_UNKNOWN": lambda d: f"沒有叫做「{d.get('name', '')}」的姿勢",
    },
}


def refusal_text(code: str, details: Details, lang: str) -> str:
    templates = TEMPLATES.get(lang, TEMPLATES["en"])
    template = templates.get(code)
    if template is None:  # a code without a template still gets a plain sentence
        return "Not moving." if lang == "en" else "已停止"
    return template(details)


def suggestion_text(suggestion: dict[str, Any], lang: str) -> str:
    x, y, z = (suggestion.get(k) for k in ("x_mm", "y_mm", "z_mm"))
    where = f"({x}, {y}, {z}) mm"
    fraction = suggestion.get("fraction")
    if lang == "zh-TW":
        part = f"，約目標的 {fraction:.0%}" if isinstance(fraction, int | float) else ""
        return f"可以改去途中安全的位置 {where}{part}"
    part = f", {fraction:.0%} of the way" if isinstance(fraction, int | float) else ""
    return f"A safe spot on the way is {where}{part}."


def humanize(refusal: dict[str, Any], lang: str) -> dict[str, Any]:
    """Add `message_human` (and `suggestion.human`) to a refusal dict, in place; returns it."""
    refusal["message_human"] = refusal_text(str(refusal.get("code")), refusal.get("details") or {}, lang)
    suggestion = refusal.get("suggestion")
    if isinstance(suggestion, dict):
        suggestion["human"] = suggestion_text(suggestion, lang)  # pyright: ignore[reportUnknownArgumentType]
    return refusal
