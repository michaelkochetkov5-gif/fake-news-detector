"""Numeric checker: извлечение, нормализация и сравнение чисел.

Модуль выдаёт только признаки и корректирует P(ФЕЙК) NLI в зоне неуверенности.
Вердикт не выносит, метки и changed_detail не читает.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

LETTER = "A-Za-zА-Яа-яЁё"
_NOT_LETTER = "(?![" + LETTER + "])"

# Априорные константы коррекции NLI. НЕ подобраны по датасету.
GATE = 0.9
SHRINK = 0.5

_NUM_CAND = re.compile(
    r"(?<![\d.,])"
    r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d{3})*(?:[.,]\d+)?)"
    r"(?!\d)"
)

_YEAR_RANGE = re.compile(r"(?<!\d)((?:1\d|20)\d{2})\s*[–-]\s*(\d{2})(?!\d)")

_MULT = [
    (r"million|миллион\w*|млн\.?", Decimal(10) ** 6),
    (r"billion|миллиард\w*|млрд\.?", Decimal(10) ** 9),
    (r"thousand|тысяч\w*|тыс\.?", Decimal(10) ** 3),
]

_PCT = r"%|процент\w*|per\s?cent|percent"

_TEMP = [
    (r"[°º]\s?[C\u0421]|degrees?\s+celsius|градус\w*\s+цельсия", "C"),
    (r"[°º]\s?F|degrees?\s+fahrenheit|градус\w*\s+фаренгейта", "F"),
]

_UNITS = [
    (r"g/cm(?:3|³)|г/см(?:3|³)", "g/cm3", "density", Decimal(1)),
    (r"km|км|kilomet(?:er|re)s?|километр(?:а|ов|ы)?", "km", "length", Decimal(1000)),
    (r"cm|см|centimet(?:er|re)s?|сантиметр(?:а|ов|ы)?", "cm", "length", Decimal("0.01")),
    (r"mm|мм", "mm", "length", Decimal("0.001")),
    (r"miles?|миль|мили|миля|миле", "mi", "length", Decimal("1609.344")),
    (r"feet|foot|ft|футов|фута|фут", "ft", "length", Decimal("0.3048")),
    (r"meters?|metres?|метр(?:а|ов)?|m|м", "m", "length", Decimal(1)),
    (r"kg|кг|kilograms?|килограмм(?:а|ов)?", "kg", "mass", Decimal(1)),
    (r"grams?|граммов?|грамма?|g|г", "g", "mass", Decimal("0.001")),
    (r"pounds?|lbs?|фунт(?:а|ов)?", "lb", "mass", Decimal("0.45359237")),
    (r"tonnes?|tons?|тонн(?:а|ы)?|т", "t", "mass", Decimal(1000)),
]

_MONEY = [
    (r"USD|dollars?|долл\w*", "USD"),
    (r"EUR|euros?|евро", "EUR"),
    (r"RUB|рубл\w*|руб\.?", "RUB"),
]
_MONEY_SYMBOL = {"$": "USD", "€": "EUR", "₽": "RUB", "£": "GBP"}

_CONVERTIBLE = {"length", "mass", "temp"}
_MATCH = {"exact", "rounded", "derived", "percent_equiv"}
_MATCH_ORDER = ["exact", "rounded", "derived", "percent_equiv"]

_PATTERN_CACHE: dict = {}


def _match_at(pattern: str, text: str, pos: int):
    rx = _PATTERN_CACHE.get(pattern)
    if rx is None:
        rx = re.compile(r"\s*(?:" + pattern + r")" + _NOT_LETTER, re.IGNORECASE)
        _PATTERN_CACHE[pattern] = rx
    return rx.match(text, pos)


@dataclass
class Num:
    literal: str
    value: Decimal
    res: Decimal
    decimals: int
    unit: Optional[str] = None
    dim: Optional[str] = None
    base: Optional[Decimal] = None
    base_res: Optional[Decimal] = None
    percent: bool = False
    identifier: bool = False
    derived: bool = False
    ambiguous_comma: bool = False
    start: int = 0
    side: str = ""

    @property
    def salient(self) -> bool:
        return bool(
            self.unit or self.percent or self.derived
            or self.decimals > 0 or self.value >= 100
        )


def _parse_literal(lit: str):
    s = re.sub(r"[ \u00a0\u202f]", "", lit)

    if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", s):
        s2 = s.replace(",", "")
        return s2, len(s.split(".")[1]), False

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", s):
        s2 = s.replace(".", "").replace(",", ".")
        return s2, len(s2.split(".")[1]), False

    if re.fullmatch(r"\d+(?:\.\d{3}){2,}", s):
        return s.replace(".", ""), 0, False

    if re.fullmatch(r"\d+(?:,\d{3})+", s) and not s.startswith("0,"):
        return s.replace(",", ""), 0, s.count(",") == 1

    if re.fullmatch(r"\d+,\d+", s):
        ip, fp = s.split(",")
        return f"{ip}.{fp}", len(fp), False

    if re.fullmatch(r"\d+\.\d+", s):
        return s, len(s.split(".")[1]), False

    if re.fullmatch(r"\d+", s):
        return s, 0, False

    raise ValueError(f"не удалось разобрать число: {lit!r}")


def _read_suffix(text: str, start: int, end: int) -> dict:
    info = {"percent": False, "mult": None, "unit": None, "dim": None, "factor": None}

    before = text[max(0, start - 2):start].rstrip()
    for sym, cur in _MONEY_SYMBOL.items():
        if before.endswith(sym):
            info["unit"], info["dim"] = cur, "money"

    if _match_at(_PCT, text, end):
        info["percent"] = True
        info["unit"], info["dim"] = "%", "percent"
        return info

    pos = end
    for pat, mult in _MULT:
        m = _match_at(pat, text, pos)
        if m:
            info["mult"] = mult
            pos = m.end()
            break

    for pat, unit in _TEMP:
        if _match_at(pat, text, pos):
            info.update(unit=unit, dim="temp")
            return info

    for pat, canon, dim, factor in _UNITS:
        if _match_at(pat, text, pos):
            info.update(unit=canon, dim=dim, factor=factor)
            return info

    for pat, cur in _MONEY:
        if _match_at(pat, text, pos):
            info.update(unit=cur, dim="money")
            return info

    return info


def extract_numbers(text: str, side: str = "text") -> list:
    text = str(text or "")
    out: list = []
    skip = set()

    for m in _YEAR_RANGE.finditer(text):
        skip.add(m.start(2))
        y = int(m.group(1)[:2] + m.group(2))
        out.append(Num(
            literal=m.group(0), value=Decimal(y), res=Decimal(1), decimals=0,
            derived=True, start=m.start(), side=side,
        ))

    for m in _NUM_CAND.finditer(text):
        if m.start() in skip:
            continue
        try:
            lit, decimals, ambiguous = _parse_literal(m.group(0))
        except ValueError:
            continue

        value = Decimal(lit)
        info = _read_suffix(text, m.start(), m.end())
        mult = info["mult"] or Decimal(1)
        value_final = value * mult
        res = Decimal(1).scaleb(-decimals) * mult

        unit, dim = info["unit"], info["dim"]
        if dim == "temp":
            if unit == "F":
                base = (value_final - 32) * 5 / 9
                base_res = res * 5 / 9
            else:
                base, base_res = value_final, res
        elif info["factor"] is not None:
            base = value_final * info["factor"]
            base_res = res * info["factor"]
        else:
            base, base_res = value_final, res

        prev = text[m.start() - 1] if m.start() > 0 else " "
        identifier = prev.isalpha() or (
            prev == "-" and m.start() > 1 and text[m.start() - 2].isalpha()
        )

        out.append(Num(
            literal=m.group(0), value=value_final, res=res, decimals=decimals,
            unit=unit, dim=dim, base=base, base_res=base_res,
            percent=info["percent"], identifier=identifier,
            ambiguous_comma=ambiguous, start=m.start(), side=side,
        ))

    out.sort(key=lambda n: n.start)
    return out


def _pair_status(c: Num, s: Num) -> Optional[str]:
    if c.identifier != s.identifier:
        return None

    if c.percent != s.percent:
        pct, plain = (c, s) if c.percent else (s, c)
        if plain.unit is not None:
            return None
        tol = max(pct.res, plain.res * 100) / 2
        if plain.value <= 1:
            ok = abs(plain.value * 100 - pct.value) < tol
        else:
            ok = abs(plain.value - pct.value) < tol
        if ok:
            return "percent_equiv"
        return "mismatch" if (plain.salient and pct.salient) else None

    # Разные единицы: конверсия не используется для совпадения, статус нейтральный.
    if c.unit and s.unit and c.unit != s.unit:
        return "unit_mismatch"

    if (c.unit is None or s.unit is None) and not (c.salient and s.salient):
        return "exact" if c.value == s.value else None

    tol = max(c.res, s.res) / 2
    diff = abs(c.value - s.value)
    if diff == 0:
        return "derived" if (c.derived or s.derived) else "exact"
    if diff < tol:
        return "rounded"
    return "mismatch"


def _classify(c: Num, sources: list):
    found = []
    for s in sources:
        st = _pair_status(c, s)
        if st:
            found.append((st, s))

    for want in _MATCH_ORDER:
        for st, s in found:
            if st == want:
                return st, s
    for want in ("mismatch", "unit_mismatch"):
        for st, s in found:
            if st == want:
                return st, s
    return "absent", None


def _features(c_nums: list, s_nums: list, matches: list) -> dict:
    counted = [m for m in matches if not m["identifier"]]
    st = [m["status"] for m in counted]
    n_match = sum(1 for x in st if x in _MATCH)
    n_mismatch = st.count("mismatch")
    denom = n_match + n_mismatch
    return {
        "n_claim_nums": len(counted),
        "n_source_nums": sum(1 for s in s_nums if not s.identifier),
        "n_matched": n_match,
        "n_exact": st.count("exact"),
        "n_rounded": st.count("rounded"),
        "n_derived": st.count("derived"),
        "n_percent_equiv": st.count("percent_equiv"),
        "n_mismatch": n_mismatch,
        "n_unit_mismatch": st.count("unit_mismatch"),
        "n_absent": st.count("absent"),
        "n_ambiguous_comma": sum(1 for n in c_nums + s_nums if n.ambiguous_comma),
        "match_ratio": round(n_match / denom, 4) if denom else None,
    }


def check_numbers(claim: str, source_text: str) -> dict:
    c_nums = extract_numbers(claim, "claim")
    s_nums = extract_numbers(source_text, "source")

    matches = []
    for c in c_nums:
        status, src = _classify(c, s_nums)
        matches.append({
            "claim": c.literal,
            "unit": c.unit,
            "identifier": c.identifier,
            "status": status,
            "source": src.literal if src is not None else None,
        })

    return {
        "claim_numbers": [_num_dict(n) for n in c_nums],
        "source_numbers": [_num_dict(n) for n in s_nums],
        "matches": matches,
        "features": _features(c_nums, s_nums, matches),
    }


def _num_dict(n: Num) -> dict:
    return {
        "literal": n.literal,
        "value": str(n.value),
        "unit": n.unit,
        "percent": n.percent,
        "identifier": n.identifier,
        "derived": n.derived,
        "ambiguous_comma": n.ambiguous_comma,
        "salient": n.salient,
    }


def adjust_nli_confidence(p_contradiction: float, features: dict,
                          gate: float = GATE, shrink: float = SHRINK) -> dict:
    """Коррекция P(ФЕЙК) только по правилу disjoint.

    Правило all_matched отключено: совпадение чисел не доказывает правдивость
    (см. stage3_numeric_analysis.md, §4).
    """
    p = float(p_contradiction)
    if max(p, 1.0 - p) >= gate:
        return {"p_contradiction": round(p, 4), "rule": "gate_off", "changed": False}

    if features.get("n_claim_nums", 0) == 0:
        return {"p_contradiction": round(p, 4), "rule": "no_claim_numbers", "changed": False}

    if features.get("n_mismatch", 0) > 0 and features.get("n_matched", 0) == 0:
        return {"p_contradiction": round(p + (1.0 - p) * shrink, 4),
                "rule": "disjoint", "changed": True}

    return {"p_contradiction": round(p, 4), "rule": "no_change", "changed": False}
