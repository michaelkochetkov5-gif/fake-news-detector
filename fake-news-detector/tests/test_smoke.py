"""Быстрые проверки логики без сети и без модели NLI.

Запуск: python3 -m pytest tests/ -q   (или python3 tests/test_smoke.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nli_checker as nli  # noqa: E402


def test_number_units_known_units():
    assert nli._number_units("100 градусах Цельсия") == [("100", "\u00b0c")]
    assert nli._number_units("212 \u00b0F") == [("212", "\u00b0f")]
    assert nli._number_units("40 тысяч километров") == [("40", "тыс")]
    assert nli._number_units("101 мегапаскаля") == [("101", "мпа")]
    assert nli._number_units("1,2 километра") == [("1.2", "км")]
    assert nli._number_units("8 процентов") == [("8", "%")]


def test_number_units_ignores_plain_words():
    # Раньше любое слово после числа считалось единицей и давало ложное противоречие.
    assert nli._number_units("Задержано 15 рейсов") == []
    assert nli._number_units("151 утверждение") == []
    assert nli._number_units("два часа") == []


def test_numbers_conflict_same_unit_only():
    assert nli._numbers_conflict("100 °C", "вода кипит при 50 °C") is True
    assert nli._numbers_conflict("100 °C", "вода кипит при 100 °C") is False
    # разные шкалы не сравниваются между собой
    assert nli._numbers_conflict("0 °C", "212 °F") is False


def test_combine_without_flud():
    fact = {"verdict": nli.VERDICT_UNKNOWN, "confidence": 0.5}
    assert nli.combine(fact, None)["verdict"] == nli.VERDICT_BIG_DOUBTS


def test_combine_agreement():
    fact = {"verdict": nli.VERDICT_TRUE, "confidence": 0.95}
    flud = {"verdict": nli.VERDICT_TRUE, "confidence": 0.90}
    assert nli.combine(fact, flud)["verdict"] == nli.VERDICT_TRUE


def test_logic_version_present():
    assert isinstance(nli.LOGIC_VERSION, str) and nli.LOGIC_VERSION


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
