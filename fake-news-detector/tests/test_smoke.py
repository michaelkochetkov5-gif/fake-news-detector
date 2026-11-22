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


def test_numbers_bound_to_object():
    """Число сравнивается только с величиной того же объекта (J03, K02)."""
    assert nli._numbers_conflict(
        "Длина экватора Земли около 40 тысяч километров.",
        "Длина этой границы 217 километров.",
    ) is False
    assert nli._numbers_conflict(
        "Температура плавления льда 0 градусов Цельсия.",
        "или 212,0 \u00b0F Температура кипения воды :",
    ) is False
    assert nli._numbers_conflict(
        "В сутках 48 часов.", "В сутках 24 часа.",
    ) is True


def test_relation_swap():
    """Перестановка ролей ловится как противоречие (C16, H16, I16, L18)."""
    assert nli._relation_conflict(
        "В процессе фотосинтеза растения поглощают кислород и выделяют углекислый газ.",
        "растения поглощают CO2 и выделяют кислород",
    ) is True
    assert nli._relation_conflict(
        "Вода состоит из двух атомов кислорода и одного атома водорода.",
        "молекула которого состоит из двух атомов водорода и одного атома кислорода",
    ) is True
    assert nli._relation_conflict(
        "Ньютон \u2014 единица измерения давления.",
        "единица измерения силы \u2014 ньютон",
    ) is True


def test_relation_no_false_positive():
    assert nli._relation_conflict(
        "Вода состоит из двух атомов водорода и одного атома кислорода.",
        "молекула которого состоит из двух атомов водорода и одного атома кислорода",
    ) is False
    assert nli._relation_conflict(
        "В сутках 24 часа.", "В сутках 24 часа.",
    ) is False


def test_entities_include_first_word():
    assert "\u043d\u0438\u043b" in nli._entity_lemmas("\u041d\u0438\u043b \u0432\u043f\u0430\u0434\u0430\u0435\u0442 \u0432 \u0422\u0438\u0445\u0438\u0439 \u043e\u043a\u0435\u0430\u043d.")
    assert "\u0432" not in nli._entity_lemmas("\u0412 \u041c\u043e\u0441\u043a\u0432\u0435 \u043e\u0442\u043a\u0440\u044b\u043b\u0438 \u043c\u043e\u0441\u0442.")


def test_number_grouping_with_spaces():
    """Разряды через пробел — одно число, а не несколько (расстояние до Луны)."""
    assert nli._quantity_bindings("363 000 \u043a\u043c")[0][0] == 363000.0
    assert nli._quantity_bindings("5 200 000 \u043a\u043c")[0][0] == 5200000.0
    assert len(nli._quantity_bindings("363 000 \u043a\u043c")) == 1


def test_dimension_words_are_not_objects():
    """'расстояние', 'высота' — род величины, а не объект."""
    assert nli._numbers_conflict_strict(
        "Луна находится на расстоянии около 384 тысяч километров от Земли.",
        "на расстоянии 363 000 км минимальное расстояние Луны от Земли",
    ) is False
    assert nli._numbers_conflict_strict(
        "Луна находится на расстоянии около 384 тысяч километров от Земли.",
        "астероид находится на расстоянии 5 200 000 км от Земли",
    ) is False


def test_strict_conflict_requires_same_object():
    assert nli._numbers_conflict_strict(
        "Высота Эвереста около 5000 метров.",
        "Высота Эвереста составляет 8850 метров.",
    ) is True
    assert nli._numbers_conflict_strict(
        "Длина экватора Земли около 40 тысяч километров.",
        "Длина этой границы 217 километров",
    ) is False
    assert nli._numbers_conflict_strict(
        "Температура плавления льда 0 градусов Цельсия.",
        "212 \u00b0F кипение воды",
    ) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
