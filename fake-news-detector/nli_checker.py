"""FLUD: NLI zero-shot + numeric checker — CPU fallback для сайта."""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from main_code import claim_key_terms, select_terms

# Морфология нужна только для привязки чисел и ролей. Импорт необязательный:
# в тестах main_code может быть подменён заглушкой без этих имён.
try:
    from main_code import MORPH_AVAILABLE, _MORPH
except ImportError:  # pragma: no cover
    # Стенд подменяет main_code заглушкой: берём собственную морфологию,
    # иначе правила ведут себя иначе, чем в продукте.
    try:
        import pymorphy3

        _MORPH = pymorphy3.MorphAnalyzer()
        MORPH_AVAILABLE = True
    except Exception:
        _MORPH = None
        MORPH_AVAILABLE = False

VERDICT_TRUE = "ПРАВДА"
VERDICT_FAKE = "ФЕЙК"
VERDICT_UNKNOWN = "НЕИЗВЕСТНО"
VERDICT_BIG_DOUBTS = "БОЛЬШИЕ СОМНЕНИЯ"

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Версия логики входит в ключ кэша сайта: после изменения правил старые
# сохранённые вердикты не должны переиспользоваться.
LOGIC_VERSION = "v10-object-binding"

MIN_CONFIDENCE = 0.50
FACT_MIN_CONFIDENCE = 0.60
FACT_FAKE_MIN_CONFIDENCE = 0.80
FACT_SKIP_CONFIDENCE = 0.85
MIN_SHARED_TERMS = 3
ENTAIL_COVERAGE_MIN = 0.60
NONNUM_CONTRA_COVERAGE_MIN = 1.0
MAX_PREMISES = 8

SOURCE_MESSAGES = {
    "factchecker": (
        "Вердикт основан на проверке источников."
    ),
    "flud": (
        "Вердикт основан на сравнении утверждения "
        "с найденными источниками через NLI "
        "и numeric checker."
    ),
    "factchecker+flud": (
        "Фактчекер и FLUD согласны."
    ),
    "nli": (
        "Вердикт основан на сравнении утверждения "
        "с найденными источниками через NLI "
        "и numeric checker."
    ),
    "nli_no_premise": (
        "FLUD не получила тексты источников от FactChecker."
    ),
    "nli_uncertain": (
        "FLUD недостаточно уверена в вердикте."
    ),
    "nli_unavailable": "FLUD недоступна.",
    "nli_error": (
        "FLUD столкнулась с ошибкой "
        "при анализе источников."
    ),
    "uncertain": (
        "FactChecker и FLUD не смогли дать "
        "уверенный вердикт по найденным источникам."
    ),
}


# Единицы, которые вообще сравнимы между утверждением и цитатой.
# Раньше единицей считалось любое следующее слово, поэтому "151 утверждение"
# и "15 рейсов" превращались в "единицы" и давали ложные противоречия.
UNIT_ALIASES = {
    "%": "%", "процент": "%", "процента": "%", "процентов": "%", "проц": "%",
    "год": "год", "года": "год", "году": "год", "лет": "год", "г": "год", "гг": "год",
    "месяц": "мес", "месяца": "мес", "месяцев": "мес",
    "день": "сут", "дня": "сут", "дней": "сут", "сутки": "сут", "суток": "сут",
    "час": "ч", "часа": "ч", "часов": "ч", "ч": "ч",
    "минута": "мин", "минуты": "мин", "минут": "мин", "мин": "мин",
    "секунда": "с", "секунды": "с", "секунд": "с", "сек": "с", "с": "с",
    "миллиметр": "мм", "миллиметра": "мм", "миллиметров": "мм", "мм": "мм",
    "сантиметр": "см", "сантиметра": "см", "сантиметров": "см", "см": "см",
    "метр": "м", "метра": "м", "метров": "м", "м": "м",
    "километр": "км", "километра": "км", "километров": "км", "км": "км",
    "грамм": "г", "грамма": "г", "граммов": "г", "гр": "г",
    "килограмм": "кг", "килограмма": "кг", "килограммов": "кг", "кг": "кг",
    "тонна": "т", "тонны": "т", "тонн": "т", "т": "т",
    "паскаль": "па", "паскаля": "па", "паскалей": "па", "па": "па",
    "килопаскаль": "кпа", "килопаскаля": "кпа", "килопаскалей": "кпа", "кпа": "кпа",
    "мегапаскаль": "мпа", "мегапаскаля": "мпа", "мегапаскалей": "мпа", "мпа": "мпа",
    "ньютон": "н", "ньютона": "н", "ньютонов": "н", "н": "н",
    "джоуль": "дж", "джоуля": "дж", "джоулей": "дж", "дж": "дж",
    "ватт": "вт", "ватта": "вт", "вт": "вт",
    "тысяча": "тыс", "тысячи": "тыс", "тысяч": "тыс", "тыс": "тыс",
    "миллион": "млн", "миллиона": "млн", "миллионов": "млн", "млн": "млн",
    "миллиард": "млрд", "миллиарда": "млрд", "миллиардов": "млрд", "млрд": "млрд",
    "градус": "\u00b0", "градуса": "\u00b0", "градусов": "\u00b0",
}

NUMBER_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(\u00b0\s*[a-z\u0430-\u044f\u0451]*|[a-z\u0430-\u044f\u0451%]+)"
)


# Основы для падежных форм: точного списка не хватает ("градусах", "километрах").
UNIT_STEMS = (
    ("градус", "\u00b0"), ("процент", "%"), ("килопаскал", "\u043a\u043f\u0430"),
    ("мегапаскал", "\u043c\u043f\u0430"), ("паскал", "\u043f\u0430"),
    ("километр", "\u043a\u043c"), ("миллиметр", "\u043c\u043c"),
    ("сантиметр", "\u0441\u043c"), ("килограмм", "\u043a\u0433"),
    ("миллиард", "\u043c\u043b\u0440\u0434"), ("миллион", "\u043c\u043b\u043d"),
    ("тысяч", "\u0442\u044b\u0441"), ("грамм", "\u0433"), ("\u043c\u0435\u0442\u0440", "\u043c"),
    ("\u043d\u044c\u044e\u0442\u043e\u043d", "\u043d"), ("\u0434\u0436\u043e\u0443\u043b", "\u0434\u0436"),
    ("\u0432\u0430\u0442\u0442", "\u0432\u0442"), ("\u043c\u0438\u043d\u0443\u0442", "\u043c\u0438\u043d"),
    ("\u0441\u0435\u043a\u0443\u043d\u0434", "\u0441"), ("\u0447\u0430\u0441\u043e\u0432", "\u0447"),
    ("\u043c\u0435\u0441\u044f\u0446", "\u043c\u0435\u0441"), ("\u0441\u0443\u0442\u043a", "\u0441\u0443\u0442"),
)


def _normalize_unit(raw_unit: str, tail: str) -> Optional[str]:
    unit = raw_unit.lower().replace("\u0451", "\u0435").replace(" ", "")
    if unit.startswith("\u00b0"):
        base = "\u00b0"
        tail = (unit[1:] + " " + tail).strip()
    else:
        base = UNIT_ALIASES.get(unit)
        if base is None:
            for stem, mapped in UNIT_STEMS:
                if unit.startswith(stem):
                    base = mapped
                    break
    if base is None:
        return None
    if base == "\u00b0":
        if tail.startswith("\u0446\u0435\u043b\u044c\u0441") or tail.startswith("c"):
            return "\u00b0c"
        if tail.startswith("\u0444\u0430\u0440\u0435\u043d") or tail.startswith("f"):
            return "\u00b0f"
    return base


def _number_units(text: str) -> List[tuple]:
    pairs = []
    cleaned = text.lower().replace("\u00a0", " ").replace("*", "")
    for match in NUMBER_UNIT_RE.finditer(cleaned):
        value = match.group(1).replace(",", ".")
        tail = cleaned[match.end():match.end() + 20].strip()
        unit = _normalize_unit(match.group(2), tail)
        if unit is None:
            # Слово после числа не является единицей измерения: сравнивать нечего.
            continue
        pairs.append((value, unit))
    return pairs


# Слова, которые не могут быть объектом величины: связки, предлоги, наречия.
_NOT_OBJECT = {
    "составлять", "составить", "равняться", "быть", "являться", "достигать",
    "около", "примерно", "почти", "более", "менее", "всего", "уже", "ещё",
}


# Слова-связки, которые не являются именованными сущностями.
CAPITAL_STOP = {
    "\u043a\u043e\u0442\u043e\u0440\u044b\u0439", "\u044d\u0442\u043e", "\u0442\u0430\u043a", "\u043a\u0430\u043a",
}

# Служебные слова и частые начала предложений: сущностями не считаются.
FUNCTION_LEMMAS = {
    "\u0432", "\u0432\u043e", "\u043d\u0430", "\u0441", "\u0441\u043e", "\u043f\u043e", "\u0438\u0437", "\u043e\u0442", "\u0434\u043e",
    "\u0437\u0430", "\u043f\u0440\u0438", "\u0434\u043b\u044f", "\u043e", "\u043e\u0431", "\u0447\u0442\u043e", "\u043d\u0435",
    "\u044d\u0442\u043e\u0442", "\u0432\u0435\u0441\u044c", "\u043a\u0430\u0436\u0434\u044b\u0439", "\u043d\u043e\u0432\u044b\u0439",
}


def _entity_lemmas(text: str) -> set:
    """Имена собственные: слова с заглавной буквы без служебных.

    Первое слово предложения тоже учитывается: иначе "Нил" и "Ртуть"
    выпадали из проверки, и цитаты о других объектах проходили.
    """
    raw_words = re.findall(r"[\w\u0430-\u044f\u0451\u0410-\u042f\u0401-]+", text or "")
    entities = set()
    for position, word in enumerate(raw_words):
        if not word[:1].isupper():
            continue
        lemma = _parse(word.lower())[0]
        if not lemma or len(lemma) <= 2:
            continue
        if lemma in CAPITAL_STOP or lemma in FUNCTION_LEMMAS:
            continue
        if MORPH_AVAILABLE and not _is_proper(word):
            # Заглавная буква в первом слове — это начало предложения,
            # а не имя: "Столицей Канады", "Вода состоит".
            continue
        entities.add(lemma)
    return entities


TOKENIZE_RE = re.compile(
    r"\u00b0|[\u0430-\u044f\u0451a-z][\u0430-\u044f\u0451a-z0-9]*|\d+(?:[.,]\d+)?",
    re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


def _lemmas(text: str) -> List[str]:
    """Леммы текста средствами самого модуля.

    Не через main_code.lemma_tokens: стенд подменяет его заглушкой, и фильтры
    начинали работать иначе, чем в продукте.
    """
    return [_parse(token)[0] for token in _tokens(text) if _is_word(token)]


PROPER_TAGS = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}


def _is_proper(word: str) -> bool:
    """Имя собственное по морфологии: Канада, Нил, Цельсий."""
    if not MORPH_AVAILABLE or not word:
        return False
    try:
        tag = _MORPH.parse(word.lower())[0].tag
        return any(flag in tag for flag in PROPER_TAGS)
    except Exception:
        return False


def _is_word(token: str) -> bool:
    return bool(token) and token[0].isalpha()


def _tokens(text: str) -> List[str]:
    return [tok.strip() for tok in TOKENIZE_RE.findall((text or "").lower()) if tok.strip()]


def _parse(word: str):
    """(лемма, часть речи, вид) — минимально нужное для привязки чисел."""
    if not MORPH_AVAILABLE or not word:
        return word, "NOUN", None
    try:
        tag = _MORPH.parse(word)[0].tag
        pos = str(tag.POS)
        return _MORPH.parse(word)[0].normal_form, pos, tag.case
    except Exception:
        return word, "NOUN", None


def _read_number(tokens: List[str], start: int):
    """(значение, индекс после числа) или (None, start)."""
    if start >= len(tokens) or not NUMBER_TOKEN_RE.match(tokens[start]):
        return None, start

    raw = tokens[start].replace(",", ".")
    end = start + 1

    # "363 000" и "5 200 000": пробел разделяет разряды, а не два числа.
    if "." not in raw and len(raw) <= 3:
        parts = [raw]
        while (
            end < len(tokens)
            and NUMBER_TOKEN_RE.match(tokens[end])
            and "." not in tokens[end]
            and len(tokens[end]) == 3
        ):
            parts.append(tokens[end])
            end += 1
        if len(parts) > 1:
            raw = "".join(parts)

    try:
        return float(raw), end
    except ValueError:
        return None, start


SCALES = {"тыс": 1e3, "млн": 1e6, "млрд": 1e9}
APPROX_WORDS = ("около", "примерно", "почти", "приблизительно", "порядка")


def _quantity_bindings(text: str) -> List[tuple]:
    """Возвращает тройки (значение, единица, объект) для каждой величины.

    Объект — ближайшее существительное ПЕРЕД числом: именно оно называет
    величину, к которой число относится ("длина экватора … 40 тысяч км").
    """
    tokens = _tokens(text)
    binds = []
    consumed = 0  # индекс, до которого токены уже вошли в предыдущее число
    for index, token in enumerate(tokens):
        if index < consumed:
            continue
        value, end = _read_number(tokens, index)
        if value is None:
            continue
        consumed = end
        unit = None
        number = value
        if end < len(tokens) and tokens[end] == "\u00b0":
            # "°" идёт отдельным токеном: "100 °C"
            tail = " ".join(tokens[end:end + 3]).lower().replace("\u00b0", "", 1).strip()
            unit = _normalize_unit("\u00b0", tail)
        elif end < len(tokens) and _is_word(tokens[end]):
            # Хвост — слова ПОСЛЕ единицы: "градусов | цельсия", иначе
            # шкала не распознаётся и °C превращается в просто "градусы".
            tail = " ".join(tokens[end + 1:end + 4])
            unit = _normalize_unit(tokens[end].lower(), tail.lower())
            if unit in SCALES:
                # "40 тысяч километров": масштаб относится к следующей единице.
                scale = SCALES[unit]
                following = None
                for nxt in range(end + 1, min(end + 3, len(tokens))):
                    if not _is_word(tokens[nxt]):
                        continue
                    following = _normalize_unit(
                        tokens[nxt].lower(), " ".join(tokens[nxt:nxt + 3]).lower()
                    )
                    if following and following not in SCALES:
                        break
                    following = None
                if following:
                    unit = following
                    number = value * scale
                else:
                    unit = None
        if unit is None:
            continue
        obj = None
        for back in range(index - 1, max(-1, index - 8), -1):
            tok = tokens[back]
            if not _is_word(tok):
                continue
            lemma, pos, _ = _parse(tok.lower())
            if pos != "NOUN" or lemma in _NOT_OBJECT:
                continue
            if lemma in DIMENSION_WORDS:
                # "расстояние", "высота", "площадь" называют род величины,
                # а не объект: ищем дальше имя ("расстояние ... Луны").
                continue
            obj = lemma
            break
        binds.append((number, unit, obj, _is_approx(tokens, index)))
    return binds


def _is_approx(tokens: List[str], index: int) -> bool:
    for back in range(index - 1, max(-1, index - 3), -1):
        if tokens[back].lower().replace("\u0451", "\u0435") in APPROX_WORDS:
            return True
    return False


# Химические формулы: цитата пишет "CO2", утверждение — "углекислый газ".
FORMULA_MAP = {
    "co2": "\u0443\u0433\u043b\u0435\u043a\u0438\u0441\u043b\u044b\u0439", "o2": "\u043a\u0438\u0441\u043b\u043e\u0440\u043e\u0434",
    "h2o": "\u0432\u043e\u0434\u0430", "co": "\u0443\u0433\u0430\u0440\u043d\u044b\u0439",
    "ch4": "\u043c\u0435\u0442\u0430\u043d", "n2": "\u0430\u0437\u043e\u0442",
}

# Слова-счётчики: "два атома кислорода" — величина относится к кислороду.
COUNTER_NOUNS = {
    "\u0430\u0442\u043e\u043c", "\u043c\u043e\u043b\u0435\u043a\u0443\u043b\u0430", "\u0448\u0442\u0443\u043a\u0430",
    "\u0447\u0435\u043b\u043e\u0432\u0435\u043a", "\u0440\u0430\u0437", "\u0442\u0438\u043f",
}


def _norm_token(token: str) -> str:
    return FORMULA_MAP.get(token.lower(), token)


def _verb_objects(text: str) -> Dict[str, set]:
    """Глагол -> существительные-объекты рядом с ним (до 4 токенов)."""
    tokens = _tokens(text)
    frames: Dict[str, set] = {}
    for index, token in enumerate(tokens):
        if not _is_word(token):
            continue
        lemma, pos, _ = _parse(token)
        if pos != "VERB":
            continue
        for step in range(index + 1, min(index + 4, len(tokens))):
            candidate = tokens[step]
            if not _is_word(candidate):
                continue
            obj, obj_pos, _ = _parse(_norm_token(candidate))
            if obj_pos in ("NOUN", "ADJF"):
                # первое существительное или прилагательное = прямое дополнение
                frames.setdefault(lemma, set()).add(obj)
                break
    return frames


NUMERIC_WORDS = {
    "\u043e\u0434\u0438\u043d": 1, "\u043e\u0434\u043d\u0430": 1, "\u043e\u0434\u043d\u043e": 1, "\u0434\u0432\u0430": 2, "\u0434\u0432\u0435": 2,
    "\u0442\u0440\u0438": 3, "\u0447\u0435\u0442\u044b\u0440\u0435": 4, "\u043f\u044f\u0442\u044c": 5, "\u0448\u0435\u0441\u0442\u044c": 6, "\u0441\u0435\u043c\u044c": 7,
    "\u0432\u043e\u0441\u0435\u043c\u044c": 8, "\u0434\u0435\u0432\u044f\u0442\u044c": 9, "\u0434\u0435\u0441\u044f\u0442\u044c": 10,
}


def _count_objects(text: str) -> set:
    """(число, объект) для конструкций "два атома кислорода"."""
    tokens = _tokens(text)
    pairs = set()
    for index, token in enumerate(tokens):
        value, end = _read_number(tokens, index)
        if value is None:
            word_value = NUMERIC_WORDS.get(_parse(token)[0]) if _is_word(token) else None
            if word_value is None:
                continue
            value, end = float(word_value), index + 1
        nouns = []
        for step in range(end, min(end + 4, len(tokens))):
            candidate = tokens[step]
            if not _is_word(candidate):
                continue
            obj, obj_pos, _ = _parse(_norm_token(candidate))
            if obj_pos == "NOUN":
                nouns.append(obj)
                if len(nouns) >= 2:
                    break
        if len(nouns) == 2 and nouns[0] in COUNTER_NOUNS:
            nouns = nouns[1:]
        for obj in nouns:
            pairs.add((value, obj))
    return pairs


DIMENSION_WORDS = {
    "\u0441\u0438\u043b\u0430", "\u0434\u0430\u0432\u043b\u0435\u043d\u0438\u0435", "\u044d\u043d\u0435\u0440\u0433\u0438\u044f",
    "\u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c", "\u043c\u0430\u0441\u0441\u0430", "\u0434\u043b\u0438\u043d\u0430",
    "\u0432\u0440\u0435\u043c\u044f", "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430",
    "\u0440\u0430\u0441\u0441\u0442\u043e\u044f\u043d\u0438\u0435", "\u0432\u044b\u0441\u043e\u0442\u0430", "\u043f\u043b\u043e\u0449\u0430\u0434\u044c",
    "\u043e\u0431\u044a\u0451\u043c", "\u0441\u043a\u043e\u0440\u043e\u0441\u0442\u044c", "\u0433\u043b\u0443\u0431\u0438\u043d\u0430",
    "\u0448\u0438\u0440\u0438\u043d\u0430", "\u0434\u0438\u0430\u043c\u0435\u0442\u0440", "\u0432\u043e\u0437\u0440\u0430\u0441\u0442",
    "\u0432\u0435\u043b\u0438\u0447\u0438\u043d\u0430", "\u0434\u043e\u043b\u044f", "\u043f\u0440\u043e\u0446\u0435\u043d\u0442",
}


def _unit_dimension(text: str) -> Optional[str]:
    """Размерность после слова "измерение": "единица измерения давления"."""
    tokens = _tokens(text)
    for index, token in enumerate(tokens):
        if not _is_word(token):
            continue
        lemma, _, _ = _parse(token)
        if lemma != "\u0438\u0437\u043c\u0435\u0440\u0435\u043d\u0438\u0435":
            continue
        for step in range(index + 1, min(index + 3, len(tokens))):
            candidate = _parse(tokens[step])[0] if tokens[step].isalpha() else None
            if candidate in DIMENSION_WORDS:
                return candidate
    return None


EXCEPTION_WORDS = ("\u043a\u0440\u043e\u043c\u0435", "\u0438\u0441\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435\u043c", "\u0437\u0430 \u0438\u0441\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435\u043c",
                    "\u043d\u0435 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f", "\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0435\u0442", "\u043e\u0442\u043b\u0438\u0447\u0430\u0435\u0442\u0441\u044f")


def _first_noun(text: str) -> Optional[str]:
    for token in _tokens(text):
        if not _is_word(token):
            continue
        lemma, pos, _ = _parse(token)
        if pos == "NOUN":
            return lemma
    return None


def _has_exception(text: str) -> bool:
    low = (text or "").lower()
    return any(word in low for word in EXCEPTION_WORDS)


def _relation_conflict(claim: str, text: str) -> bool:
    """Противоречие по связи объектов, а не по набору слов."""
    if _numbers_conflict(claim, text):
        return True

    # Перестановка ролей у глаголов: "поглощают кислород" против "выделяют кислород".
    claim_frames = _verb_objects(claim)
    text_frames = _verb_objects(text)
    for verb, objects in claim_frames.items():
        if verb not in text_frames:
            continue
        for obj in objects:
            if obj in text_frames[verb]:
                continue
            for other_verb, other_objects in text_frames.items():
                if other_verb != verb and obj in other_objects:
                    return True

    # Перестановка счётчиков: "два атома кислорода" против "два атома водорода".
    claim_counts = _count_objects(claim)
    text_counts = _count_objects(text)
    for count, obj in claim_counts:
        moved = {o for c, o in text_counts if c == count and o != obj}
        if not moved:
            continue
        if any((c, obj) in text_counts for c, _ in claim_counts if c != count):
            return True

    # Разная размерность у одной и той же единицы: ньютон — сила или давление.
    claim_dim = _unit_dimension(claim)
    text_dim = _unit_dimension(text)
    if claim_dim and text_dim and claim_dim != text_dim:
        return True

    # Цитата исключает ровно тот случай, о котором говорит утверждение.
    if _has_exception(text) and not _has_exception(claim):
        subject = _first_noun(claim)
        if subject and subject in set(_lemmas(text)):
            return True

    return False


def _close_numbers(left, right, approx: bool) -> bool:
    if left == right:
        return True
    if not approx:
        return False
    base = max(abs(left), abs(right)) or 1.0
    return abs(left - right) / base <= 0.10


def _same_object(left, right) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    stem = min(len(left), len(right), 5)
    return stem >= 4 and left[:stem] == right[:stem]


def _numbers_conflict_strict(claim: str, text: str) -> bool:
    """Противоречие числа только при совпадении объекта по корню.

    Отличается от _numbers_conflict тем, что не делает поблажек: нужно,
    чтобы объект величины в цитате был тем же, что и в утверждении.
    Используется как сильный признак, достаточный без второго источника.
    """
    text_binds = _quantity_bindings(text)
    for value, unit, obj, approx in _quantity_bindings(claim):
        if not obj:
            continue
        bound = [
            b for b in text_binds
            if b[1] == unit and _same_object(obj, b[2])
        ]
        if not bound:
            continue
        if all(
            not _close_numbers(value, item[0], approx or item[3])
            for item in bound
        ):
            return True
    return False


def _numbers_conflict(claim: str, text: str) -> bool:
    """Противоречие по числу только внутри одной величины.

    Раньше любое число с той же единицей в любом месте цитаты считалось
    противоречием: "экватор ≈ 40 тыс. км" конфликтовал с "граница 217 км".
    Теперь число сравнивается только с величиной того же объекта.
    """
    text_binds = _quantity_bindings(text)
    for value, unit, obj, approx in _quantity_bindings(claim):
        same_unit = [b for b in text_binds if b[1] == unit]
        if not same_unit:
            continue
        if obj:
            bound = [b for b in same_unit if _same_object(obj, b[2])]
            if not bound:
                # Цитата говорит о другой величине — сравнивать нечего.
                continue
            if all(
                not _close_numbers(value, item[0], approx or item[3])
                for item in bound
            ):
                return True
        else:
            # Объект не найден: ведём себя как раньше, но только для
            # величин той же единицы без объекта.
            # Объект не определился: сравниваем со всеми величинами той же
            # единицы, иначе пропустим противоречие в коротких утверждениях.
            if not any(
                _close_numbers(value, b[0], approx or b[3]) for b in same_unit
            ):
                return True
    return False


class NLIChecker:
    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
    ):
        self.model_name = model_name
        self.model = None
        self.load_error = None

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as exc:
            self.load_error = (
                f"FLUD недоступна: "
                f"{type(exc).__name__}: {exc}"
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        return self.available

    def _shared_terms(self, claim: str, text: str) -> int:
        selected = select_terms(claim_key_terms(claim))
        lemmas = set(_lemmas(text))
        return sum(1 for lem, _ in selected if lem in lemmas)

    def _extract_premises(
        self,
        claim: str,
        fact_result: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        needed = min(
            MIN_SHARED_TERMS,
            len(select_terms(claim_key_terms(claim))),
        )
        premises = []
        entities = _entity_lemmas(claim)

        for detail in fact_result.get("details", []):
            reason = str(detail.get("reason", "")).strip()
            source = str(detail.get("source", "")).strip()
            links = detail.get("links") or []
            source_key = str(links[0]) if links else (source or reason[:80])

            if not reason and not source:
                continue

            quotes = re.findall(
                r'["«]([^"»]{10,1000})["»]',
                reason,
            )
            candidates = quotes[:3] if quotes else [
                text for text in (reason, source) if text
            ]

            for text in candidates:
                text = text.strip()
                if len(text) < 10:
                    continue
                if self._shared_terms(claim, text) < needed:
                    continue
                entity_ok = True
                if entities:
                    present = set(_lemmas(text)) | set(
                        _lemmas(str(detail.get("fact", "")))
                    )
                    # Цитата без имени из утверждения не может подтверждать
                    # ("Нил" подменяется "Ди"), но может противоречить.
                    entity_ok = bool(entities & present)
                premises.append(
                    {"text": text, "source": source_key, "entity_ok": entity_ok}
                )

        return premises[:MAX_PREMISES]

    def verify(
        self,
        claim: str,
        fact_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.available:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": f"FLUD недоступна: {self.load_error}",
                "details": [],
                "raw_response": "",
                "decision_source": "nli_unavailable",
            }

        premises = self._extract_premises(claim, fact_result)

        if not premises:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": (
                    "FLUD не нашла цитат, в которых совпадают "
                    "не меньше трёх ключевых терминов утверждения."
                ),
                "details": [],
                "raw_response": "",
                "decision_source": "nli_no_premise",
            }

        try:
            scores = self.model.predict(
                [(p["text"], claim) for p in premises],
                batch_size=1,
                show_progress_bar=False,
            )
        except Exception as exc:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": (
                    f"Ошибка FLUD-инференса: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "details": [],
                "raw_response": "",
                "decision_source": "nli_error",
            }

        scores = np.asarray(scores)
        exp_scores = np.exp(
            scores - scores.max(axis=-1, keepdims=True)
        )
        probabilities = exp_scores / exp_scores.sum(
            axis=-1, keepdims=True
        )
        raw = str(probabilities.tolist())

        n_terms = max(1, len(select_terms(claim_key_terms(claim))))
        by_source: Dict[str, Dict[str, float]] = {}
        for premise, row in zip(premises, probabilities):
            entry = by_source.setdefault(
                premise["source"],
                {"contra": 0.0, "entail": 0.0, "strong_numeric": False},
            )
            coverage = self._shared_terms(claim, premise["text"]) / n_terms
            if _numbers_conflict_strict(claim, premise["text"]):
                entry["strong_numeric"] = True
            if _relation_conflict(claim, premise["text"]) or (
                NONNUM_CONTRA_COVERAGE_MIN
                and coverage >= NONNUM_CONTRA_COVERAGE_MIN
            ):
                entry["contra"] = max(entry["contra"], float(row[0]))
            if coverage >= ENTAIL_COVERAGE_MIN and premise.get("entity_ok", True):
                entry["entail"] = max(entry["entail"], float(row[1]))

        supporting = sorted(
            (e["entail"] for e in by_source.values()
             if e["entail"] >= MIN_CONFIDENCE),
            reverse=True,
        )
        contradicting = sorted(
            (e["contra"] for e in by_source.values()
             if e["contra"] >= MIN_CONFIDENCE),
            reverse=True,
        )

        if len(supporting) >= 2 and len(supporting) > len(contradicting):
            return {
                "verdict": VERDICT_TRUE,
                "confidence": supporting[1],
                "reason": (
                    "Минимум два независимых источника согласуются "
                    "с утверждением."
                ),
                "details": [],
                "raw_response": raw,
                "decision_source": "nli",
            }

        # Число, привязанное к тому же объекту, — сильный признак: ошибка
        # в цифре у именованного объекта не требует второго источника.
        strong_numeric = [
            e for e in by_source.values() if e.get("strong_numeric")
        ]

        if len(strong_numeric) >= 1 and not supporting:
            best = max(strong_numeric, key=lambda e: e["contra"])
            return {
                "verdict": VERDICT_FAKE,
                "confidence": max(float(best["contra"]), MIN_CONFIDENCE),
                "reason": (
                    "Число в утверждении противоречит источнику, где указана "
                    "та же величина того же объекта."
                ),
                "details": [],
                "raw_response": raw,
                "decision_source": "nli",
            }

        if len(contradicting) >= 2 and len(contradicting) > len(supporting):
            return {
                "verdict": VERDICT_FAKE,
                "confidence": contradicting[1],
                "reason": (
                    "Минимум два независимых источника противоречат "
                    "утверждению."
                ),
                "details": [],
                "raw_response": raw,
                "decision_source": "nli",
            }

        all_values = [
            value
            for e in by_source.values()
            for value in (e["contra"], e["entail"])
        ]
        return {
            "verdict": VERDICT_UNKNOWN,
            "confidence": max(all_values),
            "reason": (
                "FLUD не нашла подтверждения от двух независимых "
                "источников ни в одну сторону."
            ),
            "details": [],
            "raw_response": raw,
            "decision_source": "nli_uncertain",
        }

    def check_fact(
        self,
        claim: str,
        fact_result: Dict[str, Any],
    ):
        result = self.verify(
            claim,
            fact_result,
        )

        return {
            "label": result["verdict"],
            "score": result["confidence"],
            "details": result["details"],
            "explanation": result["reason"],
        }


def verify_claims(
    checker: "NLIChecker",
    text: str,
    fact_result: Dict[str, Any],
    max_claims: int = 4,
) -> Optional[Dict[str, Any]]:
    """Проверяет каждое утверждение текста отдельно и сводит результаты.

    Раньше FLUD получала весь текст: цитаты подбирались по терминам всего
    абзаца, и подтверждение одного предложения переносилось на другое.
    Теперь утверждения проверяются по отдельности, а вердикты сводятся:
    любой ФЕЙК -> ФЕЙК, иначе большинство ПРАВДА -> ПРАВДА.
    """
    from main_code import classify_claim_importance, split_into_claims

    claims = [c for c in split_into_claims(text) if c.strip()][:max_claims]
    if not claims:
        return None

    # Важные утверждения идут первыми, но и остальные проверяются:
    # иначе ложное утверждение без чисел ("Луна — спутник Марса")
    # вообще не попадало в проверку.
    important = [
        c for c in claims
        if classify_claim_importance(c) == "important"
    ]
    rest = [c for c in claims if c not in important]
    targets = (important + rest)[:max_claims]

    results = []
    for claim in targets:
        result = checker.verify(claim, fact_result)
        if result.get("verdict") in (VERDICT_TRUE, VERDICT_FAKE):
            results.append((claim, result))

    if not results:
        return checker.verify(targets[0], fact_result)

    # Если уверенный вердикт есть лишь по меньшинству утверждений,
    # говорить ПРАВДА или ФЕЙК нельзя: большая часть текста не проверена.
    if len(results) * 2 < len(targets):
        best = max(
            (r for _, r in results),
            key=lambda r: float(r.get("confidence", 0.0)),
        )
        return {
            **best,
            "verdict": VERDICT_UNKNOWN,
            "reason": (
                "Уверенный вердикт получен лишь по части утверждений, "
                "остальные источники не подтвердили."
            ),
            "claims_checked": len(results),
            "claims_total": len(claims),
        }

    fake = [r for _, r in results if r["verdict"] == VERDICT_FAKE]
    true = [r for _, r in results if r["verdict"] == VERDICT_TRUE]

    if fake:
        best = max(fake, key=lambda r: float(r.get("confidence", 0.0)))
        return {
            **best,
            "reason": "Хотя бы одно утверждение текста противоречит источникам.",
            "claims_checked": len(results),
            "claims_total": len(claims),
        }
    if len(true) > len(results) / 2:
        best = max(true, key=lambda r: float(r.get("confidence", 0.0)))
        return {
            **best,
            "reason": "Большинство проверенных утверждений подтверждено источниками.",
            "claims_checked": len(results),
            "claims_total": len(claims),
        }

    best = max(
        (r for _, r in results),
        key=lambda r: float(r.get("confidence", 0.0)),
    )
    return {
        **best,
        "verdict": VERDICT_UNKNOWN,
        "reason": "Утверждения текста подтверждаются источниками по-разному.",
        "claims_checked": len(results),
        "claims_total": len(claims),
    }


def is_confident(
    fact_result: Dict[str, Any],
) -> bool:
    verdict = fact_result.get("verdict")
    confidence = float(
        fact_result.get("confidence", 0.0)
    )

    return (
        verdict in (VERDICT_TRUE, VERDICT_FAKE)
        and confidence >= MIN_CONFIDENCE
    )


def combine(
    fact_result: Dict[str, Any],
    flud_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fact_verdict = fact_result.get(
        "verdict",
        VERDICT_UNKNOWN,
    )
    fact_confidence = float(
        fact_result.get("confidence", 0.0)
    )

    if flud_result is None:
        if not is_confident(fact_result):
            return {
                "verdict": VERDICT_BIG_DOUBTS,
                "confidence": 0.20,
                "reason": (
                    "FactChecker не дал уверенного вердикта, "
                    "а FLUD не была вызвана."
                ),
                "details": fact_result.get(
                    "details",
                    [],
                ),
                "decision_source": "uncertain",
            }

        return {
            "verdict": fact_verdict,
            "confidence": fact_confidence,
            "reason": fact_result.get(
                "reason",
                "Вердикт основан на проверке источников.",
            ),
            "details": fact_result.get(
                "details",
                [],
            ),
            "decision_source": "factchecker",
        }

    flud_verdict = flud_result.get(
        "verdict",
        VERDICT_UNKNOWN,
    )
    flud_confidence = float(
        flud_result.get("confidence", 0.0)
    )

    fact_is_confident = (
        fact_verdict == VERDICT_TRUE
        and fact_confidence >= FACT_MIN_CONFIDENCE
    ) or (
        fact_verdict == VERDICT_FAKE
        and fact_confidence >= FACT_FAKE_MIN_CONFIDENCE
    )
    flud_is_confident = (
        flud_verdict in (
            VERDICT_TRUE,
            VERDICT_FAKE,
        )
        and flud_confidence >= MIN_CONFIDENCE
    )

    if (
        fact_is_confident
        and flud_is_confident
        and fact_verdict == flud_verdict
    ):
        return {
            "verdict": fact_verdict,
            "confidence": max(
                fact_confidence,
                flud_confidence,
            ),
            "reason": (
                "FactChecker и FLUD согласны в вердикте."
            ),
            "details": fact_result.get(
                "details",
                [],
            ),
            "decision_source": "factchecker+flud",
        }

    if flud_is_confident:
        return {
            "verdict": flud_verdict,
            "confidence": flud_confidence,
            "reason": flud_result.get(
                "reason",
                "FLUD определила вердикт "
                "по найденным источникам.",
            ),
            "details": fact_result.get(
                "details",
                [],
            ),
            "decision_source": "flud",
        }

    if fact_is_confident:
        return {
            "verdict": fact_verdict,
            "confidence": fact_confidence,
            "reason": fact_result.get(
                "reason",
                "Вердикт основан на проверке источников.",
            ),
            "details": fact_result.get(
                "details",
                [],
            ),
            "decision_source": "factchecker",
        }

    return {
        "verdict": VERDICT_BIG_DOUBTS,
        "confidence": 0.20,
        "reason": (
            "FactChecker и FLUD не смогли дать "
            "уверенный вердикт по найденным источникам."
        ),
        "details": fact_result.get(
            "details",
            [],
        ),
        "decision_source": "uncertain",
    }
