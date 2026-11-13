"""Извлечение осмысленных поисковых фраз из текста."""

import re
from typing import List

try:
    from pymorphy3 import MorphAnalyzer
except Exception:
    MorphAnalyzer = None

STOP_WORDS = {
    "и",
    "а",
    "но",
    "что",
    "как",
    "это",
    "этот",
    "эта",
    "эти",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "для",
    "от",
    "до",
    "из",
    "у",
    "же",
    "бы",
    "ли",
    "не",
    "ни",
    "о",
    "об",
    "то",
    "так",
    "вот",
    "всё",
    "все",
    "который",
    "которая",
    "которое",
    "которые",
    "чтобы",
    "если",
    "когда",
    "где",
    "куда",
    "зачем",
    "почему",
    "потому",
    "поэтому",
    "также",
    "тоже",
    "ещё",
    "уже",
    "очень",
    "более",
    "менее",
    "самый",
    "самая",
    "самое",
    "самые",
}

morph = None

if MorphAnalyzer is not None:
    try:
        morph = MorphAnalyzer()
    except Exception:
        morph = None


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(
        r"[^\w\sа-яёА-ЯЁ.,!?;:()\-\"]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    parts = re.split(
        r"[.!?;]+",
        text,
    )

    sentences = []

    for part in parts:
        part = part.strip()

        if part:
            sentences.append(part)

    return sentences


def _split_clauses(sentence: str) -> List[str]:
    parts = re.split(
        r"[,]+",
        sentence,
    )

    clauses = []

    for part in parts:
        part = part.strip()

        if part:
            clauses.append(part)

    return clauses


def _get_pos(word: str) -> str:
    if morph is None:
        return ""

    parsed = morph.parse(word)

    if not parsed:
        return ""

    return parsed[0].tag.POS or ""


def _is_important_word(word: str) -> bool:
    if len(word) < 3:
        return False

    if word in STOP_WORDS:
        return False

    pos = _get_pos(word)

    important_pos = {
        "NOUN",
        "ADJF",
        "ADJS",
        "VERB",
        "INFN",
        "PRTF",
        "PRTS",
        "GRND",
        "NUMR",
    }

    if pos and pos not in important_pos:
        return False

    return True


def _extract_candidates(
    text: str,
) -> List[List[str]]:
    cleaned = _clean_text(text.lower())

    candidates = []

    for sentence in _split_sentences(cleaned):
        for clause in _split_clauses(sentence):
            words = re.findall(
                r"[а-яёa-z0-9]+",
                clause,
            )

            important_words = [
                word
                for word in words
                if _is_important_word(word)
            ]

            for i in range(
                len(important_words) - 1
            ):
                bigram = important_words[
                    i:i + 2
                ]

                candidates.append(bigram)

            for i in range(
                len(important_words) - 2
            ):
                trigram = important_words[
                    i:i + 3
                ]

                candidates.append(trigram)

    return candidates


def _phrase_key(words: List[str]) -> str:
    return " ".join(words)


def extract_key_phrases(
    text: str,
) -> List[str]:
    if not text or not text.strip():
        return []

    if morph is None:
        return []

    candidates = _extract_candidates(text)

    if not candidates:
        return []

    scored = []

    for words in candidates:
        phrase = _phrase_key(words)

        score = 0.0

        if len(words) == 3:
            score += 2.0
        else:
            score += 1.0

        if any(
            _get_pos(word) == "NOUN"
            for word in words
        ):
            score += 1.0

        if any(
            _get_pos(word) in (
                "VERB",
                "INFN",
            )
            for word in words
        ):
            score += 0.5

        scored.append(
            (score, len(words), phrase)
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
        )
    )

    selected = []
    selected_words = []

    for score, length, phrase in scored:
        words = phrase.split()

        overlap = set(words) & set(
            selected_words
        )

        if len(overlap) >= 2:
            continue

        selected.append(phrase)
        selected_words.extend(words)

        if len(selected) >= 3:
            break

    return selected