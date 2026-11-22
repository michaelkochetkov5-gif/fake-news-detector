"""Проверка сведения вердиктов в verify_claims без сети и без модели."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nli_checker as nli  # noqa: E402


class StubChecker:
    """Отдаёт заранее заданный вердикт для каждого утверждения."""

    def __init__(self, answers):
        self.answers = answers
        self.seen = []

    def verify(self, claim, fact_result=None):
        self.seen.append(claim)
        verdict, confidence = self.answers[claim]
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": "stub",
            "details": [],
        }


TEXT = (
    "Земля вращается вокруг Солнца. "
    "Вода кипит при 100 градусах Цельсия. "
    "Луна является спутником Марса."
)


def claims():
    from main_code import split_into_claims

    return split_into_claims(TEXT)


def test_any_fake_wins():
    parts = claims()
    answers = {c: (nli.VERDICT_TRUE, 0.9) for c in parts}
    answers[parts[-1]] = (nli.VERDICT_FAKE, 0.95)
    checker = StubChecker(answers)
    result = nli.verify_claims(checker, TEXT, {}, max_claims=4)
    assert result["verdict"] == nli.VERDICT_FAKE, result
    assert result["claims_checked"] == len(parts)


def test_majority_true():
    parts = claims()
    answers = {c: (nli.VERDICT_TRUE, 0.9) for c in parts}
    result = nli.verify_claims(StubChecker(answers), TEXT, {}, max_claims=4)
    assert result["verdict"] == nli.VERDICT_TRUE, result


def test_split_verdict_is_unknown():
    parts = claims()
    answers = {c: (nli.VERDICT_UNKNOWN, 0.4) for c in parts}
    answers[parts[0]] = (nli.VERDICT_TRUE, 0.9)
    result = nli.verify_claims(StubChecker(answers), TEXT, {}, max_claims=4)
    assert result["verdict"] == nli.VERDICT_UNKNOWN, result


def test_claims_are_checked_separately():
    parts = claims()
    answers = {c: (nli.VERDICT_TRUE, 0.9) for c in parts}
    checker = StubChecker(answers)
    nli.verify_claims(checker, TEXT, {}, max_claims=4)
    assert len(checker.seen) <= len(parts)
    assert len(set(checker.seen)) == len(checker.seen), checker.seen


def test_empty_text_returns_none():
    assert nli.verify_claims(StubChecker({}), "   ", {}, max_claims=4) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
