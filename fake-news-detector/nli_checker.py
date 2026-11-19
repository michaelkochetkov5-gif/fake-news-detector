"""FLUD: NLI zero-shot + numeric checker — CPU fallback для сайта."""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from main_code import claim_key_terms, lemma_tokens, select_terms

VERDICT_TRUE = "ПРАВДА"
VERDICT_FAKE = "ФЕЙК"
VERDICT_UNKNOWN = "НЕИЗВЕСТНО"
VERDICT_BIG_DOUBTS = "БОЛЬШИЕ СОМНЕНИЯ"

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MIN_CONFIDENCE = 0.50
FACT_MIN_CONFIDENCE = 0.60
FACT_FAKE_MIN_CONFIDENCE = 0.80
FACT_SKIP_CONFIDENCE = 0.90
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


NUMBER_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(\u00b0\s*[a-z\u0430-\u044f\u0451]|[a-z\u0430-\u044f\u0451%]+)")


def _number_units(text: str) -> List[tuple]:
    pairs = []
    cleaned = text.lower().replace("\u00a0", " ").replace("*", "")
    for match in NUMBER_UNIT_RE.finditer(cleaned):
        value = match.group(1).replace(",", ".")
        unit = re.sub(r"\s+", "", match.group(2)).replace("\u0441", "c")[:3]
        if unit in ("\u0433", "\u0433\u0433"):
            unit = "\u0433\u043e\u0434"
        pairs.append((value, unit))
    return pairs


def _numbers_conflict(claim: str, text: str) -> bool:
    text_pairs = _number_units(text)
    for value, unit in _number_units(claim):
        same_unit = [v for v, u in text_pairs if u == unit]
        if same_unit and value not in same_unit:
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
        lemmas = set(lemma_tokens(text))
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
                premises.append({"text": text, "source": source_key})

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
                premise["source"], {"contra": 0.0, "entail": 0.0}
            )
            coverage = self._shared_terms(claim, premise["text"]) / n_terms
            if _numbers_conflict(claim, premise["text"]) or (
                NONNUM_CONTRA_COVERAGE_MIN
                and coverage >= NONNUM_CONTRA_COVERAGE_MIN
            ):
                entry["contra"] = max(entry["contra"], float(row[0]))
            if coverage >= ENTAIL_COVERAGE_MIN:
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
