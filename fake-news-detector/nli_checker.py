"""NLI zero-shot + numeric checker — CPU fallback для сайта."""

import re
from typing import Any, Dict, List

import numpy as np

import numeric_checker

VERDICT_TRUE = "ПРАВДА"
VERDICT_FAKE = "ФЕЙК"
VERDICT_UNKNOWN = "НЕИЗВЕСТНО"

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MIN_CONFIDENCE = 0.50


class NLIChecker:
    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.load_error = None

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as exc:
            self.load_error = (
                f"NLI недоступна: {type(exc).__name__}: {exc}"
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        return self.available

    def _extract_premises(
        self,
        fact_result: Dict[str, Any],
    ) -> List[str]:
        premises = []

        for detail in fact_result.get("details", []):
            reason = str(detail.get("reason", "")).strip()
            source = str(detail.get("source", "")).strip()

            if not reason and not source:
                continue

            quotes = re.findall(
                r'["«]([^"»]{10,1000})["»]',
                reason,
            )

            if quotes:
                premises.extend(quotes[:3])
            else:
                if reason:
                    premises.append(reason)

                if source and source not in premises:
                    premises.append(source)

        cleaned_premises = []

        for premise in premises:
            premise = premise.strip()

            if len(premise) >= 10:
                cleaned_premises.append(premise)

        return cleaned_premises[:3]

    def verify(
        self,
        claim: str,
        fact_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.available:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": f"NLI недоступна: {self.load_error}",
                "details": [],
                "raw_response": "",
                "decision_source": "nli_unavailable",
            }

        premises = self._extract_premises(fact_result)

        if not premises:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": "NLI не нашла подходящих цитат из источников.",
                "details": [],
                "raw_response": "",
                "decision_source": "nli_no_premise",
            }

        inputs = [
            (premise, claim)
            for premise in premises
        ]

        try:
            scores = self.model.predict(inputs)
        except Exception as exc:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": (
                    f"Ошибка NLI-инференса: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "details": [],
                "raw_response": "",
                "decision_source": "nli_error",
            }

        scores = np.asarray(scores)

        # CrossEncoder возвращает logits.
        # Преобразуем их в вероятности через softmax.
        exp_scores = np.exp(
            scores - scores.max(axis=-1, keepdims=True)
        )

        probabilities = exp_scores / exp_scores.sum(
            axis=-1,
            keepdims=True,
        )

        # Порядок меток:
        # 0 = contradiction
        # 1 = entailment
        # 2 = neutral
        contradiction_score = 0.0
        entailment_score = 0.0
        best_premise = ""

        for premise, probability in zip(premises, probabilities):
            contradiction = float(probability[0])
            entailment = float(probability[1])

            if contradiction > contradiction_score:
                contradiction_score = contradiction
                best_premise = premise

            if entailment > entailment_score:
                entailment_score = entailment

        if contradiction_score >= entailment_score:
            verdict = VERDICT_FAKE
            confidence = contradiction_score
        else:
            verdict = VERDICT_TRUE
            confidence = entailment_score

        numeric_result = numeric_checker.check_numbers(
            claim,
            best_premise,
        )

        adjusted = numeric_checker.adjust_nli_confidence(
            contradiction_score,
            numeric_result["features"],
        )

        adjusted_contradiction = adjusted["p_contradiction"]

        if (
            adjusted["changed"]
            and adjusted_contradiction > entailment_score
        ):
            verdict = VERDICT_FAKE
            confidence = adjusted_contradiction

        if confidence < MIN_CONFIDENCE:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": confidence,
                "reason": "NLI недостаточно уверена в вердикте.",
                "details": [],
                "raw_response": str(probabilities.tolist()),
                "decision_source": "nli_uncertain",
            }

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": (
                "NLI сравнила утверждение с найденным источником; "
                "numeric checker проверил числовые совпадения."
            ),
            "details": [],
            "raw_response": str(probabilities.tolist()),
            "decision_source": "nli",
        }

    def check_fact(
        self,
        claim: str,
        fact_result: Dict[str, Any],
    ):
        result = self.verify(claim, fact_result)

        return {
            "label": result["verdict"],
            "score": result["confidence"],
            "details": result["details"],
            "explanation": result["reason"],
        }
