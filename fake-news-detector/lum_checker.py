"""LUM v2 — семантический fallback-ассистент фактчекера.

Модель обучена через Unsloth / PEFT LoRA и лежит в локальной папке
``lum-factchecker-lora-v2``. Модуль импортируется без torch/unsloth:
тяжёлые зависимости подтягиваются лениво, в момент первого вызова.
"""

import json
import re

LUM_MODEL_PATH = "lum-factchecker-lora-v2"

VERDICT_TRUE = "ПРАВДА"
VERDICT_FAKE = "ФЕЙК"
VERDICT_UNKNOWN = "НЕИЗВЕСТНО"

DECISION_THRESHOLD = 0.70

SOURCE_MESSAGES = {
    "factchecker": "Вердикт основан на проверке источников.",
    "lum": "Вердикт основан на семантической модели LUM v2.",
    "factchecker+lum": "Фактчекер и LUM v2 согласны.",
    "uncertain": "Недостаточно данных для надёжного вердикта.",
}

_SYSTEM_PROMPT = (
    "Ты — фактчекер. Оцени утверждение пользователя, опираясь на проверяемые факты.\n"
    "Ответь строго одним JSON-объектом без пояснений и без markdown:\n"
    '{"verdict": "ПРАВДА" | "ФЕЙК" | "НЕИЗВЕСТНО", '
    '"confidence": число от 0 до 1, "reason": "...", "sources_used": []}\n'
    'Если данных недостаточно — verdict "НЕИЗВЕСТНО".'
)

_VERDICT_RE = re.compile(
    r'(?:verdict|вердикт)["\']?\s*[:=]\s*["\']?\s*(ПРАВДА|ФЕЙК|НЕИЗВЕСТНО)',
    re.IGNORECASE,
)
_CONF_RE = re.compile(
    r'(?:confidence|уверенность)["\']?\s*[:=]\s*["\']?\s*'
    r'(\d+(?:\s*[.,]\s*\d+)?)\s*%?',
    re.IGNORECASE,
)
)
_REASON_RE = re.compile(
    r'(?:reason|причина)["\']?\s*[:=]\s*["\']((?:[^"\'\\]|\\.)*)',
    re.IGNORECASE,
)
_SOURCES_RE = re.compile(
    r'(?:sources_used|источники)["\']?\s*[:=]\s*\[(.*?)\]',
    re.DOTALL | re.IGNORECASE,
)
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _normalise_confidence(raw):
    """Приводит '0.8', '0,8', '80%', 0.8 к float в диапазоне [0, 1]."""
    if raw is None:
        return 0.0
    try:
        value = float(
    re.sub(r"\s+", "", str(raw).strip()).replace(",", ".")
)
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < value <= 100.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _strip_code_fences(text):
    """Убирает markdown-обёртку ```json ... ```."""
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_lum_response(raw):
    """Извлекает verdict/confidence/reason/sources_used из ответа LUM v2.

    Сначала пробует разобрать JSON целиком, затем падает на регулярные
    выражения. Оба пути принимают confidence с точкой, запятой и знаком %.
    """
    text = _strip_code_fences((raw or "").strip())

    parsed = {
        "verdict": VERDICT_UNKNOWN,
        "confidence": 0.0,
        "reason": "",
        "sources_used": [],
        "verdict_found": False,
    }

    if not text:
        return parsed

    match = _JSON_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(0))
        except (ValueError, TypeError):
            payload = None

        if isinstance(payload, dict):
            verdict = str(payload.get("verdict", "")).upper().replace("Ё", "Е").strip()
            if verdict in (VERDICT_TRUE, VERDICT_FAKE, VERDICT_UNKNOWN):
                parsed["verdict"] = verdict
                parsed["verdict_found"] = True

            if "confidence" in payload:
                parsed["confidence"] = _normalise_confidence(payload["confidence"])

            reason = payload.get("reason")
            if isinstance(reason, str):
                parsed["reason"] = reason.strip()

            sources = payload.get("sources_used")
            if isinstance(sources, list):
                parsed["sources_used"] = [str(item) for item in sources]

            if parsed["verdict_found"]:
                return parsed

    match = _VERDICT_RE.search(text)
    if match:
        verdict = match.group(1).upper().replace("Ё", "Е").strip()
        if verdict in (VERDICT_TRUE, VERDICT_FAKE, VERDICT_UNKNOWN):
            parsed["verdict"] = verdict
            parsed["verdict_found"] = True

    match = _CONF_RE.search(text)
    if match:
        parsed["confidence"] = _normalise_confidence(match.group(1))

    match = _REASON_RE.search(text)
    if match:
        parsed["reason"] = match.group(1).replace('\\"', '"').strip()

    match = _SOURCES_RE.search(text)
    if match:
        try:
            items = json.loads("[" + match.group(1) + "]")
        except (ValueError, TypeError):
            items = None
        if isinstance(items, list):
            parsed["sources_used"] = [str(item) for item in items]

    return parsed


class LUMChecker:
    """Обёртка над LUM v2. Модель грузится лениво и только один раз."""

    def __init__(
        self,
        model_path=LUM_MODEL_PATH,
        max_seq_length=4096,
        load_in_4bit=True,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    ):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = None
        self.tokenizer = None
        self.load_error = None

        self._torch = None
        self._load_attempted = False

    @property
    def available(self):
        return self.model is not None and self.tokenizer is not None

    def load(self):
        """Загружает модель. Возвращает True/False, причину кладёт в load_error."""
        if self._load_attempted:
            return self.available
        self._load_attempted = True

        try:
            import torch
            from unsloth import FastLanguageModel
        except Exception as exc:
            self.load_error = (
                f"unsloth/torch недоступны: {type(exc).__name__}: {exc}"
            )
            return False

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
        except Exception as exc:
            self.load_error = (
                f"не удалось загрузить '{self.model_path}': "
                f"{type(exc).__name__}: {exc}"
            )
            return False

        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch
        self.load_error = None
        return True

    def _build_prompt(self, text):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        return f"{_SYSTEM_PROMPT}\n\nУтверждение: {text}\nОтвет:"

    def verify(self, text):
        """Возвращает результат, совместимый по ключевым полям с FactChecker.verify()."""
        if not self.load():
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": f"LUM v2 недоступна: {self.load_error}",
                "details": [],
                "raw_response": "",
            }

        raw = ""

        try:
            prompt = self._build_prompt(text)
            encoded = self.tokenizer([prompt], return_tensors="pt")
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}

            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "use_cache": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            # temperature имеет смысл только при сэмплировании;
            # при do_sample=False transformers игнорирует её и ругается в лог.
            if self.do_sample:
                generation_kwargs["temperature"] = self.temperature

            with self._torch.no_grad():
                output = self.model.generate(**encoded, **generation_kwargs)

            generated = output[0][encoded["input_ids"].shape[-1]:]
            raw = self.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
        except Exception as exc:
            return {
                "verdict": VERDICT_UNKNOWN,
                "confidence": 0.0,
                "reason": f"Ошибка инференса LUM v2: {type(exc).__name__}: {exc}",
                "details": [],
                "raw_response": raw,
            }

        parsed = parse_lum_response(raw)

        if not parsed["verdict_found"]:
            reason = "Не удалось разобрать ответ LUM v2."
        else:
            reason = parsed["reason"] or "LUM v2 не вернула обоснование."

        return {
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "reason": reason,
            "details": [],
            "raw_response": raw,
        }

    def check_fact(self, text):
        result = self.verify(text)

        return {
            "label": result["verdict"],
            "score": result["confidence"],
            "details": result["details"],
            "explanation": result["reason"],
        }


def is_confident(result, threshold=DECISION_THRESHOLD):
    """True, если фактчекер дал определённый вердикт с confidence >= threshold."""
    if not isinstance(result, dict):
        return False

    if result.get("verdict") not in (VERDICT_TRUE, VERDICT_FAKE):
        return False

    try:
        return float(result.get("confidence") or 0.0) >= threshold
    except (TypeError, ValueError):
        return False


def _as_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def combine(fact_result, lum_result):
    """Сводит вердикты FactChecker и LUM v2 по подтверждённой таблице решений.

    Все существующие поля fact_result (verdict, confidence, reason, details)
    сохраняются; переопределяются только те, что требует итоговая логика.
    """
    fact_result = fact_result or {}
    lum_result = lum_result or {}

    fact_verdict = fact_result.get("verdict", VERDICT_UNKNOWN)
    lum_verdict = lum_result.get("verdict", VERDICT_UNKNOWN)

    fact_confidence = _as_float(fact_result.get("confidence"))
    lum_confidence = _as_float(lum_result.get("confidence"))

    final = dict(fact_result)
    final["details"] = fact_result.get("details", [])

    # 1. FactChecker уверен — LUM не участвует.
    if is_confident(fact_result):
        final["decision_source"] = "factchecker"
        return final

    # 2-4. FactChecker дал вердикт, но с низкой уверенностью.
    if fact_verdict in (VERDICT_TRUE, VERDICT_FAKE):
        if lum_verdict == fact_verdict:
            final["confidence"] = max(fact_confidence, lum_confidence)
            final["decision_source"] = "factchecker+lum"

        elif lum_verdict in (VERDICT_TRUE, VERDICT_FAKE):
            final["verdict"] = VERDICT_UNKNOWN
            final["confidence"] = min(fact_confidence, lum_confidence)
            final["reason"] = (
                "Фактчекер и LUM v2 дали противоположные вердикты; "
                "требуется дополнительная проверка."
            )
            final["decision_source"] = "uncertain"

        else:
            final["decision_source"] = "factchecker"

        return final

    # 5-6. FactChecker не определился — решает LUM.
    if lum_verdict in (VERDICT_TRUE, VERDICT_FAKE):
        final["verdict"] = lum_verdict
        final["confidence"] = lum_confidence
        final["reason"] = lum_result.get("reason") or fact_result.get("reason", "")
        final["decision_source"] = "lum"
    else:
        final["verdict"] = VERDICT_UNKNOWN
        final["confidence"] = fact_confidence
        final["reason"] = (
            fact_result.get("reason") or lum_result.get("reason") or ""
        )
        final["decision_source"] = "uncertain"

    return final
