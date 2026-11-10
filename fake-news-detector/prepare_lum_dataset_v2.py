import json
import os

OLD_JSONL = "factcheck_dataset.jsonl"
NEW_JSON = "dataset_binary_checkpoint_868.json"
OUTPUT_JSONL = "factcheck_dataset_v2.jsonl"

INSTRUCTION = (
    "Ты — фактчекер для русскоязычных новостей. "
    "Проанализируй утверждение и источники. "
    "Верни только валидный JSON без пояснений."
)

def normalize_label(label):
    label = str(label).strip().upper()

    if label in {"ПРАВДА", "TRUE", "REAL"}:
        return "ПРАВДА"

    if label in {"ФЕЙК", "FALSE", "FAKE"}:
        return "ФЕЙК"

    raise ValueError(f"Неизвестная метка: {label}")

def make_input(claim, source_text, source_url):
    parts = [f"Утверждение: {claim.strip()}"]

    if source_text:
        parts.append(f"\nИсточник 1: {source_text.strip()}")

    if source_url:
        parts.append(f"\nИсточник URL: {source_url.strip()}")

    return "\n".join(parts)

def make_output(label, source_url):
    verdict = normalize_label(label)

    return json.dumps(
        {
            "verdict": verdict,
            "confidence": 0.90,
            "reason": (
                "Утверждение подтверждается предоставленным источником."
                if verdict == "ПРАВДА"
                else "Утверждение противоречит предоставленному источнику."
            ),
            "sources_used": [1] if source_url else [],
        },
        ensure_ascii=False,
    )

def load_old_jsonl(path):
    rows = []

    if not os.path.exists(path):
        return rows

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows

def load_new_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    old_rows = load_old_jsonl(OLD_JSONL)
    new_rows = load_new_json(NEW_JSON)

    all_rows = []
    seen_claims = set()

    for row in old_rows:
        claim = str(row.get("input", "")).strip()
        key = claim.lower()

        if key not in seen_claims:
            seen_claims.add(key)
            all_rows.append(row)

    added = 0

    for row in new_rows:
        claim = str(row.get("claim", "")).strip()
        key = claim.lower()

        if not claim or key in seen_claims:
            continue

        label = row.get("label", "")
        source_text = row.get("source_text", "")
        source_url = row.get("source_url", "")

        if not source_text:
            continue

        example = {
            "instruction": INSTRUCTION,
            "input": make_input(claim, source_text, source_url),
            "output": make_output(label, source_url),
        }

        all_rows.append(example)
        seen_claims.add(key)
        added += 1

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Старых примеров: {len(old_rows)}")
    print(f"Новых добавлено: {added}")
    print(f"Всего в v2: {len(all_rows)}")
    print(f"Сохранено: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()