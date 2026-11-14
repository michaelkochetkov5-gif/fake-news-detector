"""LUM -> FLUD: безопасное переименование. Запуск: python rename_lum_to_flud.py"""

import re
import shutil
import py_compile
from pathlib import Path

FILES = ["site.py", "nli_checker.py", "lum_checker.py", "pages/test_factchecker.py"]

# Путь к весам на диске остаётся прежним. Защищаем его до всех замен.
PROTECT = "lum-factchecker-lora-v2"
SENTINEL = "\x00WEIGHTS_PATH\x00"

# Явные идентификаторы: заменяются целиком, до regex-прохода.
EXPLICIT = [
    ("LUM_MODEL_PATH", "FLUD_MODEL_PATH"),
    ("parse_lum_response", "parse_flud_response"),
    ("load_lum_checker", "load_flud_checker"),
    ("lum_is_confident", "flud_is_confident"),
    ("lum_confidence", "flud_confidence"),
    ("lum_verdict", "flud_verdict"),
    ("lum_checker", "flud_checker"),
    ("lum_result", "flud_result"),
    ("lum_reason", "flud_reason"),
    ("LUMChecker", "FLUDChecker"),
    ("factchecker+lum", "factchecker+flud"),
    ("lum-nli-numeric-v2", "flud-nli-numeric-v2"),
]

report = []

for name in FILES:
    path = Path(name)
    if not path.exists():
        report.append(f"ПРОПУСК (нет файла): {name}")
        continue

    original = path.read_bytes().decode("utf-8")
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    text = original.replace(PROTECT, SENTINEL)

    for old, new in EXPLICIT:
        text = text.replace(old, new)

    # По границам слов: не задевает columns, formula, volume, album.
    text = re.sub(r"\bLUM\b", "FLUD", text)
    text = re.sub(r"\blum\b", "flud", text)

    text = text.replace(SENTINEL, PROTECT)

    if text != original:
        path.write_bytes(text.encode("utf-8"))

    changed = sum(1 for a, b in zip(original.split("\n"), text.split("\n")) if a != b)
    report.append(f"{name}: изменено строк {changed}")

    # Что осталось
    leftovers = [
        (i, line.strip()[:90])
        for i, line in enumerate(text.split("\n"), 1)
        if "lum" in line.lower() and PROTECT not in line
    ]
    for i, line in leftovers:
        report.append(f"    остаток {name}:{i}: {line}")

# Переименование файла модуля
old_file = Path("lum_checker.py")
if old_file.exists():
    old_file.with_suffix(".py.bak").unlink(missing_ok=True)
    shutil.move("lum_checker.py", "flud_checker.py")
    report.append("файл lum_checker.py -> flud_checker.py")

# Проверка компиляции
report.append("\n=== py_compile ===")
for name in ["site.py", "nli_checker.py", "flud_checker.py",
             "pages/test_factchecker.py", "main_code.py", "numeric_checker.py",
             "query_extractor.py"]:
    p = Path(name)
    if not p.exists():
        continue
    try:
        py_compile.compile(str(p), doraise=True)
        report.append(f"  OK    {name}")
    except py_compile.PyCompileError as exc:
        report.append(f"  ОШИБКА {name}: {exc.msg}")

print("\n".join(report))
print("\nБэкапы: *.bak рядом с исходными файлами.")
