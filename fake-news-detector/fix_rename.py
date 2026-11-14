"""LUM -> FLUD: полное переименование. Запуск: python fix_rename.py"""

import re
import shutil
import py_compile
from pathlib import Path

FILES = ["site.py", "nli_checker.py", "lum_checker.py", "pages/test_factchecker.py"]

PROTECT = "lum-factchecker-lora-v2"
SENTINEL = "\x00WEIGHTS\x00"

# Всё, что начинается с lum_ и является идентификатором
EXPLICIT = [
    ("LUM_MODEL_PATH", "FLUD_MODEL_PATH"),
    ("LUMChecker", "FLUDChecker"),
    ("parse_lum_response", "parse_flud_response"),
    ("load_lum_checker", "load_flud_checker"),
    ("lum_is_confident", "flud_is_confident"),
    ("lum_confidence", "flud_confidence"),
    ("lum_verdict", "flud_verdict"),
    ("lum_checker", "flud_checker"),
    ("lum_result", "flud_result"),
    ("lum_reason", "flud_reason"),
    ("lum_correct", "flud_correct"),
    ("lum_checked", "flud_checked"),
    ("lum_accuracy", "flud_accuracy"),
    ("lum_raw", "flud_raw"),
    ("factchecker+lum", "factchecker+flud"),
    ("lum-nli-numeric-v2", "flud-nli-numeric-v2"),
]

report = []

for name in FILES:
    path = Path(name)
    if not path.exists():
        report.append(f"ПРОПУСК: {name}")
        continue

    original = path.read_bytes().decode("utf-8")
    # бэкап только если его ещё нет — не затираем исходник
    backup = Path(str(path) + ".bak")
    if not backup.exists():
        shutil.copy2(path, str(backup))

    text = original.replace(PROTECT, SENTINEL)
    for old, new in EXPLICIT:
        text = text.replace(old, new)

    text = re.sub(r"\bLUM\b", "FLUD", text)
    # ВАЖНО: (?<![A-Za-z0-9]) вместо \b — подчёркивание не блокирует
    text = re.sub(r"(?<![A-Za-z0-9])lum_", "flud_", text)
    text = re.sub(r"(?<![A-Za-z0-9])lum(?![A-Za-z0-9_])", "flud", text)

    text = text.replace(SENTINEL, PROTECT)

    if text != original:
        path.write_bytes(text.encode("utf-8"))

    left = [
        f"{name}:{i}: {l.strip()[:80]}"
        for i, l in enumerate(text.split("\n"), 1)
        if re.search(r"(?<![A-Za-z0-9])lum", l) and PROTECT not in l
    ]
    report.append(f"{name}: изменено {sum(1 for a,b in zip(original.split(chr(10)), text.split(chr(10))) if a != b)} строк, остатков {len(left)}")
    report.extend("    " + x for x in left)

if Path("lum_checker.py").exists():
    shutil.move("lum_checker.py", "flud_checker.py")
    report.append("lum_checker.py -> flud_checker.py")

report.append("\n=== py_compile ===")
for name in ["site.py", "nli_checker.py", "flud_checker.py",
             "pages/test_factchecker.py", "main_code.py"]:
    p = Path(name)
    if not p.exists():
        continue
    try:
        py_compile.compile(str(p), doraise=True)
        report.append(f"  OK     {name}")
    except py_compile.PyCompileError as exc:
        report.append(f"  ОШИБКА {name}: {exc.msg}")

print("\n".join(report))
