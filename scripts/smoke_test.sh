#!/usr/bin/env bash
# Preview routing only (no backends required).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from router.intent_rules import route_intent

cases = [
    "Переклади цей абзац англійською.",
    "Напиши Python функцію для парсингу JSON.",
    "Хто такий Іван Франко? Що нового у 2025?",
    "Виконай інструкцію покроково і верни JSON з полями a,b.",
    "Привіт! Як справи? Розкажіть анекдот українською.",
]
for t in cases:
    d = route_intent(t)
    print(f"{d.model:6} | {d.reason:40} | {t[:60]}")
PY

echo
echo "Optional live preview (router up on :4010):"
echo "  curl -s localhost:4010/v1/route/preview -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\":\"Напиши код на Python\"}'"
