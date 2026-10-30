# Foundation verification record

30.10.2026, Asia/Qyzylorda. Исполнитель — Codex по запросу пользователя;
независимый human reviewer не участвовал. Это local setup evidence, не REPRO-002,
не воспроизведение checkpoint и не measured experiment results.

## Installation evidence

Создана отдельная `.venv` на Python3.12.10, system-site-packages отключены.
Первый ensurepip с default sandbox temp не прошёл из-за permissions; повтор с
TEMP/TMP внутри ignored workspace прошёл. Restricted-network pip не разрешил DNS;
установка с разрешённым network access прошла. Глобальные packages не менялись.

Пара torch2.9.0/vision0.24.0 выбрана по official Python compatibility и install
tables, а не только по planning candidate. Получены cu126 wheels. Прямые versions,
38 installed packages и sources/recipe сохранены в requirements*.txt и
[REPRODUCIBILITY](../docs/REPRODUCIBILITY.md). Повторный `pip install -r
requirements-lock.txt --disable-pip-version-check` не изменил packages.
Fresh-machine install из lock и независимое review pending.

## Выполненные проверки

Все команды через `.venv/Scripts/python.exe` из repository root.

| Команда / проверка | Результат | Предел |
|---|---|---|
| `-m pip check` | PASS: No broken requirements found | Metadata constraints, дополнены real imports/ops |
| `-m scripts.check_environment` | PASS: imports всех direct dependencies/pytest/Pillow, PNG encode/decode round-trip | Synthetic image, не CIFAR integrity/loader |
| CPU synthetic conv forward/backward и finite gradients | PASS | Нет project CNN или optimizer/training loop |
| CUDA synthetic conv forward/backward и finite gradients | PASS | RTX3060 Laptop, driver555.97, runtime12.6, cuDNN91002; не все CUDA operations |
| torchvision compiled NMS CPU/CUDA | PASS | Проверка binary extension compatibility |
| deterministic flags / IEEE precision | Включены, smoke PASS | Не доказательство exact resume/cross-device equality |
| `-m pytest tests -q --basetemp=.venv/pytest-tmp` | **116 passed**, 0 failed, 0 skipped | Сейчас collected только config tests; data/corruption placeholders пустые |
| `-m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair` | PASS: 2configs, controls equal | Preparation schema; data artifacts и training readiness не verified |
| Invalid config CLI / controlled mismatches / no-artifact loading | PASS в tests/test_utils.py | exit2 и понятные ошибки; чтение не создаёт данных |
| `git diff --check`, staged file review перед каждым commit | PASS | Только конкретные task files, без .venv/credentials/temp outputs |
| Git ignore checks | PASS для .venv, tmp, pytest cache, run/prediction outputs, .env.local | Никакие secrets не создавались |
| Documentation consistency audit | PASS:21 Markdown files, links/fences;49cards×20fields, acyclic dependencies;96planned matrix rows | Не подтверждает team approval или результаты experiments |
| Scope audit against0633f67 | PASS: excluded pipeline modules/scripts/tests byte-identical и пустые; data/output directories только .gitkeep | Dataset/test set не читались |

[setup_environment.json](setup_environment.json) — curated runtime evidence;
содержит actual check timestamp, exact direct versions, hardware/backend и SHA-256
lock после нормализации newlines к LF. Absolute local paths и raw temp logs не
сохраняются. Environment checker — commit b3c5c81; configs/tests — cb06ee3.

## Оставшиеся проверки и ограничения

- SETUP-001: latest LMS/rubric, receipt/submitted proposal version, policy conflict,
  cutoff/timezone и presentation slot не подтверждены.
- PROTO-001/TEAM-001: team review/freeze, role mapping/availability/backups pending.
- SETUP-002/003: local technical acceptance закрыт; формальный human reviewer
  approval из roadmap DoD не получен. Поэтому полный task status PARTIAL.
- REPRO-001/002: stable identity/RNG/checkpoint/backup infrastructure и независимое
  fresh-machine воспроизведение ещё не реализованы/не выполнены.
- CPU-only wheel, другие OS/GPU/drivers не проверены; нет wheel hash lock/offline
  package archive. Численная и training reproducibility не обещается.
- CIFAR-10 не скачан; split/stats hashes null. Test set не использовался.
  Data/corruptions/model/train/eval/plots остаются вне выполненного этапа.
