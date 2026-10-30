# Verification record — foundation and DATA-001

## DATA-001 technical acceptance — 30–31.10.2026

Исполнитель — Codex по запросу пользователя, на прежнем Windows AMD64 /
Python3.12.10 environment. Завершение записано31.10.2026 по дате текущего чата;
acquisition manifest хранит фактический UTC timestamp
`2026-10-30T07:59:58.062311+00:00` (30.10,12:59:58 Asia/Qyzylorda).
Human reviewer не участвовал. **Local technical acceptance PASS; full DATA-001
DoD PARTIAL**, independent human artifact/code/test review pending. Это не
REPRO-002, не course compliance и не measured model results.

Initial state: clean main7658894, fetch подтвердил0ahead/0behind origin/main.
AGENTS.md в repository/родительской цепочке отсутствовал. Existing `.venv`, exact
lock, configs и backend сохранены. Исходный pip check PASS; baseline suite116PASS
с документированным `--basetemp`. Первый запуск без него дал104PASS/12ERROR из-за
sandbox default-temp permissions; это не успешный baseline run. Повтор с
workspace basetemp прошёл. Dependencies/global env не менялись; повтор CUDA smoke
не требовался и не выполнялся, прежнее CUDA evidence не выдаётся за новый run.

### Actual CIFAR-10 evidence

Before acquisition data/raw содержал только .gitkeep. Источники сверены по
[официальной странице](https://cave.cs.toronto.edu/kriz/cifar.html) и
[torchvision v0.24.0 CIFAR source](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/datasets/cifar.py),
затем с установленной0.24.0+cu126. URL torchvision
`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` перенаправил на official
`https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`; mirrors/HTTP fallback
не применялись. Code проверяет source restrictions и reference compatibility.

| Фактически проверено | Результат |
|---|---|
| Python archive |170498071 bytes; upstream MD5 `c58f30108f718f92721af3b95e74349a` PASS |
| Archive SHA-256 (observed identity) |`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| Extracted files |5train batches, test_batch и batches.meta: все7 official torchvision MD5 PASS; sizes/SHA-256 в ignored manifest |
| Original train |50000 uint8 RGB32×32; labels0..9; ровно5000/class PASS |
| Original test |10000 uint8 RGB32×32; labels0..9; ровно1000/class PASS |
| Class mapping |0airplane,1automobile,2bird,3cat,4deer,5dog,6frog,7horse,8ship,9truck PASS |
| Raw loader |PIL RGB32×32 + Python int labels; обе partitions offline PASS |
| Integrity manifest |`data/raw/cifar10_integrity_v1.json`, schema1,3696 bytes, method official_https_download |
| Manifest SHA-256 |`62cd6967926f0546a09cba8ec6424a0f14f954a5841afa9c419c95e5332d68b7` |
| Recorded verifier |Python3.12.10; torchvision0.24.0+cu126; LF source hashes соответствуют acquisition code50a9fc4 |
| Repeated verify/acquire/verify |Actual opt-in test сравнил SHA-256 и mtime всех raw files до/после: PASS; manifest не переписан, сеть заблокирована |

Manifest связывает bytes/source/counts с UTC acquisition и source hashes; он не
подписан внешней стороной. MD5 — upstream integrity reference, не гарантия
cryptographic authenticity. SHA-256 — фактическая identity локальных bytes, не
независимый official SHA-256 reference. Historical provenance manifest сохраняется
при повторных runs; текущие bytes каждый раз проверяются заново. Test partition
читался исключительно для integrity; scientific decisions по test не выполнялись.
Split/stats hashes остаются null.

### Commands and results

Все команды из root через `.venv/Scripts/python.exe`:

| Команда / check | Результат / граница |
|---|---|
| `-m scripts.download_data --root data/raw` |PASS, exit0, status=verified, archive_verified=true, manifest=created, train50000/test10000 |
| `-m scripts.download_data --root data/raw --verify-only` |PASS в actual checks и отдельно direct module command: exit0, archive_verified=true, manifest=unchanged, train50000/test10000 |
| `-m pytest tests -q --basetemp=.venv/pytest-data001-final` |**180 passed,3 skipped**,52.98s;116 foundation +64 offline data cases; три actual tests default-disabled |
| `RUN_CIFAR_INTEGRATION=1`, затем `-m pytest tests/test_cifar10_local.py -q --basetemp=.venv/pytest-data001-actual` |**3 passed,0 failed,0 skipped**,41.20s; actual counts/contracts/checksums + CLI idempotence, network forbidden |
| `-m pip check` |PASS, No broken requirements found |
| `-m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair` |PASS,2configs, controlled settings equal; preparation only |
| Missing/corrupt/stale/invalid CLI tests |PASS; errors nonzero, invalid arguments exit2, failed verification exit1, foreign/damaged artifacts preserved |
| Git ignore |PASS для archive, extracted batches, manifest и diagnostic staging; raw data/absolute local paths в commits отсутствуют |
| Scope audit against7658894 |Configs/requirements/lock/utils/environment smoke и future ML modules byte-identical; training не запускался |
| Documentation/source evidence audit |PASS:10changed Markdown files, local links/fences,49cards×20fields, acyclic prerequisites,96planned matrix rows; acquisition source/manifest SHA-256 match. Final source edit is a comment clarification only, AST unchanged |

Default SKIP — только explicit actual opt-in guard, не dataset PASS. При opt-in
отсутствие/повреждение dataset приводит к FAIL. Fresh-machine installation и
independent human review не выполнялись. CUDA/training/evaluation checks вне
этого этапа. REPRO-001 остаётся TODO: dataset-only manifest не закрывает general
identity/RNG/atomic checkpoint/resume/backup infrastructure.

### Acquisition diagnostics and Git

Первый full GET был остановлен агентом из-за ошибочно интерпретированного размера
открытого файла в Windows directory metadata. После закрытия обнаружено113246208
bytes, поэтому доказанного source/network failure для этого run не заявляем.
Partial сохранён в ignored `data/raw/.cifar-acquire-mi72s456/cifar-10-python.tar.gz`,
не использован и не помечен verified. Следующий CLI run получил полный archive
проверяемыми1MiB HTTPS ranges (206/Content-Range/length), затем прошёл полный MD5
и7file checks. Его successful staging удалён; diagnostic partial сохранён без
автоматического удаления/подмены.

Все блоки выполнены в main; feature branches, force push, amend/reset отсутствуют.
Перед каждым commit проверены diff, конкретные staged files, diff --check и relevant
tests; git add . / -A не применялись. Уже опубликованы обычным push в origin main:

- `5fd4088` — loader/safe acquisition +40offline cases; suite156PASS.
- `f159fd7` — CLI/manifest + tests/actual opt-in; suite176PASS/3SKIP.
- `50a9fc4` — validated bounded HTTPS ranges +4offline cases; suite180PASS/3SKIP.

Заключительный evidence/docs commit также публикуется обычным push; его hash и
итоговая синхронизация сообщаются в финальном ответе (без self-referential hash).
Human reviewer approval pending; policy/LMS/submitted proposal/team uncertainties
не изменились. Следующий DATA-002: persisted stratified split494 исходного
train50k→45k/5k (4500/500per class), disjoint/exhaustive indices, manifest/hashes
и tamper/idempotence tests. DATA-003 normalization/transforms и scientific runs
здесь не реализованы.

## Historical foundation verification record

Весь текст ниже — сохранённый foundation snapshot до DATA-001; его statements
про отсутствие dataset/pipeline относятся к тому этапу, а не к текущему состоянию.

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
