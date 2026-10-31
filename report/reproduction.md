# Verification record — foundation and DATA-001..003

## DATA-003 independent technical acceptance — 31.10.2026

**PASS independent technical scope; PARTIAL full DoD.** CORR-001 prerequisite
remains TODO, CORR-002/actual robust integration/dependent acceptance and human
review pending. Agent checks do not replace team/course approval or REPRO-002.
Initial clean main=origin/main517b64f verified by fetch,0ahead/behind, staged empty;
no applicable AGENTS.md. Graphify historical navigation checked against source,
not rebuilt. Existing Python3.12.10/NumPy2.5.4/torch2.9.0+cu126 env unchanged.

### Actual stats and loader evidence

Only frozen45000 original-train IDs enter raw RGB uint8/255 float64 two-pass
population moments. No val/test pixels enter stats; tests prevent test decoding
on stats/train-val paths. Mean:
`[0.4913397271582242,0.48209963252314836,0.44655201729302807]`;
std:`[0.2469009494849942,0.24334132835714897,0.2615066322799661]`.
Independent uint64 moment oracle (different chunk137 vs256) at atol1e-12 PASS,
max mean error2.7755575615628914e-16, std8.326672684688674e-17.

| Artifact / identity | Actual result |
|---|---|
| Stats | data/splits/normalization.json, schema1,2429bytes; full precision |
| Stats FILE SHA-256 | `332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688` |
| Stats semantic SHA-256 | `4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877` |
| Train array semantic SHA-256 | `7d59b48de10d46620a62bdc85d9d3dd62a8c2a2b902617c6807569f4a0a2ee41` |
| Selected uint8 train pixels semantic SHA-256 | `f421843aa7eddc42d869105fb764b6d533b1d6e4d533458a30ebdbf6b70d26a6` |
| Stats creator LF region SHA-256 | `1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c` |
| Frozen NPZ FILE SHA-256 | `1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284`, unchanged |
| Frozen manifest FILE SHA-256 | `8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21`, unchanged |
| Full typed coverage | train45000/val5000/test10000; float32 NCHW, int64 labels/IDs, unique complete IDs and label alignment; class4500/500/1000 |
| Batches | train352(last72), val20(last136), test40(last16); no dropped samples |
| Paired boundary / Windows spawn | synthetic identity policy only; epochs0/1, workers0/2,129actual train IDs, batches64/37 bitwise equal |
| Preservation | All13initial raw/split files hashes/bytes/mtime_ns identical, including11raw files with historical manifest/diagnostics; stats reuse unchanged |

[Compact actual evidence](data003_evidence.json) contains exact provenance,
before/after snapshots, source hashes and normalized tensor/order stream SHA-256
for all three partitions. Image stream hash concatenates batch NCHW little-endian
float32 bytes in yielded order; ID hash uses split semantic int64 convention.
This records deterministic data mechanics, not performance metrics. Test images
were inspected only for the fixed loader/integrity contract, never tuning.

### Commands / PASS, FAIL, SKIP

All commands use `.venv/Scripts/python.exe` from repository root.

| Command / check | Actual result |
|---|---|
| `-m pip check` before/after | PASS, No broken requirements found |
| Baseline `-m pytest tests -q --basetemp=.venv/pytest-data003-baseline` |282PASS/0FAIL/5SKIP,93.07s |
| `-m scripts.download_data --root data/raw --verify-only` / `--verify-split` before implementation | PASS exit0, official raw50k/10k, frozen45k/5k, correct NPZ hash |
| `-m scripts.prepare_normalization --root data/raw` and `--verify-only` | PASS: create, repeated prepare/reuse, read-only verification; train only |
| `-m pytest tests -q --basetemp=.venv/pytest-data003-final` |**357PASS/0FAIL/8SKIP**,137.61s console (137.44s JUnit suite) |
| RUN_CIFAR_INTEGRATION=1, RUN_CIFAR_SPLIT=1, RUN_CIFAR_NORMALIZATION=1; `-m pytest tests/test_cifar10_local.py tests/test_split_local.py tests/test_normalization_local.py -q -s --basetemp=.venv/pytest-data003-actual` |**8PASS/0FAIL/0SKIP**,163.90s console (163.91s JUnit suite) |
| `-m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair` | PASS,2configs; only stats FILE identity added symmetrically; schema alone does not authenticate artifacts |
| Scope / staged bytes | PASS, frozen DATA-002 creator/hash/mtime and excluded modules/old CLI/requirements/lock unchanged; staged stats bytes exactly equal evidence FILE hash |
| Final docs/evidence QA | PASS:13Markdown files,96local links/fences/portable paths;49cards×20fields; prerequisite edges unchanged/acyclic;96unchanged matrix rows; source/artifact hashes/mtimes and config controls verified |
| CUDA smoke | SKIP: unchanged dependencies/backend |
| Model/training/eval/EDA/scientific runs | SKIP: outside DATA-003 scope |

Eight default SKIPs are explicit actual opt-ins, not actual PASS claims. Actual
opt-in missing/corrupt raw/split/stats FAIL. Default tests are independent of those
artifacts. TEST-C08 here verifies a synthetic callable boundary only; no production
corruption implementation or actual robust acceptance. Stable seed/worker helpers
are minimal REPRO-001 contributions; general global seeding/checkpoint/resume/backup
remains TODO. Readiness lock is not a power-loss transaction. Hashes are integrity
identities, not signed authenticity. Local locked CPU equality does not promise
cross-version/platform or model reproducibility. Human/policy/LMS/proposal/team
and independent fresh-machine reviews remain pending; no training-ready claim.

### Published logical blocks

Every block: full diff/relevant tests, exact staged paths and full staged diff,
both diff --check checks, separate main commit and immediate ordinary push.

| Commit | Block | Push |
|---|---|---|
|cdffe3d | Train-only normalization/schema/provenance/failure safety +48offline tests | PASS origin main |
|df6745b | Keyed transforms/order/loaders/seed +20offline tests, real spawn2 | PASS origin main |
|60464bd | Stats prepare/read-only verify CLI +7offline tests | PASS origin main |
|de303d0 | Actual stats, symmetric configs,3actual tests, evidence;357offline+8actual PASS | PASS origin main |

Documentation/handoff is a separate checked commit; final hash/push/clean sync
are reported in chat/Git history without a self-referential commit hash. Next
concrete task: CORR-001 noise/severity using existing stable_seed, then CORR-002;
preserve DATA-002/003 artifacts and versioned regions.

## Historical DATA-002 technical acceptance — 31.10.2026

**PASS technical acceptance; PARTIAL full DoD.** Исполнитель/reviewer checks —
Codex agent по прямому запросу пользователя. Independent human review,
second-machine reproduction, course policy/source/team sign-off pending.
Работа начата с clean main7a2b2f3; fetch подтвердил main=origin/main,0ahead/0behind.
Применимого AGENTS.md не обнаружено. Graphify snapshot использован для навигации,
source contracts сверены по текущим files; graph не перестраивался.

Existing Python3.12.10/NumPy2.5.4 `.venv`, exact lock, backend и scientific settings
сохранены. Из YAML изменён только dataset.split_sha256 симметрично; normalization
hash остаётся null. CUDA smoke не повторялся, поскольку env/backend не менялись.
Training/normalization/loaders/corruptions/model/evaluation/EDA не выполнялись.

### Actual persisted split evidence

Actual source — verified official original train50000, original namespace
`cifar10-python/original-train`, labels0..9/5000per class. Единственный local
default_rng494, class order0..9, first500/class validation,4500train, final sort.
Test10000 не участвует в split; test files могут хешироваться для raw integrity,
но split CLI не декодирует test images/targets. DATA-001 actual tests отдельно
декодируют оба original partitions только для integrity.

| Artifact / invariant | Actual result |
|---|---|
| NPZ | `data/splits/cifar10_split_v1.npz`,76550bytes; sorted `<i8` train_indices[45000]/val_indices[5000] |
| NPZ file SHA-256 | `1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284` |
| Manifest | `data/splits/split_manifest.json`,schema1,2539bytes; file SHA-256 `8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21` |
| Train semantic SHA-256 | `7d59b48de10d46620a62bdc85d9d3dd62a8c2a2b902617c6807569f4a0a2ee41` |
| Validation semantic SHA-256 | `40ddd3d462690ac2f02790e1bacc3365bc83936fa48c74462e90a55552b93f13` |
| Original label-order semantic SHA-256 | `bba2873f62cbf0374dffec49080d49cdf34353988e10b1410f20e65059eb7535` |
| Creation source identity | LF source region src/data.py:DATA-002-v1, `4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96`; implementation revision f4d7b42 (region unchanged since fd0a272) |
| Class counts | Exactly4500train/500validation for each0..9 |
| IDs/invariants | Range0..49999, sorted/unique, disjoint, exhaustive union50000, label alignment and independent protocol replay PASS |
| Source archive |170498071bytes, MD5 c58f30108f718f92721af3b95e74349a; SHA-2566d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce |
| Reuse | create/load + CLI prepare/verify + raw verify: split hashes and file mtimes identical;11raw files including preserved acquisition diagnostic unchanged |
| Historical raw manifest | SHA-25662cd6967926f0546a09cba8ec6424a0f14f954a5841afa9c419c95e5332d68b7 and mtime unchanged; not rewritten |

Source train batch SHA-256/MD5/size, creator, every invariant and exact before/after
mtime_ns/hash snapshots — [data002_evidence.json](data002_evidence.json). Full
[schema/hash/write contract](../docs/TECHNICAL_ARCHITECTURE.md#split-artifact-contract-v1).
Semantic identity hashes compact sorted JSON dtype/shape + LF + C-order bytes;
NPZ SHA-256 hashes file bytes. `.gitattributes` preserves binary NPZ/LF JSON.
Fresh-process byte-identical synthetic regeneration observed in this locked env;
cross-version/platform regeneration not guaranteed. Saved actual indices canonical.

### Commands and actual results

Из repo root, interpreter `.venv/Scripts/python.exe`:

| Command / check | PASS / FAIL / SKIP |
|---|---|
| `-m pip check` before/after | PASS, No broken requirements found |
| `-m pytest tests -q --basetemp=.venv/pytest-data002-baseline` |180PASS/0FAIL/3opt-in SKIP,50.45s |
| `-m scripts.download_data --root data/raw --verify-only` before/after | PASS exit0, train50000/test10000, archive_verified=true, raw manifest unchanged |
| `-m scripts.download_data --root data/raw --prepare-split --split-seed 494` | PASS exit0, actual NPZ/manifest created, repeated run reused |
| `-m scripts.download_data --root data/raw --verify-split` | PASS exit0, read-only; also verified module entry point in fresh process |
| `-m pytest tests -q --basetemp=.venv/pytest-data002-final` |**282PASS/0FAIL/5SKIP**,104.69s;116config +166offline data, no actual-data/split dependency |
| `RUN_CIFAR_INTEGRATION=1`, `RUN_CIFAR_SPLIT=1`; `-m pytest tests/test_cifar10_local.py tests/test_split_local.py -q --basetemp=.venv/pytest-data002-actual` |**5PASS/0FAIL/0SKIP**,67.61s (3raw +2split), offline actual inputs |
| `-m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair` | PASS,2configs, shared settings equal; schema check does not authenticate artifacts |
| Tamper/partial/failure/CLI argument checks | PASS: actionable DataIntegrityError/exit1, invalid flags exit2; preserved failures, explicit recovery to empty directory |
| Git ignore/attributes/staged bytes | PASS: indices/manifest tracked; raw/env/caches/diagnostics ignored; staged artifact bytes exactly match evidence SHA-256 |
| Final docs/evidence QA | PASS:13changed Markdown files,84local links/fences/portable paths,49cards x20fields/acyclic dependencies,96unchanged protocol matrix rows; frozen artifact bytes/mtimes/creator and unchanged scientific settings/lock/future modules |
| CUDA smoke / scientific runs | SKIP, unchanged backend / outside DATA-002 scope |

Default SKIPs — explicit opt-in guards, не actual-data PASS. Opted-in missing/corrupt
inputs FAIL. Global NumPy/Python/torch RNG states unchanged. API validates supplied
labels and expected protocol/source references; actual source certification comes
from verified raw loader. Hashes/MD5 are integrity identities/references, not signed
external authentication. Readiness lock protects two-file publication but is not a
multi-file atomic filesystem/power-loss guarantee; interrupted locks/staging/pairs
are rejected/preserved. General REPRO-001 identity/checkpoint/RNG/backup remains TODO.

### Published logical blocks

Each block had full diff review, exact staged list, git diff --check and cached
--check, relevant tests; implementation/tests together, immediately ordinary push
origin main. No feature branch, force push, amend/reset or history rewrite.

| Commit | Block / relevant result | Push |
|---|---|---|
|4044d0c | Exact construction/invariants + offline tests;96data PASS, final focused31PASS | PASS main→origin/main |
|fd0a272 | Persistence/manifest/provenance/failure safety + tests/contracts;149data PASS +1publication PASS | PASS main→origin/main |
|f4d7b42 | Offline split CLI + tests/command docs;282PASS/3SKIP | PASS main→origin/main |
|cf8e684 | Actual artifacts/config identity/evidence/actual tests;282PASS/5SKIP +5actual PASS | PASS main→origin/main |

Final docs/evidence consistency checks and DATA-003 prompt are a separate commit;
its hash/push and final clean main=origin/main status are reported in the chat
(no self-referential commit hash). Next DATA-003:45k-only normalization and keyed
transforms/loaders; CORR-001 prerequisite still TODO, so robust integration cannot
be declared complete. Human/course/source/team gates remain pending.

## Historical DATA-001 technical acceptance — 30–31.10.2026

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
