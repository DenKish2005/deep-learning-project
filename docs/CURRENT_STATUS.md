# CURRENT PROJECT STATUS

## METRIC-002 — 21.11.2026 · technical implementation ready for review

Added strict validation and aggregation for the complete 2-model × 3-seed ×
16-condition matrix. The validator checks core profile/status, unique model/seed/
condition keys, 10,000 samples per condition, shared hashes across runs and
unique hashes across conditions, frozen dataset identities, per-run artifact
hashes, and consistency of the stored
confusion matrix, per-class recalls, macro-F1, and counts. Aggregation produces
per-condition, per-run, per-model, paired-seed, and per-class CSV tables; sample
SD uses `ddof=1`, and treatment differences are paired by seed in percentage
points. The CLI publishes all seven tables as one new output directory and
refuses to overwrite an existing directory.

All **62 tests in `tests/test_metrics.py` passed** in a synthetic CPU-only run
with Python 3.13, NumPy 2.5.5, and pandas 3.1.1. `py_compile` and `git diff --check`
passed. This is technical evidence for METRIC-002, not full acceptance:
EVAL-004 real results/checkpoints are not available, so the 96-row final matrix
and scientific output tables have not been generated. The repository's pinned
course environment is Python 3.12 and needs its own verification. Formal
PROTO-001/SETUP-002 review remains pending.

## METRIC-001 — 19.11.2026 · ready for review

Classification and per-run robustness metrics are implemented in
[`src/metrics.py`](../src/metrics.py). All 57 synthetic metrics tests passed.
The full suite passed with **1,007 passed, 20 opt-in skips, and 0 failures**.
Raw data, split, normalization, and controlled configs passed read-only checks.
Strict frozen CORR-003 replay failed on this runtime. The recorded Python version
and recomputed pixel hash differ, and all 36 protected files match their original
size, SHA-256, and mtime. [Verification record](../report/reproduction.md)

Full METRIC-001 acceptance is pending human review and formal PROTO-001 and
SETUP-002 sign-off. For review, check confusion-matrix orientation and absent-class
behavior, the 16-condition input contract, and fraction versus percentage-point
units. Training, evaluation, and seed aggregation remain open.

## Historical MODEL-001 — 18.11.2026

MODEL-001 — 18.11.2026: CompactCNN v1 independent synthetic technical QA **PASS**;
full DoD **PARTIAL**. Exact288746 parameters, raw logits, protocol init, paired
initial states42/123/2026, BN/dropout/modes/backward/strict state_dict verified.
Default950PASS/0FAIL/20opt-in SKIP; separate executed CPU/CUDA2PASS/0SKIP.
All36protected files retain bytes/SHA-256/mtime; historical DATA/CORR/ROBUST actual
opt-ins not rerun. No training/optimizer steps, CIFAR model inference or EDA.
Human understanding/review, PROTO-001/SETUP-002/003 formal approvals, course/source/
team gates and independent reproduction pending. Paired-init constructor QA does
not establish optimizer budget/full TEST-S01 or training readiness.

Implemented src/models.py with only torch/stdlib dependencies. Caller seeds before
CPU construction; default layer draws precede one final protocol init. No hidden
reseed. Reinitialize explicitly resets BN buffers and consumes caller RNG.
Model state_dict includes `_extra_state` architecture/classes/dropout identity;
load strict=True into a fresh compatible instance, with explicit map_location and
weights_only=True. Standard PyTorch load has no transactional rollback guarantee.

[Evidence](../report/model001_evidence.json),17819bytes, SHA-256
`f658440c8c34425cae6c9e9831dd822c345e7f04a5e1bae39c1e3471677ee362`.
[Commands/results](../report/reproduction.md). CPU/CUDA repeated eval is same-device
only; backend flags were unchanged (deterministic algorithms/cuDNN deterministic
false in these checks). No deterministic-training or cross-device equality claim.

| Task | Full DoD | Remaining |
|---|---|---|
| MODEL-001 | PARTIAL; synthetic technical PASS | Independent human review/understanding, formal PROTO-001/SETUP-002 acceptance |
| ROBUST-001 / DATA-003 | PARTIAL; prior technical PASS preserved | Human/formal dependent acceptance; budget/full scientific controls |
| TRAIN-* / REPRO-001 / METRIC-001 | TODO | Shared trainer; general identity/resume/backup; metrics API |

Published main blocks489eb04 API/tests,11cffd5 modes/strict identity,6dad624 devices/
evidence; each full diff/tests/staged checks and immediate ordinary push PASS,
fresh fetch clean0/0. Final docs/handoff separately published. Next independent
bounded candidate [METRIC-001](NEXT_STEP_PROMPT.md), not started in this stage.

## Historical status snapshots

## ROBUST-001 — 18.11.2026

Independent technical scope **PASS**; full DoD **PARTIAL**. Production
CorruptionPolicy and existing explicit train-only integration verified; src/data.py
and src/utils.py unchanged. Frozen probabilities.50/.17/.17/.16, one event/image,
uniform corrupted severity1..3, identity0; explicit caller-owned keyed generator.
No training/models/evaluation/EDA/core runs or test-based tuning.

Default **914PASS/0FAIL/19opt-in SKIP**; separately **6actual PASS/0FAIL/0SKIP**:
3new policy CPU/CUDA cases,1old train-only reference replay,2grid regressions.
Other13historical opt-ins not repeated (8DATA,3CIFAR corruption,2old CUDA).
Fresh pip/raw/split/stats/config CLI checks PASS. Heavy checks run sequentially;
first parallel reference attempt failed MemoryError and first actual spawn attempt
was stopped after WinError1455. Both unchanged sequential reruns passed; failures
are retained in [evidence](../report/robust001_evidence.json).

100000keyed selections:49914/16835/17223/16028; max frequency error.00223,
conditional severity error.00539976. Actual17fixed persisted train IDs,3seeds,
epochs0/1:102conditions, correct labels/IDs and clean val17samples. NumPy2D/direct
noise/factor oracles: max pixel1.7881393432617188e-7, normalized7.748603820800781e-7.
Actual spawn0/2 batch7/5 equality and global RNG isolation PASS; CUDA RTX3060Laptop
same-device replay PASS, max oracle1.1920928955078125e-7. No CPU/CUDA equality claim.

All28protected files bytes/hashes/mtimes and frozen DATA-002/003 regions preserved.
Old CORR references independently replayed; frozen CORR-003 current-source mismatch
remains an expected error. Authenticated read-only source replay at21ede65 PASS;
PNG/PDF/manifest/evidence unchanged. [Reproduction](../report/reproduction.md).

| Task | Full DoD | Remaining |
|---|---|---|
| ROBUST-001 | PARTIAL; independent technical PASS | Independent human code/artifact review and formal prerequisite approvals |
| DATA-003 | PARTIAL; production robust integration technically verified | Human review/formal dependent acceptance |
| CORR-001/002/003 | PARTIAL; technical PASS | Human review; Person5/team visual review; protocol/course/team approvals |
| REPRO-001 / second-machine reproduction | Open | General identity/checkpoints/resume/backup; independent reproduction |

Published main blocks:ef78e05 frozen replay,0c3d3ca policy/tests,d4d56b0 integration
tests,01bcfb7 actual/evidence; each immediately ordinary push origin main PASS.
Final docs/handoff published separately. Next bounded candidate:
[MODEL-001 after prerequisite review](NEXT_STEP_PROMPT.md). PROTO-001/team freeze,
course/source-policy/LMS/proposal approvals remain pending. Training readiness and
full scientific TEST-S01/init/budget controls are not established by policy checks.

## Historical CORR-003 — 18.11.2026

Independent technical scope **PASS**; full DoD **PARTIAL**. Actual train-only grid:
10 classes x16 conditions, minimum original IDs29/4/6/9/3/27/0/7/8/1 among persisted
train45k, class order0..9. CPU float32 raw [0,1], common local audit epsilon per
image, no geometry/Normalize/auto-contrast or official-test decoding in grid.

Actual [PNG](../outputs/figures/corruption_examples.png)3520x2320,
[PDF](../outputs/figures/corruption_examples.pdf)1page,
[source manifest](../outputs/figures/corruption_examples.source.json),
[evidence](../report/corr003_evidence.json), [commands](../report/reproduction.md).
Default **815PASS/0FAIL/16opt-in SKIP**; separately **2actual PASS/0FAIL/0SKIP**.
Independent numeric160cell replay: noise/darkening bitwise, max blur NumPy2D
error1.7881393432617188e-7. PNG/PDF480RGB samples identical, source error<1/255.
Agent opened both exports: labels readable, progressive effects, no obvious
edge/channel artifact. Dark cat/dog s5 difficult; visual audit is bounded, not
human approval or a claim about model accuracy. Person5/team review **PENDING**.
Missing/corrupt input negative controls correctly fail; no replacement/output.
Existing14actual DATA/corruption/CUDA cases not rerun. Fresh pip/raw/split/stats/
controlled-config CLIs PASS. All23protected files bytes/hashes/mtimes and frozen
DATA-002/003 source regions unchanged; dependencies/configs/protocol untouched.

| Task | Full DoD | Remaining |
|---|---|---|
| CORR-001/002/003 | PARTIAL, independent technical PASS | Independent human code/artifact/visual review, PROTO-001/team/source-policy approvals |
| DATA-001/002 | PARTIAL, technical PASS | Independent human review |
| DATA-003 | PARTIAL, independent technical PASS | ROBUST-001/full robust integration/dependent acceptance and formal/human gates |
| ROBUST-001 / REPRO-001 | TODO | Policy; general identity/checkpoints/resume/backup |

Published main blocks:3c7a46e pixels/tests; b57a3c7 renderer/CLI/tests;
21ede65 actual artifacts/tests/evidence. Each reviewed/tested and immediately
pushed origin main PASS. Final docs/handoff published separately; final hash and
fresh-fetch clean sync in Git/chat. Next bounded technical candidate:
[ROBUST-001 after current prerequisite review](NEXT_STEP_PROMPT.md).
No training/models/evaluator/metrics/EDA/scientific results. General REPRO-001,
course/source-policy/team/protocol and second-machine reproduction stay open;
Graphify remains historical. Training readiness is not established.

## Historical CORR-002 — 18.11.2026

Independent technical scope **PASS**; full DoD **PARTIAL**. Implemented normalized
separable reflect Gaussian blur, multiplicative darkening and apply_corruption.
CPU pixel contract/range/no mutation, clean0 vs three families1..5, explicit noise
RNG and untouched deterministic-family generators verified. CORR-001 noise/keyed
functions, stable_seed, frozen source regions and scientific settings preserved.

Fresh baseline453PASS/11SKIP; final **777PASS/0FAIL/14opt-in SKIP**. Separately
**6actual PASS/0FAIL/0SKIP**: 2new CIFAR dispatcher cases +1new CUDA dispatcher
case +3CORR-001 regressions. All16conditions×17fixed train IDs×epochs0/1 =544
image-conditions; spawn0/2 batch7/5 for clean and severity3 of each family.
CPU/CUDA blur max NumPy2D oracle error1.7881393432617188e-7. Actual CUDA0 on
RTX3060Laptop; canonical keyed epsilon stays CPU, no cross-device bitwise claim.
Missing-input negative control: expected exit1/DataIntegrityError,0SKIP.
Original8actual DATA tests not rerun; fresh raw/split/stats CLIs PASS.

[Evidence](../report/corr002_evidence.json), [reference](../report/corr002_reference.npz)
187438bytes, SHA-256 `b4396faba198a1497758cd3478983b96a85bd272972cb6266e14d497437d1f03`.
Reference: train IDs0/1/2, run42 epoch0, clean0 +3families atseverity3, pixels and
normalized outputs. Existing CORR-001 reference unchanged. All23protected files
bytes/hashes/mtimes preserved; dependencies/backend/configs/protocol unchanged.

| Task | Full DoD | Remaining |
|---|---|---|
| CORR-001/002 | PARTIAL, independent technical PASS | PROTO-001/team freeze/source-policy approvals and independent human review |
| CORR-003 | TODO; next technical stage | Train-only fixed grid, source manifest and human visual audit |
| DATA-001/002 | PARTIAL, technical PASS | Independent human review |
| DATA-003 | PARTIAL, independent technical PASS | ROBUST-001 policy/full integration/dependent acceptance and formal gates/review |
| ROBUST-001 / REPRO-001 | TODO | Policy; general identity/checkpoints/resume/backup |

Published in main: d339179 blur,72c933a darkening,2586a67 dispatcher,20724ac actual
tests/artifacts/evidence; each checked and immediately pushed origin main PASS.
Final docs/handoff is a separate commit; hash/fresh-fetch clean sync in Git/chat.
[Next CORR-003 prompt](NEXT_STEP_PROMPT.md). No models/training/evaluator/metrics/
EDA/plots/scientific runs. Graphify remains historical; not rebuilt. Dependency
edges and human/course/team/protocol/reproduction gates remain open.

## Historical CORR-001 — 18.11.2026

Independent technical scope **PASS**; full DoD **PARTIAL**. Exact severity mapping,
explicit-generator Gaussian noise, `keyed_test_generator` и `keyed_test_epsilon`
реализованы поверх неизменённого stable_seed. CPU reference фиксирует original-test
namespace/IDs, eval100494 и общий epsilon для всех severity/model identities.
Никаких production wrapper/evaluator/policy или training runs не добавлено.

Fresh baseline357PASS/0FAIL/8SKIP; final **453PASS/0FAIL/11opt-in SKIP**.
Отдельно новые actual checks **3PASS/0FAIL/0SKIP**: noise→Normalize на17fixed train
IDs, epochs0/1, batch7/5 и spawn workers0/2; test IDs0/71/9999 только input contract;
CUDA primitive на RTX3060Laptop, same-device equality и wrong-device errors.
Missing actual input проверен negative control: ожидаемый DataIntegrityError/exit1,
без skip. Original8actual data cases в этом этапе не повторялись; их evidence ниже
остаётся историческим. Cross-device/platform bitwise equality не заявлена.

[Evidence](../report/corr001_evidence.json), [reference NPZ](../report/corr001_reference.npz)
97580bytes, SHA-256 `40179be75281bdd06dadd266f565e70ad0f7779e81e31fdc6bd1818b45084b33`.
Все16raw/split/stats/config files сохранили bytes/hashes/mtimes; source regions
DATA-002/003, оба YAML, lock/backend и scientific settings неизменны.

| Task | Полный DoD | Остаток |
|---|---|---|
| CORR-001 | PARTIAL, independent technical PASS | PROTO-001/team freeze/source-policy approvals и independent human review |
| CORR-002 | TODO | Reflect blur, darkening, apply_corruption; следующий технический этап |
| DATA-001/002 | PARTIAL, technical PASS | Independent human review |
| DATA-003 | PARTIAL, independent technical PASS | CORR-001 formal gates, CORR-002/ROBUST-001 full integration/dependent acceptance и human review |
| ROBUST-001 / REPRO-001 | TODO | Policy; general identity/global seed/checkpoints/resume/backup |

Commits7d47569/b73536b/f89d70f/5d12a06 проверены и сразу push origin main PASS.
Финальная документация публикуется отдельно; hash/clean sync — в Git history/ответе.
Dependency edges сохранены. Human/course/LMS/proposal/team и independent reproduction
pending. Следующий [prompt CORR-002](NEXT_STEP_PROMPT.md); Graphify historical, без rebuild.

## Исторический DATA-003 snapshot — 31.10.2026

Independent technical scope **PASS**; full DoD **PARTIAL**. CORR-001 prerequisite
ещё TODO; CORR-002/actual robust integration/dependent acceptance и independent
human review pending. Dependency graph/scientific settings не изменены.

Реализованы compute/prepare/load_normalization, strict schema/provenance и
read-only reuse; IndexedCIFAR10, keyed geometry/order, typed train/val/test loaders,
policy injection boundary и stats CLI. Actual stats только persisted45k, RGB/255,
float64 two-pass, ddof0; independent uint64 moment oracle PASS, max errors2.78e-16
mean/8.33e-17 std. Original test не участвует в statistics.

Stats2429bytes FILE SHA-256
`332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688`;
semantic stats hash `4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877`.
Оба YAML содержат verified FILE hash. NPZ/manifest hashes, mtimes, DATA-002 source
identity и11raw files (включая historical diagnostics/manifest) неизменны.
Full actual batches45k/5k/10k, class counts4500/500/1000 и label/ID coverage PASS;
train352batches, last72. Paired synthetic-policy geometry/order и workers0/2 spawn
bitwise equality на129actual train IDs, epochs0/1, batches64/37 PASS.

Default **357PASS/0FAIL/8opt-in SKIP**, separately actual **8PASS/0FAIL/0SKIP**.
pip/config CLI PASS; existing env/lock/backend сохранены. CUDA smoke SKIP:
backend неизменён. Training/model/corruptions/evaluation/plots/EDA не реализованы
и не запускались. Test images читались только для fixed loader/integrity contract.
[Evidence](../report/data003_evidence.json), [commands](../report/reproduction.md).

| Task | Полный DoD | Результат / остаток |
|---|---|---|
| DATA-001/002 | PARTIAL | Technical PASS; independent human review pending |
| DATA-003 | PARTIAL | Independent technical PASS; CORR-001/002 robust integration, dependent acceptance и human review pending |
| CORR-001/002 | TODO | Noise/severity/blur/darkening APIs отсутствуют; stable_seed уже реализован |
| REPRO-001 | TODO | Только data-specific hashes/publication и stable_seed/seed_worker; general checkpoint/RNG/backup pending |

Published main commits:cdffe3d stats/tests; df6745b keyed loaders/tests;
60464bd stats CLI/tests; de303d0 verified actual artifacts/configs/evidence/tests.
Каждый блок проверен и сразу push origin main PASS. Docs/handoff публикуются
отдельно, final hash — в Git history/ответе. Следующий этап —
[CORR-001 prompt](NEXT_STEP_PROMPT.md). Human/course policy/LMS/submitted proposal,
team freeze и independent reproduction pending. Graphify остаётся historical snapshot.

## Исторический DATA-002 snapshot — 31.10.2026

Local technical acceptance **PASS**; full DoD **PARTIAL**, independent human
artifact/code/test review pending. Initial clean main=origin/main7a2b2f3 verified
by fetch. Existing `.venv`, lock/backend/scientific settings сохранены.

`create_split`/`load_split` и CLI `--prepare-split`/read-only `--verify-split`
реализованы. Actual original train50000→45000/5000, seed494, exactly4500/500 per
class; sorted `<i8` indices0..49999, unique/disjoint/exhaustive и exact protocol
replay PASS. Original test10000 не участвует в split. Manifest проверяет source,
label order, arrays/NPZ hashes и LF creation-code identity; partial/tampered states
отвергаются, failure diagnostics сохраняются. Reuse hashes/file mtimes PASS;
11raw files, включая historical acquisition manifest и diagnostics, не изменены.

NPZ76550 bytes: `1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284`.
Оба YAML содержат этот file-byte hash; normalization_sha256 остаётся null.
[Manifest](../data/splits/split_manifest.json), [actual evidence](../report/data002_evidence.json),
[commands/results](../report/reproduction.md). Default282PASS/0FAIL/5opt-in SKIP;
отдельно actual5PASS/0FAIL/0SKIP; pip/config CLI PASS. CUDA smoke не повторялся,
dependencies/backend не менялись. Default suite offline и не требует actual data/split.

Published in main:4044d0c algorithm/tests; fd0a272 persistence/provenance/tests;
f4d7b42 CLI/tests; cf8e684 actual artifacts/configs/evidence/tests. Каждый commit
проверен и сразу опубликован обычным push. Финальный docs commit — в Git history.

| Task | Полный DoD | Результат / остаток |
|---|---|---|
| DATA-001/002 | PARTIAL, technical acceptance PASS | Raw data + frozen split verified; independent human review pending |
| DATA-003 | TODO | Normalization/transforms/loaders отсутствуют; prerequisite CORR-001 ещё TODO |
| REPRO-001 | TODO | Только минимальные dataset/split hash/write helpers; general identity/RNG/checkpoint/backup infrastructure не закрыта |

Следующий конкретный этап — [DATA-003 prompt](NEXT_STEP_PROMPT.md): stats только
по persisted45k train, deterministic keyed transforms/order и loaders в границах
зависимостей. Training readiness не достигнута; scientific runs не запускались.
Policy/LMS/submitted proposal/team/independent reproduction остаются pending.
Graphify — historical snapshot до DATA-002, сверять с current source; rebuild не выполнялся.

## Исторический DATA-001 snapshot — 30–31.10.2026

Работа идёт непосредственно в `main`; initial fetch подтвердил main=origin/main
на7658894, исходное рабочее дерево clean. AGENTS.md в repository и родительской
цепочке не найден. Foundation dependencies/backend/configs сохранены.

Реализованы raw CIFAR-10 loader с offline default, official HTTPS acquisition,
archive/seven-file checksums до unpickle, оба partition contracts, working CLI
`--root`/`--verify-only` и ignored versioned integrity manifest. Повторный acquire
переиспользует проверенные данные; verify-only не пишет artifacts; corrupt/foreign/
stale artifacts не заменяются. Default suite:180 PASS,3 actual opt-in SKIP.
Actual official archive170498071 bytes и7files прошли MD5/SHA-256 inspection;
train50000/test10000, RGB32×32,10classes,5000/1000per class PASS. Отдельные actual
tests:3PASS/0SKIP, включая read-only/idempotence hashes+mtime. Acquisition manifest
schema1 сохранён ignored с actual source/provenance. Команды/checksums — в
[reproduction record](../report/reproduction.md); default SKIP не означает CIFAR PASS.

| Task | Полный DoD | Результат / остаток |
|---|---|---|
| DATA-001 | PARTIAL, local technical acceptance PASS | Loader/CLI/manifest,180offline PASS +3actual PASS; independent human artifact/code/test reviewer approval pending |
| DATA-002/003 | TODO | Split/stats/transforms/loaders отсутствуют; оба config hashes остаются null |
| REPRO-001 | TODO | Только минимальный dataset manifest helper; generic identity/RNG/atomic checkpoint/backup infrastructure не реализована |

Training/model/evaluation/corruptions/plots не реализованы и не запускались.
Test partition разрешён только для integrity inspection, без scientific tuning.
Policy/LMS/submitted proposal/roles и human approvals остаются pending.
Следующий конкретный этап — DATA-002: persisted stratified split494 исходного
train50k→45k/5k,4500/500 per class, disjoint/exhaustive indices, manifests/hashes
и tamper/idempotence tests; test partition не участвует в split decisions.

## Обновление Git workflow

После foundation пользователь разрешил push и потребовал все следующие commits
и push только в `main`. Foundation commits перенесены в локальную `main`
fast-forward без переписывания истории. Snapshot этапа ниже описывает предыдущее
состояние до публикации; актуальную синхронизацию проверять через Git.
[Промпт следующего этапа DATA-002](NEXT_STEP_PROMPT.md) обновлён31.10.2026:
persisted split494, actual verification и маленькие проверенные commit/push только
в main. DATA-001 scope сохранён в истории Git; DATA-002 здесь ещё не выполнялся.

## Историческое состояние после foundation — 30.10.2026

База продолжения: commit0633f67, исходная ветка main была clean; последующих
commits при старте не было. Создана `codex/setup-foundation`, локальные commits
по логическим блокам; push/merge/fetch не выполнялись. Remote freshness неизвестна.
Применимых AGENTS.md в repository/родительской цепочке нет.

| Task | Статус полного DoD | Фактически выполнено / осталось |
|---|---|---|
| SETUP-001 | PARTIAL | Найдены local guidelines2026, CSR, 2proposal candidates; реестр hashes/pages. Pending: LMS freshness, detailed rubric, receipt/version, policy precedence, точные cutoff/slot |
| PROTO-001 | PARTIAL | Technical architecture/protocol review и corrections записаны; team approval/freeze pending |
| TEAM-001 | PARTIAL | Role/file plan проверен, имена найдены в candidate; mapping/availability/backup/reviewer acceptance pending |
| SETUP-002 | PARTIAL, local technical acceptance PASS | `.venv`,38 exact versions, recipe, pip check/imports/CPU+CUDA smoke PASS; formal human review pending |
| SETUP-003 | PARTIAL, local technical acceptance PASS | 2YAML, strict schema/config CLI,116 tests PASS; formal human review/protocol acceptance pending |

Это завершённый **локальный технический объём текущего этапа**, не team sign-off.
[Evidence](../report/reproduction.md), [environment snapshot](../report/setup_environment.json),
[source registry](../report/course_materials/README.md), [decisions](SETUP_DECISIONS.md).

Реализованы только config functions в src/utils.py, test_utils и environment smoke.
Данные не скачивались, test set не читался, CNN/training/evaluation/plotting не
реализованы и не запускались. Split/stats paths декларативные, hashes null.
REPRO-001 TODO (ignore protection добавлена, identity/manifests/backup ещё нет).
Следующий технический этап: DATA-001 → DATA-002 → DATA-003 с REPRO-001 primitives;
course policy/team approvals уточняются отдельно до соответствующих gates.

## Исторический аудит до planning commit

Весь текст ниже сохраняет snapshot на 5ce6b1a до создания `docs/`; его размеры,
версии, статусы и команды не описывают текущую рабочую ветку. Обследованы все29
исходных tracked файлов. Текущее состояние и проверки перечислены выше.

## Наблюдаемые факты

- Ветка `main`, HEAD `5ce6b1a` — `Architecture created`, 22.09.2026 09:48 +0500. Предыдущий commit `db34ee3` — `Initial commit`, 17.09.2026 22:11 +0500.
- Из локальных веток только main; видны `origin/main` и `origin/HEAD`, совпадающие с HEAD. Fetch не выполнялся, актуальность remote не установлена.
- Remote: `https://github.com/DenKish2005/deep-learning-project.git`. До аудита `git status --short` пустой.
- 20 из 29 файлов имеют размер 0 bytes; из них пять — ожидаемые `.gitkeep`, два — допустимые package markers. Остальные 13 — отсутствующая реализация и пустой proposal.
- 14 Python-файлов: 12 source/script/test файлов и два `__init__.py`; все пустые. Нет функций, training loop, моделей, работающих CLI и assertions.
- Нет datasets, checkpoints, metrics, figures и course documents, включая PDF/DOCX/PPTX. В ignored каталогах тоже только `.gitkeep`.
- Notebook — валидная структура nbformat 4 с `cells: []`, без EDA или execution results.
- Оба YAML содержат только comment. YAML parser вернёт `None`, а не config mapping; команды обучения сейчас нельзя считать рабочими.
- `requirements.txt` содержит восемь имён без versions: torch, torchvision, numpy, pandas, scikit-learn, matplotlib, tqdm, pyyaml. pytest и lock отсутствуют.
- Найден Python 3.12.10. Через package metadata найдены torch 2.14.0, torchvision 0.29.0, numpy 2.5.3, PyYAML 6.0.3; pytest не найден. Это **не проверка imports, ABI, CUDA или пригодности окружения**. GPU/VRAM не проверялись.
- AGENTS.md в репозитории и проверенной цепочке родительских каталогов не обнаружен.

## Реестр каждого исходного файла

В столбце «Проверка/DoD» указан будущий acceptance criterion, а не уже пройденная проверка. Контракты и зависимости раскрыты в [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

| Путь | Bytes / текущее назначение и статус | Что должно содержаться / изменение | Зависимости | Проверка / DoD |
|---|---|---|---|---|
| README.md | 436; частично: описание scaffold | Setup → data → train → eval → plots, ссылки на docs и artifacts | Все CLI | Новый участник выполняет smoke по README |
| ARCHITECTURE.txt | 1114; реализовано как дерево scaffold | Синхронизировать с финальным деревом, без заявлений о готовности | Architecture doc | Все пути дерева существуют после реализации |
| LICENSE | 1095; полный MIT license | Сохранить; dataset/paper licenses учитывать отдельно | Нет | Текст сохранён, license не выдается за dataset license |
| requirements.txt | 67; частичный список | Версии и проверенный lock, pytest/dev dependencies | Env validation | pip check + imports + smoke |
| .gitignore | 460; базовые ignore rules реализованы | Добавить outputs/runs, predictions, cache, locks при необходимости; splits и curated tables не терять | Artifact layout | git check-ignore на данных и разрешённых manifests |
| configs/clean_cnn.yaml | 42; comment placeholder | Полный config с schema_version | utils/train/data | parse + invalid config tests |
| configs/robust_cnn.yaml | 43; comment placeholder | Тот же config, отличается только regime и policy | clean config/corruptions | Controlled fields equality test |
| data/raw/.gitkeep | 0; реализованный directory marker | Сохранить; raw CIFAR отдельно от Git | download_data | checksum + class counts |
| data/processed/.gitkeep | 0; directory marker | Сохранить; cache optional и versioned | split/corruptions | Cache hash соответствует manifest |
| outputs/checkpoints/.gitkeep | 0; directory marker | Сохранить; best.pt/last.pt per run | train | load round trip, hashes |
| outputs/metrics/.gitkeep | 0; directory marker | Сохранить; 96 summaries, aggregate tables | evaluate/metrics | Уникальность matrix keys, cardinality |
| outputs/figures/.gitkeep | 0; directory marker | Сохранить; PNG/PDF figures | plots/metrics | Rendering + source provenance |
| src/__init__.py | 0; допустимый package marker | Можно оставить пустым, без side effects | Нет | import src успешен |
| src/data.py | 0; не реализован | CIFAR loader, persisted split, transforms, datasets, loaders | torchvision/corruptions | test_data_pipeline |
| src/corruptions.py | 0; не реализован | Pixel-space corruption + training policy | torch/torchvision | test_corruptions |
| src/models.py | 0; не реализован | CompactCNN, init, architecture ID | torch.nn | test_models |
| src/train.py | 0; не реализован | Shared training engine + CLI, validation, resume | data/models/utils | test_training + smoke |
| src/evaluate.py | 0; не реализован | Fixed conditions inference, predictions, summary | data/models/metrics | test_evaluation |
| src/metrics.py | 0; не реализован | Metrics, paired aggregation, table exports + CLI | numpy/pandas | test_metrics |
| src/utils.py | 0; не реализован | Config validation, seeds, atomic artifacts, metadata | YAML/stdlib/torch | test_utils |
| scripts/download_data.py | 0; не реализован | Download/integrity/split/stats CLI | src.data | Offline error + idempotence |
| scripts/run_all_evaluations.py | 0; не реализован | Matrix validation, eval orchestration | src.evaluate | Exactly 96 unique outputs |
| scripts/make_plots.py | 0; не реализован | Reusable plots from saved tables | metrics/data/matplotlib | test_plots |
| tests/__init__.py | 0; допустимый marker | Можно оставить пустым | pytest | Collection succeeds |
| tests/test_data_pipeline.py | 0; тестов нет | Split, range, transform, loader tests | data fixtures | Реальные assertions, no downloads by default |
| tests/test_corruptions.py | 0; тестов нет | Identity/range/determinism/severity tests | corruptions fixtures | Все типы/severity и invalid inputs |
| notebooks/01_eda_and_sanity.ipynb | 72; пустой notebook | EDA только train, class balance, sample grids; без production logic | src.data | Restart & Run All без скрытого state |
| report/proposal.md | 0; отсутствует содержимое | Восстановить реальный submitted proposal с provenance, не сочинять прошлую сдачу | Teammates/LMS | Source/date/status подтверждены |
| report/final_report/README.md | 67; placeholder | Report plan, source/build/export links | FINAL_DELIVERABLES | Готовый report соответствует rubric |

## Классификация готовности

**Already implemented:** skeleton directory layout, Git repository, MIT text, базовые ignore rules, краткое описание проекта и package/directory markers. Это не рабочая ML-система.

**Partially implemented:** README, dependency declaration, architecture tree, место для report. YAML и notebook имеют только структуру, полезная функциональность отсутствует.

**Not implemented:** data pipeline, split persistence, corruptions, CNN, training, robust policy, evaluation, metrics, plots, tests, experiment tracking, report, slides и reproducibility evidence.

**Requires verification:** submitted proposal, latest rubric, deadlines/timezone, presentation date/duration, AI policy, teammates' names, GPU availability, import/CUDA compatibility, доступ к dataset, branch freshness.

**Out of scope:** training и реализация сейчас; frontend/cloud/deployment; обязательный SOTA; adversarial robustness. OPTIONAL extension CIFAR-10-C не блокирует core.

## Проблемы и блокеры

| ID | Наблюдение | Последствие | Следующая задача |
|---|---|---|---|
| AUD-01 | Нет официальных документов и пустой proposal | Нельзя подтвердить оценки/даты/AI permissions | SETUP-001 |
| AUD-02 | Пустые configs и source | Не существует исполняемого experiment | SETUP-003 → DATA/MODEL/TRAIN |
| AUD-03 | Unpinned dependencies; pytest отсутствует | Нет общего проверенного env | SETUP-002 |
| AUD-04 | Split, normalization, RNG и checkpoint schema отсутствуют | При наивной реализации риск leakage и несопоставимости | DATA-002, DATA-003, TRAIN-002 |
| AUD-05 | Raw outputs целиком ignored | Потеря артефактов без внешней копии | REPRO-001, RELEASE-001 |

Дублирующего рабочего кода, circular imports и фактической утечки данных не обнаружено, поскольку кода и данных нет. Это не доказательство корректности будущей реализации. Реальных runtime bugs и accuracy пока оценить нельзя.

## Выполненные и невыполненные проверки

Выполнены `rg --files --hidden`, recursive listing with sizes, чтение всех непустых файлов, проверка zero-byte файлов, `git status`, `git branch -avv`, `git log -8`, `git remote -v`, `git ls-files`, `python --version` и package metadata inspection. Эти операции не изменяли исходные файлы и Git.

Не выполнялись: installation, imports torch/torchvision, dataset download, notebook execution, pytest, training, evaluation и model benchmarks. Плановые команды в документации не являются отчётом об успешном запуске.

После планирования добавлены только документы `docs/*.md`. Исходные 29 файлов должны остаться byte-identical; итоговая проверка перечислена в MASTER_PROJECT_PLAN.

## Проверка созданного planning package

Фактически выполнено 30.10.2026:

- Все 14 требуемых Markdown-документов существуют, непустые и читаются как UTF-8.
- Локальные Markdown-ссылки ведут на существующие документы; code fences сбалансированы.
- В roadmap ровно 49 уникальных task cards, каждая содержит все 20 обязательных полей; dependency graph не содержит неизвестных prerequisite IDs или cycles.
- В protocol ровно 96 уникальных matrix rows: 2 regimes × 3 seeds × 16 conditions. Для каждого run присутствуют clean0 и все 15 corruption conditions.
- Все 29 исходных tracked файлов совпадают с HEAD после штатных Git clean filters; `git diff --exit-code` и `git diff --check` проходят. У LICENSE checkout использует CRLF при LF в Git blob из-за `core.autocrlf=true`; это не изменение файла этой задачей.
- `git status --short` показывает только `?? docs/`. Commits, push, branch changes и remote mutations не выполнялись.

Это проверка документации и неизменности исходников. ML unit/integration tests, imports/ABI/CUDA checks, data download, training и evaluation по-прежнему **не выполнялись**.
