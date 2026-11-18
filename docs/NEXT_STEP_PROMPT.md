# Промпт следующего этапа — MODEL-001 prerequisite review и bounded model API

Обновлён18.11.2026 после ROBUST-001 independent technical PASS/full DoD PARTIAL.
Published main:ef78e05 frozen CORR-003 replay,0c3d3ca policy/tests,d4d56b0
train-only integration tests,01bcfb7 actual/evidence. Каждый блок сразу ordinary
push origin main PASS, clean0ahead/behind. Final docs/status/ai_usage/handoff
опубликованы в37731e4; fresh fetch подтвердил clean main=origin/main. Сохранение этого prompt
не запускает MODEL-001 и не разрешает training.

~~~text
Продолжи проект CSCI494 — Robustness to Noise.
Работай, коммить и пушь только в существующую ветку с точным именем main.
Не создавай ветку codex, codex/*, любые другие branches или worktrees.
Исторические branch/PR советы в документах не переопределяют это правило.
Разрешаю самостоятельные commits и ordinary git push origin main после каждого
проверенного небольшого блока без повторного согласования. Без force, amend/reset, history rewrite или
автоматического затирания divergence/conflicts. Failed push сначала расследуй;
safe fast-forward допустим, automatic merge/conflict overwrite запрещены.

Следующий bounded technical candidate — MODEL-001: prerequisite review, затем
CompactCNN v1 и independent synthetic model tests. Formal prerequisites PROTO-001
и SETUP-002 имеют pending human/course/team approvals; не объявляй их закрытыми
по агентским tests. Independent bounded technical preparation допустима, full
dependent acceptance остаётся отдельным gate. Не запускай training/optimizer
updates, overfit/pilot/core runs, evaluator/metrics/EDA/новые scientific figures,
TRAIN-* или broad general REPRO-001. Unit forward/backward проверка не training run.

СНАЧАЛА CURRENT REPOSITORY И PREREQUISITES

Прочитай AGENTS.md при наличии, README, docs/MASTER_PROJECT_PLAN.md,
START_HERE.md, CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md,
EXPERIMENT_PROTOCOL.md, IMPLEMENTATION_ROADMAP.md (MODEL-001, PROTO-001,
SETUP-002/003, ROBUST-001, DATA-003, TRAIN-001, REPRO-001), TESTING_PLAN.md,
REPRODUCIBILITY.md, SETUP_DECISIONS.md, report/reproduction.md,
report/robust001_evidence.json, report/corr001_evidence.json, corr002_evidence.json,
corr003_evidence.json, report/ai_usage.md, report/course_materials/README.md,
оба YAML, split_manifest.json и normalization.json. Current source governs;
Graphify historical pre-DATA-002 snapshot, no rebuild.
Проверь branch/status/staged/log, fresh fetch, exact main=origin/main/history.
src/models.py пока placeholder. Existing data/corruption source и policy не менять.

Existing .venv без reinstall/update/global/backend changes. Heavy checks strictly
sequential: overlapping pytest processes в ROBUST-001 вызвали MemoryError и
WinError1455; unchanged sequential reruns PASS. Перед edits:
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m pytest tests -q --basetemp=.venv/pytest-model001-baseline
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-split
.venv/Scripts/python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.verify_frozen_corr003
.venv/Scripts/python.exe -m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair

Сохрани raw/split/stats/configs/lock/protocol, CORR references/grid/manifest и все
existing evidence bytes/hashes/mtimes. Actual missing/corrupt inputs — FAIL,
без skip/download/repair/synthetic substitution. Не менять scientific settings,
prerequisite edges, DATA-002/003 frozen regions, stable_seed или old functions.

СОХРАНЁННЫЙ ROBUST/DATA/CORR CONTRACT

- CorruptionPolicy принимает только probabilities[.50,.17,.17,.16], Python max3;
  immutable settings. Один event/image, severity1..3uniform, identity0; no quotas/
  compositions/tuning. Caller-owned explicit same-device Generator; no new seeds.
- Draw order: float64 scalar torch.rand event intervals[0,.50),[.50,.67),
  [.67,.84),[.84,1); только corruption int64 randint(1,4,()); только noise после
  этого float32 randn(x.shape). Metadata event/severity audit-only. Validation
  before draws; raw finite float32 RGB CHW, CPU/CUDA, policy H,W>4 for possibleblur.
- Existing train hook raw→keyed geometry→float32/255→policy→Normalize; explicit
  injection gate/val-test rejection. Geometry/order/train_corruption keys и
  original-test seed100494 helpers unchanged. Training requests only train/val.
- Current default914PASS/0FAIL/19opt-in SKIP; separately6actual PASS/0SKIP:
  3new policy CPU/CUDA +1old train-only reference replay +2frozen grid. Other13
  opt-ins not repeated, retain old dates. Actual17train IDs0..11,13..17,
  seeds42/123/2026 epochs0/1;102conditions, spawn0/2 batch7/5seed42 PASS.
- CORR-003 strict CURRENT source verify intentionally exits1/mismatch after
  src/corruptions.py edits. Do not regenerate frozen artifacts or weaken checks.
  Bounded scripts.verify_frozen_corr003 authenticates4implementation blobs at
  recorded21ede652d2d1638ec69e4637f4707bcebf3c8224 via git show in ignored tmp,
  then invokes original read-only verifier. No branch/worktree/data copy.
- Evidence report/robust001_evidence.json23235bytes SHA256
  f306f13c5b560f6ba870c4b094200945ca56049979761fa47855ab92fffaa6c4.
  Production src/corruptions.py UTF8/LF SHA256
  8112b736a6a2853c350fce3e77e8f35b7bffd4d03e14daf6bf97d322cf4b6e50.
  Other frozen hashes/source-region IDs are in existing evidence; verify them.

MODEL-001 BOUNDED TECHNICAL SCOPE

1. Inspect protocol/model spec and config checks. Implement src/models.py
   CompactCNN(num_classes=10,dropout=.2), forward N×3×32×32→N×10 raw logits,
   initialize_weights (or documented internal init), architecture ID compact_cnn_v1.
   No softmax before CrossEntropyLoss; no dataset/policy/trainer imports.
2. Exact architecture: 2×Conv3x3+BN+ReLU at32channels→MaxPool2;
   2×Conv+BN+ReLU at64→MaxPool2; 2×Conv+BN+ReLU at128→MaxPool2;
   AdaptiveAvgPool2d(1)→flatten→Dropout.2→Linear128→10 biasTrue.
   Conv stride1,padding1,biasFalse; BN eps1e-5,momentum.1,affineTrue;
   total288746 trainable params. Stage shapes32→16→8→4; do not change capacity.
3. Init: Conv Kaiming normal fan_out/relu, BN weight1/bias0, Linear normal std.01/
   bias0. Define/document init timing and caller-owned model RNG before checks;
   no hidden reseeding, seed_everything framework or test-data decisions.
4. Meaningful independent synthetic TEST-M01..03: literal layer/parameter/shape
   oracles for batches1/4/128, raw logits, finite CE/backward/gradients (no optimizer
   steps), BN buffers and dropout train/eval behavior, deterministic eval,
   repeated caller seed gives equal full initial states including buffers;
   differing keys and paired clean/robust constructor inputs. Strict state_dict
   round trip and incompatible classes/architecture errors as scoped model QA,
   no new general checkpoint/trainer framework. Avoid flaky distribution assertions.
5. CPU reference required; CUDA only explicit actual opt-in, fail if opted in and
   unavailable; no CPU/CUDA bitwise claim. No model quality/accuracy promises.
6. Compact evidence/relevant docs/status/ai_usage/next bounded prompt. Technical
   PASS separate from full DoD, human understanding/review and formal approvals.
   Preserve ROBUST-001 full acceptance limitation: model init/budget/full scientific
   gate were not established by policy checks and must not be backdated.

НЕБОЛЬШИЕ ЗАВЕРШЁННЫЕ БЛОКИ И НЕМЕДЛЕННАЯ ПУБЛИКАЦИЯ

Работай самостоятельно до проверяемого технического результата MODEL-001.
Не останавливайся после первого промежуточного commit или только плана.
Перед edits кратко объяви текущий scope и ближайший небольшой блок.
Ориентир для блоков, с адаптацией к фактическому размеру кода:

1. Законченный CompactCNN API: architecture, forward, explicit initialization
   и непосредственно связанные shape/parameter/init tests в одном блоке.
2. Train/eval modes, BN/dropout, finite backward и strict state_dict round trip:
   meaningful tests вместе с необходимыми implementation fixes. Если реализация
   уже достаточна, допустим содержательный test-only блок без косметических edits.
3. Actual CPU и explicit opt-in CUDA checks, compact evidence и связанные tests.
   Все тяжёлые проверки последовательно; без optimizer updates/training.
4. Итоговые docs/status/report/ai_usage и следующий bounded prompt, с явным
   разделением technical PASS, full DoD и открытых human/formal gates.

Implementation и непосредственно связанные tests всегда коммить вместе.
Большой блок раздели на меньшие законченные рабочие части. Не делай commits с
заглушками, известными failing tests или искусственным косметическим дроблением.
Не накапливай всю реализацию/tests/docs до одного финального commit.

После КАЖДОГО завершённого блока останови editing:

- Проверь полный diff и relevant tests, затем git diff --check.
- Stage только явные task paths; не используй git add . или git add -A.
- Проверь точный git diff --cached --name-only и полный git diff --cached.
- Выполни git diff --check и git diff --cached --check.
- Проверь, что current branch ровно main, и сделай отдельный рабочий commit.
- СРАЗУ выполни обычный git push origin main; не откладывай push до конца этапа.
- Проверь успешный push, clean worktree, пустой staged list и main=origin/main.
- Сообщи checks, commit hash и результат push.

Только после успешной публикации переходи к editing следующего блока.
При failed push сначала разбери причину и прочитай remote history, до новых edits.
Без force, amend/reset, automatic merge и автоматического затирания conflicts.
Разрешение на commits/push действует на весь этап; повторно его не запрашивай.

Never commit raw/archives/env/caches/credentials/absolute machine paths/diagnostic
partials. End with fresh fetch, clean status, empty staged list, exact main=
origin/main; Russian PASS/FAIL/SKIP, actual evidence/hashes, each commit/push,
open limits and next concrete task. Human review/formal approvals remain human.
~~~
