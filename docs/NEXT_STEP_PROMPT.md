# Промпт следующего этапа — ROBUST-001 prerequisite review и independent technical policy

Обновлён18.11.2026 после CORR-003 independent technical PASS/full DoD PARTIAL.
Published main:3c7a46e pixels/tests, b57a3c7 renderer/CLI/tests,
21ede65 actual PNG/PDF/source manifest/tests/evidence. Каждый блок сразу ordinary
push origin main PASS. Итоговые docs/status/ai_usage/handoff — отдельный commit;
его hash и final fresh-fetch clean sync в Git/chat. Сохранение этого prompt
не запускает ROBUST-001. CORR-003 Person5/team visual review PENDING.

~~~text
Продолжи проект CSCI494 — Robustness to Noise.
Работай, коммить и пушь только в existing exact branch main. Не создавай codex/*,
feature branches или worktrees. Разрешаю самостоятельные commits и ordinary
push origin main после каждого проверенного логического блока, без повторного
согласования. Не используй force, amend/reset или автоматическое затирание
remote divergence/conflicts. При failed push сначала прочитай причину/history.

Следующий bounded technical candidate — ROBUST-001: сначала current prerequisite
review, затем production CorruptionPolicy и actual train-only integration через
existing explicit policy hook. Не объявляй formal prerequisites/reviews закрытыми
по агентским tests. CORR-001/002/003 и DATA-003 independent technical PASS/full DoD
PARTIAL; независимая техническая подготовка допустима, full dependent acceptance
и human/protocol/course/source-policy approvals остаются отдельными gates.
Не запускай обучение, модели/evaluator/metrics/EDA/новые scientific figures,
core runs или broad general REPRO-001. Не тюнь policy/severity по final test.

СНАЧАЛА ПРОВЕРЬ АКТУАЛЬНЫЙ РЕПОЗИТОРИЙ

Прочитай AGENTS.md при наличии, README, docs/MASTER_PROJECT_PLAN.md,
START_HERE.md, CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md,
EXPERIMENT_PROTOCOL.md, IMPLEMENTATION_ROADMAP.md (ROBUST-001, CORR-001..003,
DATA-003/004, REPRO-001), REPRODUCIBILITY.md, TESTING_PLAN.md, SETUP_DECISIONS.md,
report/reproduction.md, report/corr001_evidence.json, corr002_evidence.json,
corr003_evidence.json, report/ai_usage.md, report/course_materials/README.md,
оба YAML, split_manifest.json и normalization.json. Проверь current source:
CorruptionPolicy planned signature, dispatcher/explicit generators,
IndexedCIFAR10.__getitem__ и make_loaders explicit injection gate.
Branch/status/staged/log, fresh fetch, origin/main и remote history обязательны.
Graphify historical pre-DATA-002 snapshot; current source governs, no rebuild.

Existing .venv, без reinstall/update/global/backend changes. Перед edits:
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m pytest tests -q --basetemp=.venv/pytest-robust001-baseline
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-split
.venv/Scripts/python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.make_plots --corruption-grid --verify-only

Сохрани raw/split/stats/configs/lock, оба CORR reference и CORR-003 exports/manifest/
evidence bytes/hashes/mtimes. Missing/corrupt actual inputs — FAIL, не synthetic
PASS/skip/download/repair. DATA-002 region SHA4ec051baa81ab0c70bfd2ca41f7a6750ca3b13f61930cd832b2a84dbd9244f96
и DATA-003 normalization region SHA1cbf01e93d4b1a838daf222c7e1672316117ad65ec1559c4e3075151dae8327c
не менять. Не менять scientific protocol/maps/configs/dependency edges.

СОХРАНЁННЫЙ CORR/DATA CONTRACT

- CPU/CUDA dense finite float32 RGB CHW raw[0,1], positive H/W, output new storage,
  same shape/device/dtype, no mutation. Normalize AFTER corruption.
- apply_corruption: clean только Python int0, families только1..5; noise explicit
  same-device Generator, deterministic families do not consume supplied generator.
  Invalid types/values fail before draws. Exact frozen parameters:
  noise sigma.02/.05/.08/.12/.18; blur sigma.3/.7/1.1/1.5/2 with kernels3/7/9/11/13,
  normalized float64 weights castfloat32, reflect grouped separable convolution;
  brightness factors.9/.8/.7/.55/.4, clip(factor*x,0,1).
- keyed_test_generator/epsilon stay original-test namespace/seed100494 only.
  stable_seed encoding/goldens, geometry/order and train_corruption keys unchanged.
- Existing train hook: raw -> keyed geometry -> float32/255 -> policy(image,
  generator=local stable_seed(run_seed,epoch,original_id,'train_corruption'))
  -> Normalize. Policy returns (pixel Tensor, metadata dict); mutation/range/device
  guard exists. Val/test policy rejected. make_loaders augmentation.enabled must
  match explicitly injected callable; preserve this boundary, no implicit policy.
- CORR-003 minimum train IDs/class29/4/6/9/3/27/0/7/8/1, raw10x16 grid;
  audit-only seed494 namespace cifar10-python/original-train/corr003-audit, not
  production train RNG. No geometry/Normalize/auto-contrast in the visual grid.
- Default18.11:815PASS/0FAIL/16opt-in SKIP; separately2actual grid PASS/0SKIP.
  Existing14DATA/corruption/CUDA opt-ins not rerun in CORR-003; dates in evidence.
  Required baseline raw integrity CLI decodes test for integrity only; grid never does.
- Frozen split NPZ1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284;
  manifest8742e5d48e4001e7f6a61f924bc524ab33f2276a256bc716407a98fc9bc86a21;
  stats FILE332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688,
  semantic4450836de3384df4286535985ed0b56b9a411f3033a60ecf57bbfe5edcb01877.
- CORR-001 NPZ97580bytes40179be75281bdd06dadd266f565e70ad0f7779e81e31fdc6bd1818b45084b33;
  CORR-002 NPZ187438bytesb4396faba198a1497758cd3478983b96a85bd272972cb6266e14d497437d1f03.
- CORR-003 PNG776955bytes9a056d31a97522ae5089fa1999a6b20f00df4b97a98d041802f3f0b493f776f3;
  PDF738357bytes0a3fa45fcc4ded82332e0a0aeb04cd75fc528744c6c521005f4750b20962e2fa;
  source JSON7622bytesf5f09446078fc53d26f8d0e5df4d057c62042ff0540de409d363f26ae2ae5854.
  Strict audit verification includes whole LF-normalized implementation hashes.
  Editing src/corruptions.py or src/data.py will intentionally cause current-source
  mismatch. BEFORE edits plan bounded read-only replay at recorded CORR-003 source
  revision21ede65 (e.g. source snapshot in ignored tmp, no branch/worktree), with
  current-source mismatch covered as expected error. Preserve regression meaning;
  do not skip/drop tests, weaken checks or regenerate old artifacts to hide drift.
  Do not expand this necessity into a general provenance/REPRO-001 framework.

ROBUST-001 INDEPENDENT TECHNICAL SCOPE

1. Inspect existing planned constructor/API and config checks. Implement one
   per-image event: identity.50, noise.17, blur.17, brightness.16; intervals
   [0,.50),[.50,.67),[.67,.84),[.84,1). Corrupted severities uniform1..3 only,
   identity0. No compositions, batch quotas, severity4/5 or per-run tuning.
2. Caller owns explicit local generator/key; no global RNG fallback or seed
   invention. Define/document exact event/severity/noise draw ordering before
   actual checks. Use existing dispatcher. Validate args before RNG consumption;
   reject invalid probabilities/severity/pixels/generator clearly. Keep geometry
   and sample-order streams isolated. Metadata event/severity supports audit only.
3. Meaningful offline tests: deterministic interval endpoints and exact oracles,
   100000 keyed selections with frequency error<.01 and conditional severity
   frequency error<.02; global RNG isolation, invalid/no-mutation/device/dtype/
   range/new storage, same key replay and different keys, explicit injection gate,
   paired geometry/order/batch/worker equality across epochs. Numeric fixtures must
   detect wrong event, probability, range, or Normalize order; do not mirror code.
4. Actual opt-in integration on bounded pre-fixed persisted train IDs, run seeds/
   epochs declared before outputs. Read-only verified raw/split/stats; only
   partitions=('train','val') as needed, no official-test decoding. Check actual
   production policy through existing dataset/loaders, independent corruption and
   normalization oracles, labels/IDs, batch/worker0/2 spawn invariance. Missing/
   corrupt actual inputs FAIL, no skip. Actual whole-pipeline training not needed.
   Recheck existing CORR reference identities and source-region hashes. Preserve
   historical CORR-003 verification via explicit recorded-source replay.
5. Compact actual evidence, relevant docs/status/ai_usage and bounded next prompt.
   Technical PASS separate from full DoD. General REPRO-001, PROTO-001/team freeze/
   course/source-policy approvals, CORR-003 Person5/team visual review, independent
   human review and second-machine reproduction remain open. Do not claim training
   readiness or full DATA-003 dependent acceptance solely from agent checks.

SMALL BLOCKS AND PUBLICATION

Choose working logical blocks, implementation and related tests together:
policy/event semantics; loader/invariance integration; actual tests/evidence;
final docs/handoff. Adapt size to current code, no stub or cosmetic-only commits.
After EACH complete block stop editing, inspect full diff and relevant tests,
git diff --check; stage explicit task paths only, never git add . / -A. Check exact
git diff --cached --name-only and full staged diff, then both diff --check and
diff --cached --check. Commit in main and IMMEDIATELY ordinary git push origin
main; verify push success/main=origin/main/clean staged/worktree and report tests,
commit hash and push result. Only then edit next block. Continue autonomously
through bounded technical result, no repeat permission request. Failed push:
inspect cause/history first; safe fast-forward allowed, no automatic overwrite/
merge of divergence/conflicts and no force/amend/reset/history rewrite.

Never commit raw/archives/env/caches/credentials/absolute machine paths/diagnostic
partials. End with fresh fetch, clean status, empty staged list, exact
main=origin/main; Russian PASS/FAIL/SKIP, actual artifacts/evidence/hashes, each
commit/push, open gates/limits and next concrete task.
~~~
