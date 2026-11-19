# Промпт следующего этапа — METRIC-001 prerequisite review и bounded metrics API

METRIC-001 is ready for review. See the [current status](CURRENT_STATUS.md) and
[verification record](../report/reproduction.md). The original implementation
prompt is retained below for reference.

Обновлён18.11.2026 после MODEL-001 synthetic technical PASS/full DoD PARTIAL.
Published main:489eb04 model API/tests,11cffd5 modes/strict identity,6dad624 devices/
evidence; каждый immediately ordinary push origin main PASS, fresh fetch clean0/0.
Final docs/status/AI disclosure/handoff публикуются отдельно; final hash в Git/chat.
Сохранение этого prompt не запускает METRIC-001 и не разрешает training.

~~~text
Продолжи проект CSCI494 — Robustness to Noise.
Работай/коммить/пушь только в существующую ветку с точным именем main.
Не создавай codex/*, другие branches или worktrees. Historical branch/PR fields
не переопределяют это правило. Разрешаю самостоятельные commits и обычный
push origin main после каждого проверенного небольшого блока без повторного
согласования. Без force, amend/reset, history rewrite, automatic merge/conflict
overwrite. Failed push сначала расследуй по remote history; safe fast-forward
допустим только без затирания изменений.

Следующий independent bounded candidate — METRIC-001: prerequisites, classification
и per-run robustness metrics на hand/synthetic fixtures. PROTO-001/SETUP-002 formal
human/course/team gates pending; bounded technical preparation разрешена, full
DoD/dependent acceptance остаётся отдельным gate. Не запускай training/optimizer
updates, overfit/pilot/core, evaluator, actual model inference, EDA/новые scientific
figures, METRIC-002 seed aggregation, optional ECE/CI или broad general REPRO-001.

СНАЧАЛА REPO/DOCS/PREREQUISITES
Прочитай AGENTS.md при наличии, README, docs/MASTER_PROJECT_PLAN.md, START_HERE.md,
CURRENT_STATUS.md, REQUIREMENTS.md, TECHNICAL_ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md,
IMPLEMENTATION_ROADMAP.md (METRIC-001/002, PROTO-001, SETUP-002/003, MODEL-001,
ROBUST-001, EVAL-001, REPRO-001), TESTING_PLAN.md, REPRODUCIBILITY.md,
SETUP_DECISIONS.md, report/reproduction.md, model001_evidence.json,
robust001_evidence.json, corr001/002/003_evidence.json, ai_usage.md,
course_materials/README.md, оба YAML и split/stats manifests. Current source governs;
Graphify — historical pre-DATA-002 snapshot, no rebuild. src/metrics.py placeholder.
Проверь exact main/status/staged/log, fresh fetch, exact main=origin/main/history.
Existing .venv без install/update/global/backend changes. Heavy checks sequential.

Перед edits:
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m pytest tests -q --basetemp=.venv/pytest-metric001-baseline
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.download_data --root data/raw --verify-split
.venv/Scripts/python.exe -m scripts.prepare_normalization --root data/raw --verify-only
.venv/Scripts/python.exe -m scripts.verify_frozen_corr003
.venv/Scripts/python.exe -m src.utils --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --controlled-pair
Actual missing/corrupt inputs — FAIL; без skip/download/repair/synthetic substitution.
Зафиксируй hashes/bytes/mtimes protected raw/split/stats/configs/lock/protocol,
CORR references/grid/source, все previous evidence и data/utils/corruptions/models
source. Не менять scientific settings, dependency edges, DATA-002/003 frozen
regions, stable_seed, model/policy/old functions или прежние evidence bytes.

СОХРАНЁННЫЙ CONTRACT
- Default950PASS/0FAIL/20opt-in SKIP. MODEL-001 separate executed CPU/CUDA2PASS/0SKIP
  на synthetic linspace; previous19actual opt-ins не повторялись. Evidence17819bytes
  SHA256f658440c8c34425cae6c9e9831dd822c345e7f04a5e1bae39c1e3471677ee362.
- CompactCNN6Conv+BN+ReLU/3pool/GAP/Dropout.2/Linear128->10,288746params/raw logits.
  Caller seed→CPU layer constructor draws→one protocol init→device, no reseed.
  state_dict includes `_extra_state` architecture/num_classes/dropout; strict=True,
  fresh instance, weights_only=True/explicit map_location; no general checkpoint.
  src/models.py UTF8/LF SHA25637eb278933b20e538ac19041e2d2c895c119392354356a5d80bbbec59c57c787.
  Paired init constructor boundary PASS; optimizer budget/full TEST-S01 remain open.
  Backend determinism flags were not changed; no training/cross-device claim.
- Frozen policy probabilities[.50,.17,.17,.16]/max3, one event/image, explicit keyed
  caller generator; raw→geometry→float32/255→policy→Normalize. Existing streams,
  train-only injection, val/test rejection unchanged; future training train/val only.
- CORR-003 current-source strict verify intentionally mismatches after policy edits.
  Use authenticated read-only recorded-source replay at21ede65; never regenerate
  frozen exports/fingerprints or weaken verifier. Other frozen identities in evidence.

METRIC-001 BOUNDED TECHNICAL SCOPE
1. Inspect architecture/protocol schemas before choosing API representation; state
   units, accepted label/prediction types and row structure explicitly before tests.
2. Implement src/metrics.py confusion_counts/classification_metrics and per-run
   robustness_metrics as scoped roadmap: rows=true int64 CM; accuracy fraction,
   per-class recall, macroF1 over fixed classes including zero support (F1 zero
   denominator→0). Reject empty, bad length/type/range, nonfinite inputs; no mutation.
3. TEST-K01/K04 hand oracle y=[0,0,1,1], pred=[0,1,1,1]: CM[[1,1],[0,2]], acc.75,
   F1mean(2/3,.8), separate10-class/absent class/all-right/all-wrong fixtures.
   Check counts/support/orientation, invalid input diagnostics and exact units.
4. TEST-K02: drop=100*(clean-corrupted)pp, RR=corrupted/clean (clean0→null+flag),
   mCA mean exactly15corrupted rows, family means5each, gap=100*(clean-mCA)pp.
   Fixture clean.8/corrupted.6→20pp/.75/mCA.6/gap20pp. Clean excluded from mCA;
   negative drops allowed. Reject missing/duplicate/unknown severity/family or
   invalid counts/ranges/nonfinite rows; never nanmean/interpolate missing results.
5. No real evaluation/results or seed aggregation/CLI orchestration framework.
   Synthetic numbers must stay labeled fixtures. Compact evidence/docs/status/
   AI usage/next bounded prompt; technical PASS distinct from full DoD/human review.

SMALL COMPLETE BLOCKS / IMMEDIATE PUBLISH
Before edits announce scope and nearest block. Suggested: classification API plus
related tests; per-run robustness API plus related tests; evidence/docs/handoff.
Implementation and directly related tests commit together; test-only follow-up
allowed when substantive. Do not accumulate all work into one final commit.
After EACH block stop editing; inspect full diff/relevant tests; git diff --check.
Stage explicit task paths only (no git add . or -A); exact cached name list and
full git diff --cached; git diff --check and git diff --cached --check. Confirm
branch exactly main, separate commit, IMMEDIATE ordinary push origin main.
Fresh fetch and verify publication/clean status/empty staged/main=origin/main0/0;
report checks/hash/push before editing next block. Failed push investigate first.
Never commit raw/env/caches/credentials/absolute machine paths/diagnostic partials.
End with Russian PASS/FAIL/SKIP, evidence/hashes/all commits+pushes, open limits,
fresh-fetch clean exact main=origin/main and staged empty. Human understanding,
independent human review and formal approvals remain human gates.
~~~
