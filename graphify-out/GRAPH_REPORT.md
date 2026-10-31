# Graph Report - DL Robustness to Noise  (2026-10-31)

## Corpus Check
- Corpus is ~42,428 words - fits in a single context window. You may not need a graph.

## Summary
- 609 nodes · 1121 edges · 47 communities (36 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 53 edges (avg confidence: 0.92)
- Token cost: unavailable for host-agent semantic extraction (the collaboration API does not expose usage). Zero placeholders in cost.json are not measured zero-token work. AST extraction uses no LLM tokens.

## Graph Freshness
- Built from commit: `e7f6ed9f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Raw CIFAR Loading
- Planned Experiment Roadmap
- Strict Config Validation
- Research Deliverables
- Acquisition and Integrity
- Course Policy Provenance
- Clean Training Configuration
- Robust Training Configuration
- Corruption QA Plan
- Research Risks
- DATA-001 Readiness
- Future Data Interfaces
- Clean Dataset Configuration
- Robust Dataset Configuration
- Environment Reproducibility
- Team Roles and Review
- Actual Data Evidence
- Official Data Acquisition
- Future Training Interfaces
- Clean Model Configuration
- Robust Model Configuration
- Next Split Implementation
- Clean Corruption Parameters
- Clean Output Paths
- Clean Runtime Controls
- Robust Corruption Parameters
- Robust Output Paths
- Robust Runtime Controls
- Artifact Provenance and Backup
- Exact Dependency Lock
- Planned Metrics QA
- Planned Model Evaluation
- Clean Augmentation Policy
- Clean Evaluation Configuration
- Robust Augmentation Policy
- Robust Evaluation Configuration
- Final Report Placeholder
- Historical Scaffold Tree

## God Nodes (most connected - your core abstractions)
1. `Implementation roadmap and task dependencies` - 53 edges
2. `Testing and scientific QA plan` - 42 edges
3. `Planned final deliverables and acceptance` - 38 edges
4. `Proposed controlled experiment protocol v1` - 23 edges
5. `load_cifar10()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Implemented raw load_cifar10 / inspect_cifar10 contract` --references--> `main()`  [EXTRACTED]
  docs/TECHNICAL_ARCHITECTURE.md → scripts/download_data.py
- `Verified offline raw CIFAR-10 interface` --references--> `main()`  [EXTRACTED]
  README.md → scripts/download_data.py
- `Implemented raw load_cifar10 / inspect_cifar10 contract` --references--> `load_cifar10()`  [EXTRACTED]
  docs/TECHNICAL_ARCHITECTURE.md → src/data.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Proposed paired scientific control invariants** — docs_experiment_protocol_controlled_comparison, docs_experiment_protocol_fixed_split, docs_experiment_protocol_train_only_normalization, docs_experiment_protocol_training_budget, docs_experiment_protocol_clean_validation_selection [EXTRACTED 1.00]
- **Working DATA-001 integrity and offline surface** — readme_raw_cifar10_interface, readme_official_integrity_checks, readme_integrity_manifest, readme_actual_integrity_opt_in, docs_current_status_data001_technical_acceptance [EXTRACTED 1.00]
- **Planned RQ evidence artifacts** — docs_final_deliverables_final_report, docs_project_overview_rq1, docs_project_overview_rq2, docs_project_overview_rq3, docs_project_overview_rq4, docs_final_deliverables_tab_01, docs_final_deliverables_tab_02, docs_final_deliverables_tab_03, docs_final_deliverables_tab_04, docs_final_deliverables_tab_05, docs_final_deliverables_tab_06 [EXTRACTED 1.00]
- **Verified raw-data acquisition flow** — docs_reproducibility_data_001_working_commands, docs_reproducibility_official_cifar_source, docs_reproducibility_archive_identity, docs_reproducibility_integrity_manifest_v1, docs_reproducibility_artifact_preservation [EXTRACTED 1.00]
- **Controlled-comparison scientific QA gate** — docs_testing_plan_scientific_qa_gates, docs_testing_plan_test_s01, docs_testing_plan_test_s02, docs_testing_plan_test_s03 [EXTRACTED 1.00]
- **AI assistance, source-policy conflict and human review boundary** — report_ai_usage_authorization_not_course_permission, report_ai_usage_ai_human_review_pending, report_course_materials_readme_ai_policy_conflict, report_course_materials_readme_s1a, report_course_materials_readme_s1b [EXTRACTED 1.00]

## Communities (47 total, 11 thin omitted)

### Community 0 - "Raw CIFAR Loading"
Cohesion: 0.055077452667814115
Nodes (52): main(), _allowed_url(), _check_torchvision_contract(), _download_archive(), ensure_cifar10(), _error(), _extract_verified_archive(), _hash() (+44 more)

### Community 1 - "Planned Experiment Roadmap"
Cohesion: 0.07614035087719298
Nodes (59): Proposed controlled experiment protocol v1, Planned ten-chapter final report and verified PDF, ANALYSIS-001: Failure analysis и RQ evidence map, AUDIT-001: Аудит и planning package, BENCH-001: Optional CIFAR-10-C evaluation, CLEAN-001: Clean CNN seed 42 full training, CLEAN-002: Clean CNN seed 123 full training, CLEAN-003: Clean CNN seed 2026 full training (+51 more)

### Community 2 - "Strict Config Validation"
Cohesion: 0.09224318658280922
Nodes (37): Strict controlled config validation, _boolean(), _check_protocol(), _choice(), check(), ConfigError, _construct_mapping(), _fail() (+29 more)

### Community 3 - "Research Deliverables"
Cohesion: 0.048625792811839326
Nodes (43): AC01: Sources, AC02: Data, AC03: Corruptions, AC04: Model, AC05: Control, AC06: Runs, AC07: Models, AC08: Evaluations (+35 more)

### Community 4 - "Acquisition and Integrity"
Cohesion: 0.08858858858858859
Nodes (11): check_backend(), main(), _existing_manifest(), _validate_record_metadata(), _write_manifest(), DataIntegrityError, offline(), _snapshot() (+3 more)

### Community 5 - "Course Policy Provenance"
Cohesion: 0.07207207207207207
Nodes (37): Core plan: six runs and 96 evaluations, not measured results, Course compliance and AI permission are not verified, Requirements and source traceability, Optional CIFAR-10-C, ECE and additional CI/statistical tests, Source hierarchy S1–S5 and requirement traceability, Course/AI policy conflict risk, Deadline/timezone/format risk, Controlled pair exceptions limited to regime/policy/output paths (+29 more)

### Community 6 - "Clean Training Configuration"
Cohesion: 0.1
Nodes (20): training, amp, batch_size, betas, crop_padding, crop_padding_mode, crop_size, early_stopping (+12 more)

### Community 7 - "Robust Training Configuration"
Cohesion: 0.1
Nodes (20): training, amp, batch_size, betas, crop_padding, crop_padding_mode, crop_size, early_stopping (+12 more)

### Community 8 - "Corruption QA Plan"
Cohesion: 0.14210526315789473
Nodes (20): Data/test leakage risk, Planned pixel-space noise, blur, darkening and per-image policy, Testing and scientific QA plan, Offline synthetic test fixtures with explicit oracles, TEST-C01: test_contract_all_conditions, TEST-C02: test_parameter_mapping_and_errors, TEST-C03: test_noise_fixed_by_original_index, TEST-C04: test_noise_statistics_and_clipping (+12 more)

### Community 9 - "Research Risks"
Cohesion: 0.14285714285714285
Nodes (15): AS01–AS09 provisional source, availability, budget and storage assumptions, Core limitations: 32px CIFAR, three families, one architecture and three seeds, Risk, limitation and assumption register, Publish all paired deltas and preserve negative findings, GPU availability risk, Training schedule/runtime risk, Incorrect corruption implementation risk, Robust underfitting risk (+7 more)

### Community 10 - "DATA-001 Readiness"
Cohesion: 0.2564102564102564
Nodes (13): DATA-001 technical PASS / full DoD PARTIAL, Current DATA-001 status and historical audits, Scientific pipeline remains unimplemented, Historical 5ce6b1a scaffold audit, Preserved DATA-001 authorized implementation scope, Preserved DATA-001 execution prompt, Explicit opt-in actual CIFAR integrity tests, Working project interfaces and verified setup (+5 more)

### Community 11 - "Future Data Interfaces"
Cohesion: 0.21212121212121213
Nodes (12): Technical architecture: implemented subset and planned APIs, End-state tree is a future design, not working inventory, Planned IndexedCIFAR10 with original indices and keyed epoch transforms, Planned make_loaders with paired order, float32 images, int64 labels, Planned low-to-high module dependency direction, Generate CPU corruptions per image and condition, Planned versioned run and condition artifact identity, Planned normalization computed on 45k train only (+4 more)

### Community 12 - "Clean Dataset Configuration"
Cohesion: 0.18181818181818182
Nodes (11): dataset, name, normalization_path, normalization_sha256, root, split_path, split_seed, split_sha256 (+3 more)

### Community 13 - "Robust Dataset Configuration"
Cohesion: 0.18181818181818182
Nodes (11): dataset, name, normalization_path, normalization_sha256, root, split_path, split_seed, split_sha256 (+3 more)

### Community 14 - "Environment Reproducibility"
Cohesion: 0.2
Nodes (11): Deterministic inputs versus numeric portability, Reproducibility guide: working setup and planned pipeline, Future split/training/evaluation CLI interfaces, REPRO-002 independent fresh-environment checkpoint reproduction, Verified Windows/Python 3.12.10 CUDA-wheel environment, Version lock without wheel hashes, Engineering requirements are project-plan requirements, PyTorch/CUDA mismatch risk (+3 more)

### Community 15 - "Team Roles and Review"
Cohesion: 0.2
Nodes (10): Planned 186–261 active person-hours, Six-person role and review plan, User workflow: small checked commits and ordinary push only in main, Person 1: Lead/research/integration, Person 2: Dataset/corruptions, Person 4: Robust/analysis/presentation, Person 5: Evaluation/metrics, All six participants must explain the full scientific pipeline (+2 more)

### Community 16 - "Actual Data Evidence"
Cohesion: 0.3333333333333333
Nodes (9): 180 passing default tests; 3 explicit actual-data skips, Observed archive SHA-256 identity, Verified CIFAR-10: 50k/10k RGB32×32 and 5000/1000 per class, Actual verify/acquire/verify preserved file hashes and mtimes, DATA-001 local technical acceptance PASS; full DoD PARTIAL, Recorded default suite:180PASS/3opt-inSKIP, Actual foundation and DATA-001 verification record, Historical foundation snapshot, not current data status (+1 more)

### Community 17 - "Official Data Acquisition"
Cohesion: 0.25
Nodes (8): RUN_CIFAR_INTEGRATION explicit local-data opt-in, CIFAR Python archive: 170498071 bytes and upstream MD5, Preserve corrupt artifacts and fail with recovery action, DATA-001 working acquire and read-only verification, Official Toronto CIFAR-10 source and torchvision 0.24 references, CIFAR download/integrity risk, 3 actual CIFAR tests pass separately with zero skips, Unverified diagnostic partial archive preserved

### Community 18 - "Future Training Interfaces"
Cohesion: 0.25
Nodes (8): Person 3: Model/training/baseline, Implemented config validation plus planned identity/RNG/checkpoint utilities, Planned shared trainer and clean-validation selection, TEST-T01: test_epoch_logging_and_validation, TEST-T02: test_resume_matches_uninterrupted_cpu, TEST-T03: test_training_orchestration, TEST-U02: test_stable_seeds_and_identity_hash, TEST-U03: test_atomic_checkpoint_roundtrip

### Community 19 - "Clean Model Configuration"
Cohesion: 0.2857142857142857
Nodes (6): architecture, dropout, model_name, num_classes, schema_version, seeds

### Community 20 - "Robust Model Configuration"
Cohesion: 0.2857142857142857
Nodes (6): architecture, dropout, model_name, num_classes, schema_version, seeds

### Community 21 - "Next Split Implementation"
Cohesion: 0.2857142857142857
Nodes (7): Next implementation: DATA-002 persisted stratified split, Project onboarding and next actions, Pipeline readiness requires tests, overfit64, actual smoke and paired controls, Submission readiness requires report/slides, rehearsals, reproduction and receipt, Planned persisted stratified split API, Scientific gates before full training, final test and release, TEST-D02: test_stratified_persisted_split

### Community 22 - "Clean Corruption Parameters"
Cohesion: 0.3333333333333333
Nodes (6): corruptions, blur_kernel_rule, blur_padding, blur_sigma, brightness_factor, noise_sigma

### Community 23 - "Clean Output Paths"
Cohesion: 0.3333333333333333
Nodes (6): output, checkpoint_root, figures_root, metrics_root, predictions_root, root

### Community 24 - "Clean Runtime Controls"
Cohesion: 0.3333333333333333
Nodes (6): runtime, deterministic, device, num_workers, persistent_workers, pin_memory

### Community 25 - "Robust Corruption Parameters"
Cohesion: 0.3333333333333333
Nodes (6): corruptions, blur_kernel_rule, blur_padding, blur_sigma, brightness_factor, noise_sigma

### Community 26 - "Robust Output Paths"
Cohesion: 0.3333333333333333
Nodes (6): output, checkpoint_root, figures_root, metrics_root, predictions_root, root

### Community 27 - "Robust Runtime Controls"
Cohesion: 0.3333333333333333
Nodes (6): runtime, deterministic, device, num_workers, persistent_workers, pin_memory

### Community 28 - "Artifact Provenance and Backup"
Cohesion: 0.3333333333333333
Nodes (6): Verified external artifact backup and retrieval, Ignored CIFAR integrity manifest schema v1, Dataset-to-figure provenance identity chain, Checkpoint loss risk, Person 6: Environment/reproduction/plots/QA, Observed manifest schema1 SHA-256 identity

### Community 29 - "Exact Dependency Lock"
Cohesion: 0.5333333333333333
Nodes (6): Divergent local environment risk, Verified exact 38-package environment lock, 38 exact installed versions, Windows CUDA 12.6 wheels, pytest 9.2.0 pinned, torch 2.9.0+cu126 pinned, torchvision 0.24.0+cu126 pinned

### Community 30 - "Planned Metrics QA"
Cohesion: 0.3333333333333333
Nodes (6): Plotting/aggregation bug risk, Planned classification/robustness metrics and complete-matrix validation, TEST-K01: test_known_classification_metrics, TEST-K02: test_robustness_units_and_denominators, TEST-K03: test_seed_aggregation_and_pairing, TEST-K04: test_confusion_and_f1_edge_cases

### Community 31 - "Planned Model Evaluation"
Cohesion: 0.3333333333333333
Nodes (6): Planned CompactCNN: 288746 parameters, Planned frozen-checkpoint evaluation of common 10k test inputs, TEST-E01: test_checkpoint_to_evaluation, TEST-E02: test_complete_matrix, TEST-E03: test_eval_idempotent_and_order_independent, TEST-M01: test_model_shapes_and_parameter_count

### Community 32 - "Clean Augmentation Policy"
Cohesion: 0.5
Nodes (4): augmentation, enabled, probabilities, train_severities

### Community 33 - "Clean Evaluation Configuration"
Cohesion: 0.5
Nodes (4): evaluation, batch_size, seed, severities

### Community 34 - "Robust Augmentation Policy"
Cohesion: 0.5
Nodes (4): augmentation, enabled, probabilities, train_severities

### Community 35 - "Robust Evaluation Configuration"
Cohesion: 0.5
Nodes (4): evaluation, batch_size, seed, severities

## Ambiguous Edges - Review These
- `__init__()` → `__init__()`  [AMBIGUOUS]
  tests/test_data_pipeline.py · relation: calls

## Knowledge Gaps
- **76 isolated node(s):** `Original repository scaffold tree`, `H1: Accuracy generally decreases with severity`, `H2: Noise and blur may hurt more than moderate darkening`, `H3: Augmentation may improve mCA`, `H4: Robustness gains may reduce clean accuracy` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 239 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Implemented raw load_cifar10 / inspect_cifar10 contract` connect `Team Roles and Review` to `Raw CIFAR Loading`, `Official Data Acquisition`, `Future Data Interfaces`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **What connects `Original repository scaffold tree`, `H1: Accuracy generally decreases with severity`, `H2: Noise and blur may hurt more than moderate darkening` to the rest of the system?**
  _76 weakly-connected nodes found - possible documentation gaps or missing edges._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `__init__()` and `__init__()`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
## Graph Health and Audit Limits
- Raw extraction diagnostic: 88 dangling-endpoint references, 4 self-loop edges, 12 directed/undirected same-endpoint collapses. These are surfaced, not hidden.
- Build resolves the raw references as 27 explicitly external dependency nodes. Final JSON: 0 dangling endpoints; 1133 relationship records preserved. Community analysis projects them to 1121 simple undirected edges; the raw JSON is a multigraph.
- Three self-loops are recursive validation calls. The fourth is a static AST ambiguity at tests/test_data_pipeline.py:394: super().__init__ is not proof of recursive Response initialization. That candidate is retained as AMBIGUOUS (0.2), not an established recursive call.
- AST records definitions and statically resolved relationships; it is not a runtime call trace. EXTRACTED doc edges mean the document states the relationship, not that a planned task was executed.
- All 25 semantic source files were reported read to their last line and coverage matched file lengths. 20 code/config files used deterministic AST extraction. No dataset artifacts, credentials or environment packages were indexed.
- graphify-out/ was written into the working tree. Only shareable HTML/JSON/report/cost outputs and its local .gitignore are intended for Git; interpreter paths, caches and temporary extraction files are ignored. See https://docs.graphify.com/guides/team-workflows.

## Extraction Accounting
- Raw relationship records: 1133; exact confidence counts: EXTRACTED=1078, INFERRED=54, AMBIGUOUS=1.
- Manifest: 45 detected files; 25 semantic sources stamped; coverage complete, no partial semantic sources.
- Corpus detection estimated 42,428 words. The separate benchmark uses its own 30,450-word naive-corpus estimate; its 25.0x result is an estimate, not measured billing or answer accuracy.
- Benchmark: approximately 40,600 naive tokens versus 1,623 average graph-query tokens; sample reductions 25.1x / 25.2x / 25.0x / 24.8x.
- The graph was built from source commit e7f6ed9f. A subsequent commit containing only graph outputs does not indicate a change to the analyzed project source.

## Output Validation
- PASS: JSON parse, unique IDs, all exported endpoints exist, all 47 communities labelled, repository-relative source paths, HTML embeds graph and labels, and graphify explain resolves load_cifar10 with source evidence.
- Project source/configs/tests and the locked environment were unchanged by this graph build. Validation covers artifact structure and a CLI navigation smoke; browser rendering was not manually inspected.
