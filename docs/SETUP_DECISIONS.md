# Проверка готовности foundation

Обновление DATA-00331.10.2026: normalization FILE SHA-256 в обоих YAML теперь
`332c2958a59036b68b6db2b71020dcd2aa0ab021f2a91a32cb351bf446095688`.
Actual45k-only stats/loader evidence — [data003_evidence.json](../report/data003_evidence.json).
Operational schema/encoding/worker decisions зафиксированы в architecture/protocol;
scientific constants неизменны. stable_seed/seed_worker — минимальная часть REPRO-001,
general checkpoint/backup/global seed infrastructure не закрыта. DATA-003 full DoD
PARTIAL до CORR-001/002 integration и human review. Ниже historical decisions.

Обновление DATA-00231.10.2026: в обоих YAML `dataset.split_sha256` теперь равен
SHA-256 actual NPZ file bytes
`1dc8e61a237d7e8fd3ac3c2dea4787e6a82f8f235d74993386af9074f6e1d284`.
Semantic train/val/label-order hashes отдельны в split manifest; normalization hash
по-прежнему null. Schema/config CLI проверяет declarations, не artifacts или
training readiness. [Actual evidence](../report/data002_evidence.json).
Ниже сохранены foundation decisions, включая первоначально unresolved hashes.

30.10.2026. Это техническая self-review агента, не review/approval команды.
Источники: [реестр](../report/course_materials/README.md).

## PROTO-001 — PARTIAL

Сверены architecture и protocol: split494/45k/5k/10k, seeds42/123/2026,
40epochs×352steps=14080, same compact_cnn_v1/288746 params, train-only stats,
pixel-space corruption before Normalize, clean-val selection, eval100494 и
clean+3×5 conditions согласованы. Test data не читались.

Уточнения перед config implementation:

1. Architecture разрешала paths/identity exceptions слишком широко. Для пары
   configs разрешены model_name, augmentation.enabled/probabilities и output
   paths. Dataset paths, runtime и остальные scientific settings пары равны:
   иначе можно незаметно сравнить разные split/stats/backend. Machine relocation
   допустима симметрично для обоих configs. Произвольные metadata keys запрещены.
2. Blur table даёт kernels3/7/9/11/13; правило 2*ceil(3*sigma)+1 с ней согласовано.
   Храним правило и sigma table, kernels выводятся в CORR-002; новая таблица не нужна.
3. Planning prose требовала stats hash в config, но schema table не имела hash
   fields, а artifacts ещё нет. Добавляем dataset.split_sha256 и
   normalization_sha256: null в подготовительном YAML, строка64 lowercase hex
   после DATA-002/003. null означает **unresolved**, не verified artifact. Config
   loader не создаёт и не проверяет существование dataset; training gate позже
   обязан проверить artifacts/hashes и отвергнуть unresolved provenance.
4. v1 schema проверяет типы/диапазоны и замороженные scientific constants из
   protocol. Изменение core design требует обновить protocol/schema/tests вместе.

RQs, scope и технические defaults переведены в reviewable configs по запросу
пользователя. Team acceptance, независимые reviewers Persons2/5, protocol freeze
и разрешение source conflicts **PENDING**. Это не разрешение на core runs.

## TEAM-001 — PARTIAL

Есть role/file/review plan в TEAM_TASKS. В candidate S2b перечислены Daniyar
Kshibekov, Alisher Zhumadelov, Alisher Abeuov, Amangeldi Uldakhanov, Gulnur
Meirimkhanova, Darkhan Shagambayev. Это сведения документа, не независимо
подтверждённый roster. Role mapping Person1..6, availability, primary/backup и
согласие reviewers неизвестны. Не присваиваем роли по порядку имён.

## Границы

REPRO-001 (identity, seeds, atomic writes, artifact manifests/backups) остаётся TODO.
На этом этапе из utils реализуются только config functions и validation CLI.
