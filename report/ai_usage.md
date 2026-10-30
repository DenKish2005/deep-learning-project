# AI assistance record

## DATA-001 — 30–31.10.2026

По прямому запросу пользователя Codex реализовал raw offline CIFAR-10 loader,
official-HTTPS acquisition с проверкой archive/seven-file checksums до unpickle,
оба partition contracts, CLI acquire/verify-only, ignored schema-v1 manifest,
64 synthetic offline cases и3 separate opt-in actual tests. Сверены official
dataset page, torchvision v0.24.0 source и установленная0.24.0+cu126. Данные реально
получены и инспектированы только для integrity; test images/labels не применялись
к scientific decisions. Configs, dependencies/backend и training scope не менялись.

Agent checks, команды, actual results и ограничения — [reproduction.md](reproduction.md).
Код/tests опубликованы небольшими проверенными commits5fd4088/f159fd7/50a9fc4
на main; обычный push после каждого. Отдельный documentation/evidence commit
фиксирует текущий этап. Review выполнял агент; independent human reviewer,
course-policy clarification и подтверждение понимания кода командой остаются
PENDING. Полный DATA-001 DoD — PARTIAL; технические PASS не являются team sign-off.

## Исторический foundation record

30.10.2026: Codex использован по прямому запросу пользователя для source audit,
проверки protocol/architecture, isolated environment, configs/schema validation,
tests и обновления инструкций. Изменения доступны как локальные Git commits.
Data pipeline, CNN, corruptions, training и evaluation на этом этапе не реализуются.

Затронутые файлы: configs/clean_cnn.yaml, configs/robust_cnn.yaml, src/utils.py,
tests/test_utils.py; scripts/check_environment.py, requirements*.txt, .gitignore;
README, planning docs и source/reproduction records в report/. Exact diff и авторство
помощи сохраняются в commits. Local machine verification:116 config tests,
pip check, dependency imports, CPU/CUDA forward/backward, torchvision NMS/PNG
round-trip — PASS. Эти проверки выполнял агент; human understanding/review не
подтверждены ими автоматически.

## Policy status: CONFLICT_PENDING

[Реестр](course_materials/README.md): S1a p17–20 допускает debugging, explanations,
writing improvements и small code suggestions, требует ответственности за код
и понимания pipeline. S1b p7 запрещает code generation tools на graded assignments.
Applicability/precedence для course project и допустимость текущего объёма помощи
не установлены. Авторизация пользователя не является course permission.

Перед сдачей команда должна уточнить policy у course staff, проверить и понимать
каждый изменённый файл, определить требуемый disclosure. Human review,
подтверждение policy и disclosure format **PENDING**; AI self-review не заменяет их.
