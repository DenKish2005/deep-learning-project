# AI assistance record

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
