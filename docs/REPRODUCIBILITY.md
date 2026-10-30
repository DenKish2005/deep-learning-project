# Воспроизводимость и плановые команды

Все ML CLI ниже — **целевые интерфейсы после реализации соответствующих tasks**, сейчас source/scripts пустые. Нельзя запуск успешного empty script интерпретировать как выполненную задачу. На стадии планирования команды не выполнялись, dependencies не устанавливались.

## Окружение

Reference Python **3.12.x**, зафиксировать patch после проверки (локально обнаружен 3.12.10). PyTorch и torchvision выбирать согласованной парой из официального [compatibility table](https://github.com/pytorch/vision). Предложенный консервативный install candidate: torch 2.9.0 / torchvision 0.24.0; он ещё не проверен на машине команды. Это не утверждение о последней версии. Установленные metadata2.14.0/0.29.0 не доказывают совместимость imports/ops или CUDA.

SETUP-002: создать отдельный venv; проверить candidate CPU import/forward, `pip check`, dataset decoding и tests. Затем зафиксировать exact versions всех восьми runtime packages и pytest, hashes/installation source в lock и environment metadata. При непригодности candidate выбрать другую **документированную пару**, не смешивать torch и vision independently. Не фиксировать выдуманный полный lock до resolver и smoke. [Официальные install combinations](https://docs.pytorch.org/get-started/previous-versions/).

План Windows PowerShell, из repo root; отдельный interpreter без необходимости Activate.ps1:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib tqdm pyyaml pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

First-install commands содержат unpinned non-torch packages только для **разрешения candidate**. Core runs разрешены после проверки и exact lock. В будущем воспроизведение: установить нужный torch/vision backend по записанному install recipe, затем `python -m pip install -r requirements-lock.txt`; lock различает CPU/CUDA wheel suffixes. Не коммитить machine-specific absolute URLs/tokens из freeze. requirements.txt обновить согласованно; pytest можно оставить dev section комментариями в простом файле.

GPU recipe не выбирается до проверки `nvidia-smi`, GPU/VRAM, driver, torch.version.cuda/cuDNN и подходящего official wheel. System CUDA toolkit version сама по себе не доказывает wheel compatibility. Person 6 сохраняет точную pip command/index URL в environment.json и выполняет CUDA forward/backward smoke. Не запускать два concurrent jobs на одном GPU без memory profile.

Colab: fresh runtime, version check до imports, согласованная install pair, restart runtime если packages заменены, code commit/copy hash и mounted persistent artifact storage. Пути через pathlib, workers 0 reference. Не полагаться на ephemeral `/content` для единственной копии checkpoints. Результаты разных platforms отмечать, paired models по возможности обучать на одном backend.

## Determinism

Перед CUDA init в PowerShell: `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'`; затем `random.seed`, NumPy seed, torch.manual_seed и cuda.manual_seed_all при наличии. `torch.use_deterministic_algorithms(True)`, cudnn.benchmark=False, cudnn.deterministic=True; отключить TF32 для reference comparison. Unsupported nondeterministic operation должна fail с diagnostic, не silent warn-only. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

PyTorch не обещает одинаковые results между releases/platforms и CPU/GPU; deterministic mode может замедлить вычисления. Reference claims ограничивать locked software/backend. CPU pixel corruption и stable per-image streams обеспечивают общие inputs; model outputs across hardware сравнивать с заранее записанными tolerances, не заявлять bitwise cross-platform.

Worker initialization и Windows spawn требуют top-level worker functions и main guard; seed иных библиотек берётся из torch.initial_seed. Это учитывается в data interfaces. [DataLoader docs](https://docs.pytorch.org/docs/main/data.html). Exact epoch resume проще при workers 0; stateless transforms исключают скрытый worker transform state.

Не использовать global RNG для robust policy: отдельные keyed generators сохраняют одинаковые init, crop/flip и sample order между моделями. Полные seed namespaces: [EXPERIMENT_PROTOCOL §4](EXPERIMENT_PROTOCOL.md).

## Плановые команды pipeline

В примерах `python` обозначает interpreter проверенного env; на Windows можно заменить на `.\.venv\Scripts\python.exe`. Команды из root. Device cpu — рабочий fallback после реализации; cuda допустим только после validation. Parameters subset/profile относятся только к smoke/debug и запрещены для core profile.

```powershell
# DATA-001..003: download + checksum + saved split494 + train-only stats
python -m scripts.download_data --root data/raw --prepare-split --split-seed 494
python -m scripts.download_data --root data/raw --verify-only

# TEST-001: offline synthetic unit/integration tests после написания
python -m pytest tests -q

# TRAIN-003: отдельный debug overfit, test set не читается
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile debug --overfit-samples 64 --max-steps 300 --device cpu

# TEST-001: actual CIFAR smoke, no core outputs
$env:RUN_CIFAR_INTEGRATION='1'
python -m pytest tests/test_smoke.py -q

# TRAIN-004: проверить именно 6 core run specs до вычислений
python -m scripts.run_experiments --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --seeds 42 123 2026 --profile core --dry-run

# CLEAN-001 / ROBUST-002: пример пары; orchestration ниже охватывает все seeds
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile core --device cpu
python -m src.train --config configs/robust_cnn.yaml --seed 42 --profile core --device cpu

# Полные 6 training runs, completed verified runs пропускаются
python -m scripts.run_experiments --configs configs/clean_cnn.yaml configs/robust_cnn.yaml --seeds 42 123 2026 --profile core --device cpu --resume-incomplete

# EVAL-002..004: matrix planning и затем 96 evaluations
python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --require-complete --dry-run
python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --device cpu --require-complete

# METRIC-002 и PLOT-001..003
python -m src.metrics --input outputs/metrics --output outputs/metrics/aggregate
python -m scripts.make_plots --metrics outputs/metrics/aggregate --runs outputs/runs --output outputs/figures
```

`--profile debug` explicitly выключает geometry/corruption/dropout только для 64-image diagnostic; manifest содержит overrides, mode и max_steps; core validator rejects debug. `test_smoke` создаёт temporary outputs, profile smoke,2models×1 seed×16 conditions на 100 test images,1 epoch; full validator считает только core.

Single-condition evaluation после реализации:

```powershell
python -m src.evaluate --checkpoint outputs/checkpoints/RUN_ID/best.pt --corruption gaussian_noise --severity 3 --device cpu
```

`RUN_ID` — явный placeholder; взять точное имя из run manifest. Resume:

```powershell
python -m src.train --config configs/clean_cnn.yaml --seed 42 --profile core --resume outputs/checkpoints/RUN_ID/last.pt --device cpu
```

Аргументы всех CLI: `--help` со defaults, invalid argument nonzero exit, JSON status success/failed/interrupted. `run_experiments --resume-incomplete` проверяет hash/config/code identity и не запускает другой config поверх старого directory. `--require-complete` в eval требует все шесть manifests и core profile; pilot16 может запускать src.evaluate отдельно, без core aggregate.

Optional benchmark после BENCH-001: `python -m scripts.run_all_evaluations --checkpoints-root outputs/checkpoints --dataset cifar10c --data-root data/raw/CIFAR-10-C --device cpu --require-complete`; loader берет gaussian_noise/defocus_blur/brightness1..5, отдельные directories и outputs90. Этот flag отсутствует до extension task; он не меняет core 96.

## Provenance и verification

Per epoch, per run и per condition schemas canonical в EXPERIMENT_PROTOCOL §6. Обязательные identity links: data checksum → split hash → stats hash → resolved config hash → code identity → checkpoint hash → condition hash → predictions hash → metric row → figure/table manifest. Git SHA без dirty status недостаточен; core release должен ссылаться на stable code revision и record any local patch hash.

Run outputs ignored в Git должны иметь artifact_manifest и backup, доступный всем six teammates. После каждого complete epoch last atomic; после run backup best/last/history/config/env, проверить hashes после copy. Каждую ночь активных runs проверять свободное место и archive status. Dataset можно повторно скачать по checksum; unique checkpoints нельзя заменить обещанием retrain.

REPRO-002: другой участник в fresh env делает install, unit tests и actual smoke по README, затем загружает один frozen best checkpoint и воспроизводит clean + noise 5 predictions. Для одинакового locked CPU backend ожидается exact predictions; при GPU численные logits tolerance объявить (например atol1e-5, rtol1e-5) и отдельно compare predicted labels/metrics. Если tolerance не соблюдена, исследовать cause, не увеличивать допуск без evidence. Все результаты и команды в report/reproduction.md; полный retrain second machine не обязателен, но claims о retraining reproducibility без него ограничены.

## Плановый artifact audit

| Check | Pass |
|---|---|
| Data | Original checksums, one split, train-only stats и balanced counts |
| Training | 6 core manifests,40 epochs each,14080 steps each, stable code/config |
| Checkpoints | All best/last load, architecture288746 params, SHA verified |
| Evaluation | 96 unique keys,10000 predictions each, common input hashes |
| Aggregation | No missing seeds, sampleSD, paired deltas, pp/fraction correct |
| Figures | Regenerated from frozen tables; manifest lists hashes |
| Recovery | CPU reference epoch resume equality + backup retrieval test |
| Reproduction | Independent fresh-env smoke and frozen checkpoint evaluation |

Все pass записи появятся только после фактического выполнения, не при наличии command snippets.
