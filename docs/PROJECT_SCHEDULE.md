# График с 30 октября 2026

План основан на **skeleton без реализации**. Предварительный report deadline12.12.2026; exact time/timezone и presentation Week 14 ещё не подтверждены. Proposal deadline20.09.2026 — прошлое событие, не будущая задача. На SETUP-001 до 02.11 получить current LMS/calendar/AI policy; если presentation раньше планируемой rehearsal, сдвинуть presentation gates немедленно.

## Реалистичный основной график

| Даты,2026 | Milestone и concrete deliverables | Dependencies / owners | Active effort (team) | Gate |
|---|---|---|---:|---|
| 30.10–02.11 | Planning14 docs; verified sources/AI/deadlines или pending register; именованные owners; env candidate | AUDIT/SETUP-001/002,PROTO/TEAM;1/6 | 16–22 h | Official unknowns owned; protocol review; hardware/runtime plan |
| 03.11–08.11 | Config/schema,split/stats,3corruptions, CNN, metricfixtures; report Intro/Methods draft | DATA/CORR/MODEL/METRIC-001;1/2/3/5,6env | 42–56 h | Unit tests, data counts288746 params, firstgrid |
| 09.11–15.11 | Shared training/checkpoint/resume, robust hook,eval/plot/orchestrator,64-image overfit, actual smoke | TRAIN,ROBUST-001,EVAL-001,PLOT-001,TEST-001/002;all6 | 42–58 h | Smoke pass, paired invariants,40 epoch budget feasibility |
| 16.11–22.11 | Six frozen core trainings; first16baseline then full96eval; backup/log review | CLEAN/ROBUST/EVAL;3/4/5 with6 | 20–30 h + measured compute | 6 runs,96 conditions, allhashes; target core results by22.11 |
| 23.11–29.11 | Aggregate tables,8 figure types,failure/RQ analysis; report full draft;slidesv1; QA/release candidate | METRIC-002,PLOT/ANALYSIS,REPORT-002,PRESENT-001,RELEASE-001,TEST-003;all6 | 32–45 h | Complete evidence map; **MVP complete by29.11** |
| 30.11–06.12 | Independent reproduction;finalreport review;2rehearsals;presentation when verified | REPRO-002,REPORT-003,PRESENT-002/003;all6 | 26–36 h | Rehearsal1 target 01.12;rehearsal2target04.12, both subject to earlier course date |
| 07.12–09.12 | Submission-ready PDFs, links, artifact retrieval, final rubric checklist | RELEASE-002;1lead,6package, allreview | 8–14 h | Internal submission deadline09.12 |
| 10.12–12.12 | **Buffer3calendar days** for upload/export/administrative failures; submit by verified official cutoff | Course deadline and receipt;1+secondreviewer | reserve, not feature work | Receipt/download verified; no new extensions |

Effort estimates loosely sum186–261active person-hours including reviews/coauthoring; распределение в TEAM_TASKS. Milestone estimates не runtime guarantee и частично перекрываются. По одному доступному GPU sixruns могут идти последовательно; параллельная работа людей не означает simultaneous model trainings.

## Как рассчитать compute budget после pilot

TRAIN-003/ROBUST-002 измеряют median steady-state epoch time после warmup и peak memory. Пусть t_A, t_B — seconds/epoch включая validation; T_eval — measured seconds per10000-image condition включая preprocessing/I/O. Remaining serial estimate:

`T_compute = 3*40*(t_A+t_B) + 96*T_eval + artifact/plot overhead`.

Добавить **30% compute contingency**, учитывать занятость GPU/Colab quotas и interruptions. При pilot отдельно измерить robust augmentation CPU bottleneck и выбранный batch size; model memory по parameter count нельзя считать GPU peak. Если estimate не помещается 16–22.11, запускать frozen training сразу после smoke, организовать persistent GPU access или уменьшить budget симметрично **до final test**. Если изменить batch size, пересчитать steps/epoch и protocol artifacts, не сохранять ложные 14080 steps.

## Ideal / realistic / minimum

| Вариант | Даты и scope | Условия |
|---|---|---|
| Ideal | Smoke08.11, core 96by15.11,fullreport/slides22.11,reproduction/rehearsals29.11 | Достаточная availability, working GPU, fast integration; после core optional benchmark возможно |
| Realistic | Smoke15.11, core 96by22.11, MVP29.11,finalreport/rehearsals06.12, internal submission09.12 | ~5–8h/person/week, measured compute fits oneweek; buffer preserved |
| Minimum complete scope | Сохранить 6 runs×16 conditions,3families, allrequiredmetrics,8typesfigures,report/slides/repro; убрать BENCH-001,METRIC-003, дополнительные analyses и дизайн-полировку | Core complete не позже 29.11, report draft next48 h, зафиксировать actual limitations |
| Emergency partial delivery | Если 6 runs не укладываются даже после корректного до-test revision: представить реально выполненный subset и explicitly disclose missing work | Это **не full DoD**. Не сочинять seeds/metrics; требования курса и последствия сверить с instructor |

MVP означает полный core scientific product, а не one-seed prototype. Optional работу не начинать, если она отнимает время у report/rehearsals или ставит под риск 04–09.12.

## Critical path и parallel lanes

1. Official constraints+protocol/config→dataset/split/normalization+corruptions+CNN→shared trainer/checkpoints→policy+smoke/scientific gate.
2. Sixtrainingruns→96evaluations→validated aggregation→plots/failure evidence→report results/review.
3. Release candidate→independentreproduction;finalreport+slides→2rehearsals→submission verification.

Person 2data,Person3model/training, Person 4policy,Person5metrics/eval,Person6env/plots,Person1requirements/report работают параллельно. Methods и report skeleton пишутся до результатов. One-GPU compute — потенциальный bottleneck; допускается перенос machines только с environment provenance и pairedcomparison limits.

## Контроль отклонений

- 02.11: если course materials ещё нет, продолжить implementation по documented provisional scope, owner1 продолжает retrieval; не утверждать official compliance до сверки.
- 08.11: если API/data/model tests не готовы, отложить notebook polish и optional; перераспределить reviews от 5 к 6 и writing от 6 к1/4.
- 15.11: если smoke не проходит, core training не начинать. Исправить validitybug; report Methods продолжать на available protocol.
- 22.11: если runs прерываются, resume verified epoch/backup; работать с actual status, не считать partial run complete.
- 29.11: если robustness gain≤0, **график не продлевать ради положительного результата**. Публиковать отрицательный result с diagnostics.
- Как только известна presentation date P: slidesdraft≤P−7 days;rehearsal1≤P−5 days;finaldeck≤P−3 days;rehearsal2≤P−2 days;offlinepackage≤P−1day. Если P раньше 22.11, приоритет datasets/methods+честные interimresults, согласовать finalresult deadline по actual course instructions.

Week 14 не переведена в calendar date: semester start, teaching breaks и точное расписание неизвестны. Даже дата 12.12 остаётся provisional until S1. Не считать machine timezone источником LMS deadline.
