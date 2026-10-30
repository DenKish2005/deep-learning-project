# Реестр источников CSCI 494

Проверено 30.10.2026 (Asia/Qyzylorda). Область поиска: repository (tracked и
untracked), цепочка родительских AGENTS.md, имена файлов в локальных Desktop,
Downloads и Documents. LMS, почта и submission receipts недоступны в этой проверке.
Абсолютные локальные пути и оригиналы с персональными данными не включены в Git.
Имена ниже позволяют найти файлы в локальных Downloads; SHA-256 идентифицирует
содержимое, но не удостоверяет публикацию преподавателем или факт сдачи.

| ID | Источник / locator | Наблюдаемый статус | Осталось |
|---|---|---|---|
| S1a | `course_project_guidelines.pdf`, Fall Semester 2026, Siamac Fazli; 24 PDF pages / 11 slides | FOUND_LOCAL, содержание прочитано; копии `(1)`/`(2)` byte-identical | Подтвердить актуальность в LMS и отсутствие новых указаний |
| S1b | `CSCI494_CSR.pdf`, 10 pages; p10 Course Origination Date 2026-05-18, Effective from Fall 2018 | FOUND_LOCAL; копии `(1)`/`(2)` byte-identical; semester-specific authority pending | Уточнить применимость CSR и конфликт AI policy |
| S1c | отдельная детальная rubric, актуальный LMS calendar, submission instructions | NOT_FOUND в обследованных материалах; grading criteria есть в S1a p22 | Получить actual rubric/формат, cutoff/timezone, presentation slot/duration/language |
| S2a | `CSCI494_Robustness_to_Noise_Project_Proposal.docx`, date 20 September 2026 | FOUND_LOCAL_CANDIDATE; без списка имён | Submission receipt/подтверждение именно сданной версии |
| S2b | `CSCI494_Robustness_to_Noise_Project_Proposal (1).docx`, та же дата | FOUND_LOCAL_CANDIDATE; содержит шесть имён | Не считать суффикс `(1)` признаком approved/latest; подтвердить версию и роли |
| S0 | `CSCI 494.pdf`, Desktop syllabus collection, Fall 2025 | HISTORICAL_ONLY; не источник правил Fall 2026 | Не применять team-size 3–5 из p29 к текущему проекту без сверки |

SHA-256:

```text
S1a 6d3cdfbac714c749db5a19ce4cb54107614fc4052ecb620df52f8f8c29b7057b
S1b 9508c908ee3b2f506b45bf6ed0375beaad3740c1dfd877b4a10cfa214e9c82da
S2a cbb1697d874d5acc6c8e856df6a99b9370603c31efb507564a8f9d1ad6f71a2a
S2b c7d1e3a3f27f4333b72b24ac375fef65c6ead8d701a875c2e8371fac7e9a9096
S0  f210a2d16546c566bb2d828255f56e9b1b260a565f4f7457048ca8e844ed1151
```

## Проверяемые утверждения

FOUND_LOCAL означает проверку текста локального оригинала, **не VERIFIED_CURRENT**
в LMS и не согласование команды. Номера страниц — физические PDF pages (в slides
есть повторные overlays).

| Requirements | Evidence | Статус / предел |
|---|---|---|
| A01–A06 | S1a p2–7: problem, dataset, baseline, team-trained NN, controlled quantitative comparison, honest discussion | Текст подтверждён локальным документом; реализация ещё не выполнена |
| A07–A08 | S1a p16, slide6: presentation и report content | Текст подтверждён; deliverables не готовы |
| A09 | S1a p22, slide9: presentation30% / report70%; S1b p4: project25% course | Локальные источники подтверждают числа; актуальность CSR pending |
| A10 | S1a p16: Sep20 proposal, Week14 presentation, Dec12 report/end of exam period | Year из титула Fall2026; точные часы/timezone/slot не указаны |
| A11 | S1a p17–20, slide7; S1b p7 | CONFLICT_PENDING: ограниченная помощь vs запрет code generation на assignments; см. ai_usage.md |
| B01–B08 | S2a/S2b §§2–6 | Обе версии подтверждают общий scope; exact 45k/5k, seeds42/123/2026, 40epochs, severity maps и 96evaluations — последующий план, не текст proposal |
| B09 | S2b heading и §7; S2a §7 | Шесть участников в candidate; actual roles/availability и submission pending |

## Открытые действия

SETUP-001 **PARTIAL**: поиск и traceability выполнены; actual rubric/LMS freshness,
AI policy conflict, submitted version/receipt и точные даты остаются pending.
Координация по прежней роли Person1; назначенный реальный owner пока не подтверждён.
Эти сведения не блокируют отдельную .venv и schema/config tests, но должны быть
разрешены перед утверждением course compliance и сдачей.
