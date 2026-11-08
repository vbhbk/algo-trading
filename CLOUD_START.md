# Старт облачной сессии

1. `bash research-tools/cloud_setup.sh` — `uv sync` и распаковка `data/*.csv.gz` в `data/*.csv`.
   Сжатые архивы лежат в git, распакованные csv в git не попадают (`.gitignore`).
2. Прочитать `FRESH_SESSION_BRIEF.md` (данные, разбивки, конвейер обучения, планка, что не работает)
   и `AGENTS.md`. Правила соревнования — в приложенной к сессии презентации.
3. В этом репозитории всё лежит здесь, а не в `../algo-trading/data` и не в `vladyslavzk/algo-trading`.

Данные в `data/`: `sample`, `sample_v2` (+ `_a/_b`), `round1` (+ `_a/_b`), `round2_public` (+ `_a/_b`,
`round2_public_train/valid`, `round2_t40`), `round3_public` (+ `_a/_b`, `round3_a2`),
`round4_public` (+ `_a/_b`; подрезки `round4_a1/a2`, `round4_b1/b2/b3`).

Стратегии: `strategies/strategy_v8_r3.py` (`TAKE_LOTS = 45`: round3_b 217 017; с `TAKE_LOTS = 95`: 224 864 — планка),
`strategy_v8_r3f95.py` (то же, обучено на всём R3, для загрузки), `strategy_v8.py`, `strategy_v5.py`, `max_r3_ml.py`,
`max_r2_combo.py` (маркет-мейкер без модели: round2_a/b 63 776 / 67 172). Конвейер: `logs/max/r3/record.py`, `fit.py`.

Коммиты: conventional commits, цифры в теле, без `Co-Authored-By` и упоминаний Claude. `simulator/` и `explorer/` не менять.

## Задача
<!-- впишите задание -->
