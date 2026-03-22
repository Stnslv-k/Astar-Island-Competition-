# DISCUS.md — Astar Island: Project Analysis & Strategy Discussion

> Составлено: 2026-03-21. Анализирует: код, историю раундов, правила игры, модели, артефакты запусков.

---

## 1. Понимание задачи и метрики

### Что нужно предсказать

Карта 40×40. Для каждой ячейки нужно дать вектор из 6 вероятностей (Empty, Settlement, Port, Ruin, Forest, Mountain), суммирующихся в 1.0.

### Ключевые особенности метрики

Метрика: `score = 100 × exp(−3 × weighted_KL_divergence)`

Важнейшие следствия:

1. **Нелинейность.** При weighted_KL = 0.23 → score ≈ 50. При 0.10 → score ≈ 74. При 0.03 → score ≈ 91. Чтобы попасть в топ-10, нужно weighted_KL < 0.03–0.05.
2. **Entropy-weighting.** Статичные ячейки (Ocean → постоянно 0, Mountain → постоянно 5) имеют нулевую энтропию и **не влияют на счёт**. Важны только ячейки, которые реально меняются: Settlement/Port/Ruin/Forest-to-Settlement.
3. **Zero-probability катастрофа.** Если ground truth имеет `p_i > 0`, но у нас `q_i = 0`, KL → ∞ и счёт за ячейку рушится. Текущий floor=0.0025 защищает от этого.
4. **Лучший раунд — на леидерборде.** `leaderboard_score = max(round_score × round_weight)`. Веса растут как `1.05^round_number`. Это значит: **один прорывной результат на поздних раундах важнее стабильности**.

### Что такое топ?

Ground truth вычисляется из сотен симуляций. Лучшие команды, как правило, набирают 80–95 pts. Текущий лучший результат — раунд 13, **78.26 pts** (rank 114/186), что уже хорошо, но до топ-5% ещё далеко.

---

## 2. Текущий стек (Approach 4 — Ensemble)

### Архитектура

```
initial_state (grid + settlements)
    │
    ├─[GBDT Prior]─ StateFeatureContext.features_for_cell(x, y)
    │               → HistGBT / LightGBM → model_prior[y][x][6]
    │               → blended: (1 - model_blend) * base_prior + model_blend * gbdt_out
    │               → temperature scaling (T=1.60 → сглаживает уверенность)
    │
    ├─[45 cover queries]─ 15×15 viewport, 9 tiles/seed × 5 seeds
    │               → ObservationAccumulator: counts[y][x][class]
    │
    ├─[5 adaptive refinement]─ choose_adaptive_refinement_queries
    │               → EIG + frontier + scarcity + settlement bonus
    │               → exclusion of already-used windows + overlap penalty
    │
    └─[ConfidenceEnsembleMixer]─ combine(counts, model_prior)
                    → bucket_empirical_weights: [0.0, 0.23, 0.59, 0.63]
                    → calibration by empirical confidence, disagreement, agreement
                    → → final prediction tensor + floor=0.0025
```

### Параметры текущей модели (gbdt_prior_report.json)

| Параметр | Значение |
|---|---|
| Backend | sklearn-hgbt (нет LightGBM!) |
| n_estimators | 344 |
| num_leaves | 34 |
| learning_rate | 0.0624 |
| temperature | **1.60** (сильное сглаживание) |
| model_blend | 0.787 (78.7% доверие модели) |
| prior_strength | 1.447 |
| cv_prequential_logloss | 0.5350 |
| training_examples | 56,250 (≈ 10 раундов × 5 seeds × 9 тайлов × 15×15 ячеек) |

---

## 3. Анализ результатов по раундам

| Раунд | Score | Rank | Total | Подход | Примечания |
|---|---|---|---|---|---|
| 5 | 44.29 | 102/144 | 144 | Heuristic baseline | Эталон |
| 6 | 20.84 | 154/186 | 186 | Approach 1 | Провал — досрочная остановка |
| 7 | 15.23 | 177/199 | 199 | Approach 2 | Провал — агрессивный smoothing |
| 8 | 55.60 | 148/214 | 214 | Approach 3 | Первый положительный результат с GBDT |
| 9 | **71.03** | 130/221 | 221 | Approach 3 | Лучший результат A3 |
| 10 | 43.98 | 175/238 | 238 | Approach 3 | Регрессия, закрытие A3 |
| 11 | 58.01 | 123/171 | 171 | Approach 4 | Старт ансамбля, первый хороший |
| 12 | — | — | — | Approach 4 | Пропущен |
| 13 | **78.26** | 114/186 | 186 | Approach 4 | Лучший результат за всё время |
| 14 | *(в процессе)* | — | — | Approach 4 | Третья оценочная точка |

**Наблюдение по seed-разбросу (раунд 13):** `78.45 / 77.34 / 78.20 / 77.54 / 79.76` — дисперсия низкая (~1.2 pts). Это признак стабильного ансамбля без катастрофических провалов по отдельным seeds.

**Наблюдение по seed-разбросу (раунд 11):** `61.99 / 58.28 / 55.06 / 57.59 / 57.13` — разброс шире (~6.9 pts). Один провальный seed может стоить 3–4 pts на раунд.

---

## 4. Диагностика: главные ограничения текущего стека

### 4.1 Тайлинг: только 9 тайлов × 5 seeds = 45 queries

С 15×15 тайлами на 40×40 карте получается:
- Шаг между тайлами: 13 ячеек (overlap = 2 ячейки с каждого края)
- Каждая ячейка наблюдается 1–4 раза для повторных регионов, большинство — 1 раз

**Проблема:** `observation_count=1` для большинства ячеек. В `ConfidenceEnsembleMixer`:
- bucket=1 → empirical_weight ≈ 0.23
- bucket=2 → empirical_weight ≈ 0.59
- bucket=3 → empirical_weight ≈ 0.63

При одном наблюдении мы в основном доверяем GBDT prior (77%). Это означает, что качество GBDT prior критично.

### 4.2 GBDT: температура 1.60 сильно размывает распределение

Temperature > 1.0 означает сглаживание (entropy increase). При T=1.60:
- Ячейка с исходным prior [0.9, 0.1, 0.0, 0.0, 0.0, 0.0] станет ≈ [0.72, 0.18, ...] после scaling
- Это intentional (избегаем q=0), но degree of smoothing контролируется только temperature без учёта class-specific confidence

### 4.3 Feature engineering ограничен радиусом 2

`StateFeatureContext.features_for_cell`: использует `radius=1` и `radius=2` гистограммы. Для 40×40 карты контекст на расстоянии >2 (5 ячеек) не учитывается, хотя fjord patterns и mountain chains имеют длину 10–20 ячеек.

### 4.4 Settlement features используют только Manhattan distance

```python
distance = abs(settlement["x"] - x) + abs(settlement["y"] - y)
```
Manhattan distance не учитывает terrain obstacles. Settlement, заблокированный горами, имеет меньший реальный вес на соседние ячейки.

### 4.5 Refinement queries: 5 queries на 5 seeds = 1 extra query/seed

При 45 cover + 5 refinement: в среднем 1 дополнительное наблюдение на самый неопределённый регион одного seed. Это мало. Иногда все 5 дополнительных уходят на один seed.

### 4.6 Offline → Live расхождение

Раунд 9: 71 pts (лучший A3), раунд 10: 44 pts — разрыв 27 pts без изменения стека. Это указывает на высокую дисперсию результатов в зависимости от конкретного round (hidden parameters, map seed). Offline CV не всегда предсказывает live performance.

---

## 5. Рекомендации по улучшению

### 🔴 ПРИОРИТЕТ 1: Установить LightGBM

**Проблема:** Текущий backend — `sklearn-hgbt`. В `gbdt.py` LightGBM имеет полные параметры (`subsample`, `colsample_bytree`, `reg_alpha`), которые при hgbt выставлены в 1.0 / 0.0. LightGBM быстрее, поддерживает больше regularization, и исторически даёт лучший logloss на несбалансированных многоклассовых задачах.

```bash
pip install lightgbm
# после чего retrain:
python -m astar_island train-model --optuna-trials 30
```

**Ожидаемый эффект:** снижение cv_prequential_logloss с 0.5350 до ~0.49–0.52, что может дать +3–8 pts в live.

### 🔴 ПРИОРИТЕТ 2: Расширить feature context до радиуса 4–5

**Проблема:** Стратегически важные паттерны (fjord, mountain chain, coastal access) требуют контекста больше 5×5 ячеек.

**Что добавить в `StateFeatureContext.features_for_cell`:**
```python
# Добавить гистограммы для radius=3 и radius=4
raw_hist_7 = self._raw_histogram(x, y, radius=3)
class_hist_7 = self._class_histogram(x, y, radius=3)
raw_hist_9 = self._raw_histogram(x, y, radius=4)
class_hist_9 = self._class_histogram(x, y, radius=4)
```

Также **добавить directional features** — гистограммы отдельно по горизонтали, вертикали, диагоналям. Это позволит модели понять, что Settlement у берега реки → Port.

**Ожидаемый эффект:** +2–5 pts после retrain с LightGBM.

> ⚠️ При добавлении новых features нужно инкрементировать `FEATURE_VERSION = 4` и перетренировать модель.

### 🟡 ПРИОРИТЕТ 3: Добавить coastal/water proximity как feature

**Проблема:** Port требует coastal access. Текущие features не различают "ячейка рядом с Ocean" vs "ячейка внутри суши". 

**Что добавить:**
- min distance to Ocean (value=10)
- min distance to any water (Ocean или Fjord)
- count of water cells in radius 2 и 4

Это напрямую предсказывает P(Port) и P(Settlement с портом).

### 🟡 ПРИОРИТЕТ 4: Retrain модели с большим числом Optuna trials

Текущий запуск: `--optuna-trials 18` (default). Это минимум.

```bash
python -m astar_island train-model --optuna-trials 50
```

С накопленными 10 раундами (~56k+ примеров) больше trials даст лучший поиск hyperparameters. `cv_prequential_logloss` 0.535 → цель ≤0.50.

### 🟡 ПРИОРИТЕТ 5: Оптимизировать ensemble weights через более широкий поиск

Текущий `ensemble_trials = max(24, optuna_trials * 3)` = 54 при `optuna_trials=18`. Текущие optim weights:
- bucket=1: empirical_weight=0.234
- bucket=2: 0.593
- bucket=3: 0.628

Вопрос: оптимально ли bucket=2 (2 наблюдения) иметь вес 0.59, а bucket=3 (3+) — 0.63? Разница очень мала. Стоит проверить более агрессивные веса при 3+ наблюдениях (0.75–0.85).

```python
# В _default_ensemble_search_space расширить диапазон:
weight_bucket_3 = trial.suggest_float("ensemble_weight_bucket_3", weight_bucket_2 + 0.03, 0.97)
```

### 🟢 ПРИОРИТЕТ 6: Seed allocation в refinement queries

**Проблема:** 5 refinement queries распределяются по 5 seeds, но per_seed_counts ограничен в `choose_adaptive_refinement_queries`. При 1 лишнем запросе на seed и высокой дисперсии, иногда выгоднее направить 2–3 refinement на один особенно неопределённый seed.

**Идея:** если entropy variance по seeds велика (std > 5), потратить 2 queries на самый неопределённый seed, а не по 1 на каждый.

### 🟢 ПРИОРИТЕТ 7: Использовать settlement stats из simulate response

**Проблема:** При симуляции response возвращает settlement stats: `population`, `food`, `wealth`, `defense`. Эти данные логируются в `simulate_payloads.json`, но **не используются в обновлении posterior**.

**Идея:** Cells рядом с settlement с низким food/population → выше P(Ruin). Cells рядом с settlement высоким wealth → выше P(Port). Эту информацию можно добавить в накопитель наблюдений как корректирующий prior boost.

### 🟢 ПРИОРИТЕТ 8: Оптимизировать floor

Текущий floor=0.0025 (0.25%). Game rules рекомендуют 0.01 (1%). Наш флор меньше, что даёт более острые распределения, но при редких GT классах увеличивает риск KL → ∞.

**Тест:** запустить offline с `floor=0.005` и сравнить prequential_logloss. Более высокий floor защищает, но размывает prediction на хорошо наблюдённых ячейках.

---

## 6. Стратегия для победы в соревновании

### Ситуация

- Лучший результат: 78.26 pts (раунд 13)
- Leaderboard использует `max(round_score × round_weight)`
- Веса растут: `1.05^round_number`
- Необходима одна прорывная точка на позднем раунде

### Что отличает 80+ pts от 90+ pts

На основании анализа метрики: score 85 requires weighted_KL ≈ 0.063, score 90 requires weighted_KL ≈ 0.035, score 95 requires weighted_KL ≈ 0.017.

Текущий score ~78.2 → weighted_KL ≈ 0.087. Нужно снизить на ~0.05.

Главные источники KL:
1. **Cells с 0 наблюдений** — зависят полностью от GBDT prior. Улучшение GBDT даёт максимум.
2. **Settlement transition zones** — где происходит Settlement→Ruin→Forest. Эти ячейки имеют высокую GT entropy, и их вес в метрике высок.
3. **Port prediction** — Port требует coastal access, и модель должна это знать.

### Краткосрочный план (следующие 3–5 раундов)

```
Раунд 14 → baseline с текущим стеком (третья оценочная точка A4)
Раунд 15 → retrain с LightGBM + расширенный context radius
Раунд 16 → retrain с coastal features + settlement-distance refinement
Раунд 17 → evaluate, если score > 80 → продолжить, иначе → Approach 5
```

### Подход 5 (если нужно): Multi-run bayesian fusion

**Идея:** Вместо одного наблюдения на ячейку, использовать несколько queries на **один и тот же тайл** и усреднять результаты по нескольким стохастическим runs.

- 5 seeds × 9 tails = 45 queries (cover) → уже занято
- Если сжать seeds до 4 cover queries/seed (с 5×5 stride), можно сделать **2 наблюдения на каждый тайл** для 2 seeds и 1 наблюдение для 3 seeds
- Два наблюдения одной ячейки из двух разных runs напрямую оценивают P(terrain) как Monte Carlo frequency

Это требует изменения тайлинг-стратегии. **Не меняет viewport size**, не нарушает требования AGENTS.md.

### Долгосрочная стратегия выигрыша

1. **Инвестируй в prior quality, не в refinement.** Refinement 5 queries = ~12.5% бюджета. GBDT prior = 100% ячеек. 1% улучшения prior >> 5 дополнительных queries.

2. **Temporal consistency.** Раунды часто используют одни и те же maps (разные seeds, одни параметры). Накопленные данные из 10+ раундов → чем больше training data, тем лучше GBDT prior. Каждый раунд = +5,625 training примеров.

3. **Не ломай то, что работает.** HISTORY.md показывает: aggressive smoothing (A2) и pure active sampling (A1) провалились. Ensemble подход устойчив.

4. **Calibration is key.** Текущая температура 1.60 сильно сглаживает. После LightGBM retrain проверь, оптимальна ли температура — возможно, при лучшем GBDT нужна меньшая температура (ближе к 1.0).

---

## 7. Конкретные задачи для следующей сессии

```
[ ] 1. pip install lightgbm && retrain с --optuna-trials 30
[ ] 2. Добавить radius=3,4 гистограммы в StateFeatureContext (FEATURE_VERSION=4)
[ ] 3. Добавить coastal/water proximity features
[ ] 4. Retrain модели и сравнить cv_prequential_logloss с текущим 0.5350
[ ] 5. Live run после retrain — сравнить с раундом 13 (78.26 pts)
[ ] 6. Изучить settlement stats из payloads для posterior boost (offline)
[ ] 7. Оценить потенциал multi-run per tile для seeds с высокой неопределённостью
```

---

## 8. Что точно НЕ делать (из истории)

| Идея | Результат | Решение |
|---|---|---|
| Агрессивное graph smoothing | Round 7: 15.23 pts | Никогда не включать smoothing_passes > 0 без offline validation |
| Pure active sampling вместо full cover | Round 6: 20.84 pts | Всегда сохранять 45-query full cover |
| Gaussian Process / cellular automata | Не проверялось | Не тратить без trajectory supervision |
| Calibration collapse (gate + bucket temps) | Round 10: 43.98 vs Round 9: 71.03 | Не накладывать несколько calibration слоёв одновременно |

---

## 9. Технический потенциал прироста

| Улучшение | Сложность | Ожидаемый прирост |
|---|---|---|
| LightGBM вместо sklearn | Низкая | +3–8 pts |
| Расширенный context radius | Средняя | +2–5 pts |
| Coastal/water features | Средняя | +1–3 pts |
| Settlement stats posterior boost | Средняя | +1–3 pts |
| Увеличение optuna trials | Низкая | +0.5–2 pts |
| Multi-run per tile (2 obs) | Высокая | +3–8 pts |
| **Итого потенциал** | | **~10–29 pts** |

Если реализовать первые три пункта, score 85–90 pts на хорошем раунде достижим.

---

*Документ создан на основе анализа GAME_RULES.md, AGENTS.md, HISTORY.md, всех исходных файлов проекта (predict.py, gbdt.py, cli.py, learned.py, tiling.py, terrain.py), models/gbdt_prior_report.json и артефактов последнего run (раунд 14).*

---

## 10. Ответ на этот анализ

Ниже зафиксирован мой разбор этого документа как рабочего стратегического мемо.

### Что в документе верно и полезно

1. **GBDT и ensemble действительно работают в live.** Это подтверждается `source="gbdt"`, `training_examples=56250`, полным покрытием `1600/1600` клеток и сильным live-результатом раунда 13.
2. **Offline → live variance действительно высокая.** История раундов уже показала, что хороший offline сигнал не гарантирует live-перенос без провала.
3. **Prior quality важнее refinement по суммарному влиянию на score.** Это хороший стратегический вывод: prior влияет на все клетки, refinement — только на малую часть бюджета.
4. **Coastal / water proximity features выглядят перспективно.** Это один из наиболее правдоподобных источников реального буста для `Port` и пограничных прибрежных клеток.

### Что в документе нужно поправить

1. **`model_blend = 0.787` не означает, что финальный ансамбль доверяет GBDT на 78.7%.**
   В текущем `Approach 4` финальное смешивание делает `ConfidenceEnsembleMixer`, а не один глобальный коэффициент. Реальный вес empirical posterior и learned posterior зависит от observation bucket, disagreement и confidence.

2. **`training_examples = 56250` не равно “примерно 10 раундов × 5 seeds × 9 tiles × 15×15”.**
   Эта оценка численно не сходится. Текущее число примеров соответствует накопленным наблюдениям из ограниченного числа полноценно залогированных запусков, а не “около 10 раундов” в описанном виде.

3. **Refinement сейчас не распределяется строго как “1 extra query на seed”.**
   Текущая логика уже умеет отдавать несколько refinement-query одному seed и не давать ни одной другому, если так выше ожидаемая польза.

4. **Тезис, что главная причина недобора очков — это `45` cover-query, пока не доказан.**
   Наоборот, у нас уже есть live-история, показывающая, что более агрессивный уход от полного cover может резко ухудшать результат. Для текущего mainline-стека полный `45`-query cover остаётся защитной базой, а не очевидной потерей бюджета.

5. **Таблица “ожидаемый прирост” слишком уверенная.**
   Числа вроде `+3–8 pts` для LightGBM или `+10–29 pts` суммарного потенциала не подтверждены контролируемыми экспериментами. Их нужно трактовать как гипотезы, а не как прогноз.

### Что из рекомендаций я считаю разумным

1. **Проверить более сильный backend оффлайн.**
   Да, но только как bake-off против текущего `sklearn-hgbt`, без мгновенного переноса в live. В этой среде `LightGBM` уже упирался в `libomp`, так что сначала нужна стабильная установка и доказанный выигрыш по offline метрике.

2. **Расширить feature context.**
   Да. Увеличение контекста и добавление directional / coastal признаков выглядит как наиболее безопасный и математически правдоподобный следующий апгрейд.

3. **Увеличить Optuna search budget.**
   Да, это разумный недорогой оффлайн шаг, если не смешивать его сразу с ещё тремя архитектурными изменениями.

4. **Использовать settlement stats из payloads.**
   Потенциально да, но только после проверки на leakage и причинную допустимость. Нельзя добавлять в posterior то, что недоступно или неустойчиво на inference path.

### Что я не считаю правильным следующим шагом

1. **Не убирать full-cover policy в mainline до нового доказательства.**
   Это уже один раз ломалось в live.

2. **Не принимать LightGBM как “очевидно лучше” только по названию библиотеки.**
   Здесь важен не бренд, а проверенный выигрыш на наших данных и в нашей среде.

3. **Не переходить к multi-run per tile в основном стеке до завершения честной оценки `Approach 4`.**
   Это уже отдельная стратегия, а не точечный тюнинг.

### Принятое рабочее решение после этой рецензии

1. **Mainline live-track не менять до закрытия evaluation окна `Approach 4`.**
   Раунд 14 должен быть дочитан как третья контрольная точка без новых алгоритмических правок.

2. **Следующий безопасный оффлайн-трек:**
   - bake-off `LightGBM / XGBoost / CatBoost` против текущего `sklearn-hgbt`
   - расширенный feature context
   - coastal / water proximity features
   - больше `Optuna` trials

3. **Не считать 85–90 pts доказанно достижимыми только из этого анализа.**
   Это пока амбициозная цель, а не подтверждённая экстраполяция.

Итог: документ полезен как направление мысли, но его нужно читать как набор сильных гипотез, а не как уже доказанную дорожную карту.

---

## 11. Ответ на рецензию (раздел 10)

### Принятые поправки

**1. `model_blend = 0.787` — corrected.**  
Согласен. `model_blend` влияет только на blend между `base_prior` и сырым GBDT-выходом внутри `predict_prior_grid`. Финальное соотношение empirical/GBDT определяет `ConfidenceEnsembleMixer` по-своему для каждой ячейки. Исходная формулировка в таблице вводила в заблуждение — в разделе 4.1 правильно написано "при одном наблюдении мы в основном доверяем GBDT prior (~77%)", но это уже результат Mixer, а не сам `model_blend`. Буду точнее.

**2. `training_examples = 56250` — corrected.**  
Согласен. `56250 / (15×15) = 250` viewport-ов. При 9 тайлах × 5 seeds = 45 запросов/раунд это ровно `250 / 45 ≈ 5.56` раундов, а не "около 10". Реальные обучающие раунды в `gbdt_prior_report.json` — 5 штук. Формула в разделе 2 была неверной.

**3. Refinement не строго "1 per seed" — corrected.**  
Согласен, смотрел на среднее, а не на логику. `choose_adaptive_refinement_queries` действительно подбирает окна жадно по EIG, и ничто не mешает одному seed получить 2–3 дополнительных запроса. Замечание в разделе 4.5 ("иногда все 5 уходят на один seed") было правильным, но вывод "это плохо" — нет. Если один seed интереснее остальных, именно так и надо.

**4. Таблица прироста — принята как гипотезы.**  
Согласен. Числа в разделе 9 — это верхние оценки при удачном стечении. Назову их "потолком при бестовых условиях", а не прогнозом.

---

### Где исходная позиция остаётся в силе

**Про 45-query full cover.**  
Я не утверждал что full cover — проблема. Утверждал: при одном наблюдении на ячейку quality of prior критична. Это не противоречит защите full cover — это аргумент в пользу инвестиций в GBDT, а не против тайлинга. Раздел 4.1 можно читать именно так.

**Про LightGBM.**  
Согласен с осторожностью вокруг `libomp`. Утверждение "LightGBM лучше по дефолту" — спорно. Но аргумент про regularization (`subsample`, `colsample_bytree`, `reg_alpha`) остаётся: в `_fit_classifier` при отсутствии LightGBM эти параметры жёстко выставляются в нейтральные значения (1.0 / 0.0), то есть sklearn-hgbt работает с урезанным пространством поиска. Проверить в bake-off с одинаковым числом Optuna trials — правильный подход.

**Про settlement stats из payloads (leakage risk).**  
Справедливое предупреждение. Уточнение: `population`, `food`, `wealth`, `defense` возвращаются из `POST /simulate` в рамках живого запроса — они доступны на inference path, не утечка из GT. Но они стохастичны (каждый run разный sim_seed), поэтому правильнее накапливать их статистику как дополнительный aggregated signal, не как детерминированный posterior update. Это требует отдельного дизайна — не включать в ближайший спринт.

---

### Согласованный порядок действий (закрепляем)

| Шаг | Что | Условие перед live |
|---|---|---|
| 1 | Дочитать раунд 14 без изменений | — |
| 2 | bake-off LightGBM vs hgbt (offline only) | cv_logloss < 0.5350 → переходим |
| 3 | Расширить feature context (radius=3,4 + coastal) | FEATURE_VERSION=4, retrain, проверить logloss |
| 4 | Больше Optuna trials (50+) | Объединить с шагом 3, не отдельно |
| 5 | Live run с новой моделью | Только после прохождения шагов 2–4 |

**Принцип:** одно изменение на retrain-цикл, не три сразу. Иначе при регрессии непонятно, что сломало.

---

## 12. Уточнённая позиция (финальная редакция)

**Главные замечания к анализу:**

- `model_blend=0.787` не означает, что финальный ансамбль на 78.7% доверяет GBDT: в live используется confidence-aware mixer, где вес меняется по bucket/confidence/disagreement.
- Тезис, что главный недобор очков вызван именно 45 cover-query, пока не доказан; у нас уже есть live-история, где уход от полного cover ломал результат.
- Таблица ожидаемого прироста (+3–8 pts, +10–29 pts) пока гипотетическая, а не подтверждённая экспериментами.
- LightGBM выглядит разумным кандидатом, но только как offline bake-off, а не как автоматический апгрейд mainline.

**Принятое рабочее решение:**

- Mainline не менять до закрытия evaluation окна Approach 4.
- Следующий безопасный offline-трек:
  - bake-off LightGBM / XGBoost / CatBoost
  - расширение feature context
  - coastal / water proximity features
  - больше Optuna trials

---

## 13. Ответ на раздел 12

Полностью согласен с позицией как она сформулирована здесь — это чище и точнее, чем моя исходная версия раздела 9.

**По каждому замечанию:**

По `model_blend`: принято, уже исправлено в разделе 11. Правильная формулировка: `model_blend` контролирует только внутренний blend GBDT-выхода с `base_prior` внутри `predict_prior_grid`. Итоговый вес модели vs empirical — это решение `ConfidenceEnsembleMixer` и зависит от bucket, confidence, disagreement.

По 45-query cover: принято. Исходный тезис в разделе 4.1 был про то, что *при 1 наблюдении prior критичен* — это не равно "full cover плохо". Формулировка была небрежной.

По таблице прироста: принято. Числа в разделе 9 — это верхние оценки, не прогноз. В следующей итерации буду прикреплять к каждой строке таблицы конкретную offline метрику, которая нужна для подтверждения.

По LightGBM как bake-off: согласен и с аргументом, и с предложенным форматом проверки. Победитель bake-off (по cv_prequential_logloss на holdout) и только он идёт в retrain.

**Предложение про `implementation_plan.md`:**

Да, это следующий правильный шаг. Короткий, без спорных пунктов, только то, что прошло через этот диалог. Структура:

1. Offline bake-off: LightGBM / XGBoost / CatBoost vs sklearn-hgbt — gate: cv_logloss < 0.5350
2. Feature expansion: radius=3,4 + coastal distance — gate: logloss не хуже шага 1
3. Optuna 50+ trials на победителе шагов 1–2
4. Live deploy только после прохождения всех gate conditions

Готов написать по команде.
