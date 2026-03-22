# Astar Island History

## Purpose

This file records verified hypotheses, rejected ideas, accepted decisions, and the reasons behind them.
It is the canonical project memory for what was tried, what failed, and what should not be repeated blindly.

## Verified Failed Hypotheses

- Hypothesis: stronger active sampling heuristics on the last `5` queries would improve the baseline without changing the full-cover policy.
Reason it failed: the first live test under this approach produced a clearly negative result.
Evidence: round `6`, completed on `2026-03-20`, scored `20.8444 pts`, rank `154/186`, versus heuristic baseline round `5` at `44.2909 pts`, rank `102/144`.
Decision: stop `Approach 1` early and move on.

- Hypothesis: graph-style edge-aware smoothing would improve local consistency and therefore live score.
Reason it failed: the first live test under this approach collapsed even further than the stronger active-sampling attempt.
Evidence: round `7`, completed on `2026-03-20`, scored `15.2289 pts`, rank `177/199`.
Decision: stop `Approach 2` early and do not reintroduce aggressive smoothing by default.

- Hypothesis: the learned gate plus bucketed calibration stack, which looked better offline, would transfer directly to live play.
Reason it failed: offline prequential validation improved materially, but the first live round on that stack regressed badly.
Evidence: offline bundle improved to `cv_prequential_logloss=0.5541`, but round `10`, completed on `2026-03-21`, scored only `43.9795 pts`, rank `175/238`, after round `9` had scored `71.0297 pts`, rank `130/221`.
Decision: close `Approach 3` after 3 rounds and do not spend a 4th live round on that exact configuration.

- Hypothesis: the frequency fallback prior is a safe temporary substitute for the stable round-13 GBDT ensemble path.
Reason it failed: the emergency fallback submitted successfully, but live quality collapsed across all seeds.
Evidence: round `15`, completed on `2026-03-21`, scored `20.9269 pts`, rank `244/262`, with seed scores `24.5169`, `20.3268`, `19.6928`, `21.8952`, `18.2027`.
Decision: do not treat frequency fallback as a competitive live path; use it only as an emergency last resort when the trained bundle cannot run.

## Verified Disproved Hypotheses

- Hypothesis: `learned.py` is empty or cannot read historical data in this environment.
Reason it is false: direct invocation of `build_learned_prior_model('runs', '.astar_cache.sqlite3')` succeeds in this workspace.
Evidence: the fallback frequency model loads `67500` examples locally.
Decision: do not blame `learned.py` or the `runs/` directory for current live underperformance.

- Hypothesis: `model_examples=0` in `summary.json` means no learned data was used in the live run.
Reason it is false: that field refers only to the old frequency fallback model, not to the active GBDT bundle.
Evidence: fresh summaries show active `gbdt_prior.training_examples=56250` with per-seed diagnostics `source="gbdt"`.
Decision: keep telemetry explicit about the active source to avoid future misreads.

## Accepted Decisions

- Decision: keep the hard requirement of `15x15` live simulation and full `45`-query cover.
Reason: replacing full cover with all-budget active sampling is too risky for `40x40` maps and has not shown live evidence of benefit.

- Decision: treat the last `5` queries as the only safe place for active learning.
Reason: that preserves full-map observation coverage while still allowing targeted information gain where the posterior is uncertain.

- Decision: interpret sensor-placement ideas as redundancy-aware window selection, not as Gaussian-process modeling.
Reason: this task is a noisy discrete categorical grid with sharp transitions, not a smooth continuous field.

- Decision: do not spend time on cellular-automata modeling without explicit trajectory supervision.
Reason: current data contains noisy viewport observations and initial states, not time-series evolution labels.

- Decision: move to `Approach 4` as the current live path.
Reason: after three live rounds, `Approach 3` was mixed and not strong enough to justify a fourth live round on the same posterior stack.

- Decision: use a safe confidence-aware ensemble as the current posterior builder.
Reason: the current offline bundle improved to `posterior_mode=ensemble` with `cv_prequential_logloss=0.5349969331891595`, better than the previous learned-gate stack at `0.5540632020928694`.

- Decision: use expected-information-gain approximation plus overlap penalty for the last `5` refinement windows.
Reason: it is the most mathematically justified next step that can be added without breaking the full-cover policy.

- Decision: the next model-change cycle after the current `Approach 4` evaluation window will be gate-driven and offline-first.
Reason: the current live stack is finally producing strong results, so the next change should only be introduced after passing a backend bake-off, feature-expansion gate, and larger hyperparameter search, rather than by mixing multiple speculative changes directly into live rounds.

- Decision: do not spend a 4th unchanged live round on `Approach 4`.
Reason: after three completed evaluation rounds, `Approach 4` produced one breakout score but the overall top-entry trajectory remained mixed rather than clearly strengthening, so the next move should be an offline-first model-improvement cycle instead of another unchanged deployment.

- Decision: after the first coarse backend screen, keep `sklearn-hgbt` in the live path until a stronger candidate passes the full offline gate.
Reason: enabling `libomp` made `LightGBM`, `XGBoost`, and `CatBoost` runnable, but the first coarse LORO screen only identified `CatBoost` as the best default-parameter candidate; no backend came close to the actual deployment gate based on the tuned ensemble baseline.

- Decision: do not replace the current live backend with `CatBoost` after the first `FEATURE_VERSION=4` gate run.
Reason: the first `CatBoost + v4 features + mixer tuning` run improved its own coarse base score materially, but still landed at `cv_prequential_logloss=0.6127`, far above the live-gate target derived from the current tuned bundle (`0.5350`).

- Decision: keep the next offline focus on the current `sklearn-hgbt` backend rather than on backend replacement.
Reason: the first `sklearn-hgbt + v4 features + mixer tuning` run beat the corresponding `CatBoost v4` run (`0.5676` vs `0.6127`), which means the new feature set looks more promising on the existing backend than the attempted backend swap.

- Decision: do not allow live runs to abort when the serialized GBDT bundle is incompatible with the current feature space.
Reason: once `FEATURE_VERSION=4` landed, the existing live bundle at `models/gbdt_prior.joblib` could still be present while being trained on an older feature shape, which caused `run` to fail before spending any API budget.
Evidence: the first round `15` live attempt on `2026-03-21` failed locally with `X has 124 features, but HistGradientBoostingClassifier is expecting 71 features as input.` and recovered only after a manual rerun with `--disable-gbdt-prior`.
Decision: `run` now auto-disables the GBDT bundle on runtime incompatibility and falls back to the frequency prior instead of aborting the round.

- Decision: restore `feature_version` compatibility so the proven `feature_version=3` GBDT bundle can remain usable while `feature_version=4` tuning continues offline.
Reason: the stable round-13 path is materially stronger than the emergency frequency fallback, and the current full v4 tuning loop is too slow to rely on for the active round window.
Evidence: `models/gbdt_prior.joblib` is `feature_version=3`, `cv_prequential_logloss=0.5350`, and was the active source in round `13` (`78.2571 pts`); the emergency frequency fallback in round `15` collapsed to `20.9269 pts`.
Decision: `GBDTPriorBundle.predict_prior_grid()` now builds features using the bundle's own `feature_version`, so old stable bundles and newer code can coexist safely.

- Decision: do not deploy `models/sklearn_v4_tuned1.joblib` to live.
Reason: the completed full `FEATURE_VERSION=4` tuning run still underperforms the current stable live bundle on the offline gate metric.
Evidence: the finished run on `2026-03-21` produced `cv_prequential_logloss=0.559430506272988`, which is worse than the stable `feature_version=3` bundle at `0.5349969331891595`.
Decision: keep the restored `feature_version=3` bundle as the live path and treat `sklearn_v4_tuned1` as a failed gate candidate.

- Decision: for the active round `17`, orchestrate live submission around a full `feature_version=3` retrain instead of waiting manually.
Reason: ranking dropped sharply and the next round cannot be left to ad-hoc manual timing.
Evidence: round `16` confirmed the restored stable `feature_version=3` path is strong (`70.8183 pts`), while the newly finished `feature_version=4` candidate failed the offline gate.
Decision: a one-shot training job for `feature_version=3` is scheduled immediately, and a one-shot deadline submit job is scheduled for `+150` minutes. If the new `v3` candidate finishes and beats the stable report by a real margin, it submits; otherwise the deadline job submits the stable bundle.

- Decision: keep the restored stable `feature_version=3` bundle as the only live path after round `17`.
Reason: round `17` produced another solid live result without any new offline candidate beating the current baseline, while the scheduled retrain path did not yield a deployment-worthy replacement before live play mattered.
Evidence: round `17`, completed on `2026-03-21`, scored `64.8617 pts`, rank `193/283`, with seed scores `65.8869`, `66.6285`, `64.6511`, `64.8430`, `62.2989`; no newer offline bundle has beaten the stable `cv_prequential_logloss=0.5349969331891595`.
Decision: keep `models/gbdt_prior.joblib` as the live bundle for the next round and treat new training runs as offline-only until one clears the existing gate by a real margin.

- Decision: optimize the offline GBDT retrain path by caching historical feature matrices and observation views per run.
Reason: retrain latency, not just model quality, has become a hard operational bottleneck for active rounds.
Evidence: the full tuned runs take hours, and the prequential evaluation loop was repeatedly rebuilding grid features and observation maps for the same historical runs across folds and tuning trials.
Decision: `astar_island/gbdt.py` now caches per-run feature matrices by `feature_version` and cached observation maps, and reuses them during `build_training_matrix()` and `evaluate_bundle_prequential_logloss()`.

- Decision: do not deploy `models/gbdt_v3_retrain2.joblib` to live.
Reason: the accelerated stable-`v3` retrain still failed the offline gate badly despite using more history.
Evidence: `models/gbdt_v3_retrain2_report.json` on `2026-03-22` produced `training_examples=135000`, `feature_version=3`, and `cv_prequential_logloss=0.6615468842343689`, which is far worse than the stable live bundle at `0.5349969331891595`.
Decision: keep `models/gbdt_prior.joblib` as the only live bundle and treat `gbdt_v3_retrain2` as another failed offline candidate.

- Decision: stop comparing deployment candidates by raw `cv_prequential_logloss` when the models were trained and cross-validated on different historical round sets.
Reason: those scores are not comparable across different evaluation pools and were causing false rejections of candidates that actually perform better on the rounds we care about now.
Evidence: `gbdt_v3_retrain2` looked much worse than the stable bundle by its own raw report (`0.6615` vs `0.5350`), but on a common recent-history benchmark over rounds `14-19` it beat the stable bundle `0.6778` vs `0.7142`, an improvement of about `0.0364`.
Decision: deployment gating must use head-to-head evaluation on the same recent historical rounds. A benchmark-aware model comparison path is now implemented in `astar_island/gbdt.py`, `astar_island/cli.py`, and `scripts/round_submit_common.py`.

- Decision: promote `gbdt_v3_retrain2` as the next live bundle after benchmark comparison.
Reason: despite its worse raw report, it is materially better than the old stable bundle on the most recent common-history benchmark and is therefore the better bet for the next live round.
Evidence: `models/gbdt_live_selection_report.json` shows recent-round benchmark average prequential logloss `0.6778` for the candidate vs `0.7142` for the previous stable bundle on rounds `14-19`.
Decision: `models/gbdt_prior.joblib` and `models/gbdt_prior_report.json` now point to the promoted candidate; the previous stable bundle was preserved as `models/gbdt_prior_stable_backup_20260322T031440Z.joblib` and matching report.

## Current Status

- Current live approach: `Approach 4`
- Current bundle: `models/gbdt_prior.joblib`
- Current posterior mode: `ensemble`
- Latest offline validation: `cv_prequential_logloss=0.5349969331891595`
- Round `11`, completed on `2026-03-21`, scored `58.0097 pts`, rank `123/171`.
- Round `11` seed-level scores: `61.9860`, `58.2812`, `55.0643`, `57.5882`, `57.1286`.
- Interpretation of round `11`: first live result for `Approach 4` was positive versus round `10` (`43.9795 pts`) and round `8` (`55.5983 pts`), but still below round `9` (`71.0297 pts`), so the direction is promising but not yet decisive.
- Round `12` was missed and produced no submission from this workspace.
- Round `13`, completed on `2026-03-21`, scored `78.2571 pts`, rank `114/186`.
- Round `13` seed-level scores: `78.4480`, `77.3383`, `78.1995`, `77.5400`, `79.7598`.
- Interpretation of round `13`: this is the strongest completed result so far, clearly above round `11` (`58.0097 pts`) and round `9` (`71.0297 pts`), with no weak seed dragging the average down.
- Round `14`, completed on `2026-03-21`, scored `55.8597 pts`, rank `146/244`.
- Round `14` seed-level scores: `53.9839`, `58.1836`, `56.4144`, `56.4328`, `54.2840`.
- Interpretation of round `14`: the result held above the older heuristic baseline and above round `10`, but it fell far below the round `13` breakout and did not strengthen the case for another unchanged live deployment.
- Round `15`, completed on `2026-03-21`, scored `20.9269 pts`, rank `244/262`.
- Round `15` seed-level scores: `24.5169`, `20.3268`, `19.6928`, `21.8952`, `18.2027`.
- Interpretation of round `15`: the emergency frequency-fallback path was a severe regression and confirmed that fallback mode is only a safety valve, not a viable competitive deployment.
- Round `16`, completed on `2026-03-21`, scored `70.8183 pts`, rank `174/272`.
- Round `16` seed-level scores: `71.4041`, `70.6278`, `72.7148`, `68.5201`, `70.8244`.
- Interpretation of round `16`: restoring compatibility for the stable `feature_version=3` GBDT ensemble recovered the live path immediately after the round `15` collapse and brought the score back into the strong range.
- Round `17`, completed on `2026-03-21`, scored `64.8617 pts`, rank `193/283`.
- Round `17` seed-level scores: `65.8869`, `66.6285`, `64.6511`, `64.8430`, `62.2989`.
- Interpretation of round `17`: the restored stable `feature_version=3` live path held up again after round `16`; the result is below the strongest rounds (`13` and `16`) but solid enough that there is still no justification for replacing it with the failed `v4` candidates.
- Round `18`, completed on `2026-03-21`, scored `45.3033 pts`, rank `187/265`.
- Round `18` seed-level scores: `47.5590`, `42.4233`, `44.0190`, `51.5642`, `40.9508`.
- Interpretation of round `18`: this was a weak live result on the stable bundle, but it still does not justify deploying any of the new offline candidates because those candidates remain materially worse on the gate metric.
- Round `19`, completed on `2026-03-22`, scored `62.8855 pts`, rank `152/228`.
- Round `19` seed-level scores: `64.8612`, `61.6158`, `62.0661`, `62.0324`, `63.8520`.
- Interpretation of round `19`: the old stable bundle remained competitive enough to avoid panic, but the result was no longer strong enough to justify ignoring the benchmark-promoted candidate on recent-history evidence.
- Round `21`, completed on `2026-03-22`, scored `71.7469 pts`, rank `177/225`.
- Round `21` seed-level scores: `72.4649`, `74.0669`, `69.5372`, `73.7905`, `68.8751`.
- Interpretation of round `21`: the benchmark-promoted `gbdt_v3_retrain2` bundle validated positively in live play, beating round `19` on score by `8.8614` points and confirming that the benchmark-aware gate was directionally correct even though the raw report CV looked worse.
- Completed `sklearn_v4_tuned1` offline result: `cv_prequential_logloss=0.559430506272988`, `feature_version=4`, `training_examples=101250`.
- Interpretation of `sklearn_v4_tuned1`: more data and a full tuning pass were still not enough to beat the stable live bundle, so `v4` remains offline-only.
- Completed `gbdt_v3_retrain2` offline result: `cv_prequential_logloss=0.6615468842343689`, `feature_version=3`, `training_examples=135000`.
- Interpretation of `gbdt_v3_retrain2`: even the accelerated retrain on the proven feature space failed decisively, so more history alone is not enough to produce a better candidate.
- Three-round `Approach 4` summary: rounds `11`, `13`, and `14` scored `58.0097`, `78.2571`, and `55.8597`.
- Decision after the full `Approach 4` evaluation window: treat the approach as promising but not yet stable enough for a 4th unchanged live round; move to the already-defined offline bake-off and feature-expansion plan before the next model deployment.
- Active round now: round `22`, `round_id=a8be24e1-bd48-49bb-aa46-c5593da79f6f`.
- Coarse backend screen result: `sklearn-hgbt=0.7177`, `lightgbm=0.7256`, `xgboost=0.7181`, `catboost=0.7123`.
- Coarse screen interpretation: `CatBoost` is the current shortlist backend for the next full-tuning gate run, but no backend has yet earned replacement of the current tuned live bundle.
- `CatBoost v4` gate run result: `base_model_cv_prequential_logloss=0.7140`, `ensemble-tuned cv_prequential_logloss=0.6127`, `feature_version=4`.
- `CatBoost v4` interpretation: the new features plus mixer tuning helped `CatBoost` relative to its own coarse screen, but not nearly enough to justify backend replacement or live deployment.
- `sklearn-hgbt v4` gate run result: `base_model_cv_prequential_logloss=0.7225`, `ensemble-tuned cv_prequential_logloss=0.5676`, `feature_version=4`.
- `sklearn-hgbt v4` interpretation: `FEATURE_VERSION=4` is more promising on the current backend than on `CatBoost`, but the result is still above the current tuned live baseline, so feature expansion alone has not passed the deployment gate yet.

## Maintenance Rule

- After every completed live round, append the verified outcome here.
- After every major offline decision that changes the live strategy, append the decision and the reason here.
- Do not overwrite failed-hypothesis entries after the fact; add a new entry if later evidence changes the conclusion.
