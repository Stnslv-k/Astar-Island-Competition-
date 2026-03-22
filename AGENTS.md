# Astar Island Agent Notes

## Hard Requirements

- Live play must use the API simulator via `/astar-island/simulate` with a `15x15` viewport.
- Every live run must submit probability predictions for each seed in the active round.
- Do not switch viewport size during experiments unless a separate branch explicitly tests that variable.
- Always update `HISTORY.md` after every completed live round and after every major offline decision that changes the live strategy.
- If the serialized GBDT bundle is runtime-incompatible with the current feature space, do not abort the live round; auto-fallback to the frequency prior and continue the run.

## Evaluation Policy

- Run each alternative approach for exactly 3 rounds before deciding whether to continue.
- Exception: if an approach shows a clearly negative result early, such as a severe score collapse or rank deterioration, stop it immediately and move to the next approach.
- After each round:
  - record round score, rank, total teams, and seed-level scores when available
  - compare against the latest heuristic baseline
  - note whether the direction of improvement is clear, flat, or negative
- After all 3 rounds for an approach:
  - assess top-entry trajectory, not just average score
  - if the prospects for reaching the top are still unclear, move to the next approach
  - do not spend a 4th round on the same approach unless the top-entry case is clearly strengthening

## Ordered Approach Ladder

### Approach 1: Stronger Active Sampling Heuristics

Goal:
- maximize information gain from the last 5 queries without changing the basic Bayesian baseline

Work plan:
1. keep full 45-query cover on `15x15`
2. score refinement windows by uncertainty, frontier density, settlement context, and redundancy penalty
3. choose refinement windows greedily, one at a time, recomputing after each simulate result
4. log raw payloads and selected-window rationale for offline analysis

Success signal after 3 rounds:
- better rank than the heuristic baseline and a stable upward score trend

Round status:
- stopped early after round 6 because the first live result was clearly negative versus the heuristic baseline
- round 6 result: `20.8444 pts`, rank `154/186`
- latest heuristic baseline for comparison: round 5 `44.2909 pts`, rank `102/144`

### Approach 2: Graph / CRF Style Spatial Smoothing

Goal:
- improve local consistency without erasing true sharp transitions

Work plan:
1. add edge-aware smoothing over the posterior grid
2. make smoothing depend on terrain-change likelihood and settlement anchors
3. compare against plain neighbor smoothing under the same `15x15` live policy

Success signal after 3 rounds:
- seed-level gains on noisy rounds, especially where current outputs look too speckled

Round status:
- active
- stopped early after round 7 because the first live result was clearly negative
- round 7 result: `15.2289 pts`, rank `177/199`
- comparison points:
- heuristic baseline round 5: `44.2909 pts`, rank `102/144`
- stronger active sampling round 6: `20.8444 pts`, rank `154/186`

### Approach 3: Lightweight Learned Posterior Model

Goal:
- predict per-cell posterior better than hand-written priors using accumulated run artifacts

Work plan:
1. build a training dataset from saved `simulate_payloads.json`
2. train a small model on local neighborhood, initial terrain, settlement features, and repeated samples
3. use the model as a posterior proposal, then blend with empirical counts
4. add a coverage-aware learned gate for empirical posterior vs learned posterior
5. add bucketed calibration by observation count before submit
6. keep the full `45`-query cover on `15x15`; do not replace full coverage with an all-budget dynamic sampler
7. expand learned features with wider local context, settlement-distance features, and edge/context signals
8. prefer confidence-aware blending between empirical posterior and learned prior before adding any heavy spatial post-processing
9. use the last `5` queries only for adaptive refinement; improve refinement quality without sacrificing full-map observation coverage

Success signal after 3 rounds:
- clear score lift over heuristic-only methods and less calibration collapse on difficult seeds

Round status:
- round 8 result: `55.5983 pts`, rank `148/214`
- round 9 result: `71.0297 pts`, rank `130/221`
- round 10 result: `43.9795 pts`, rank `175/238`
- the learned-posterior stack now includes a trained GBDT prior, a coverage-aware learned gate, and bucketed calibration
- a tuned GBDT prior bundle is now trained from historical `simulate_payloads.json` and stored at `models/gbdt_prior.joblib`
- approach outcome after 3 rounds: mixed and not strong enough for a 4th live round
- decision: move to Approach 4 for the next live evaluation

### Approach 4: Ensemble

Goal:
- combine complementary models instead of betting on one posterior builder

Work plan:
1. keep separate predictors for empirical counts, graph smoother, and learned posterior
2. weight them by confidence or validation performance
3. run the same 3-round evaluation gate
4. start with a coverage-aware ensemble of empirical posterior plus learned prior
5. add only light smoothing if it improves offline diagnostics; do not reintroduce aggressive graph smoothing by default
6. preserve the hard requirement of full `15x15` live coverage before ensemble refinements are evaluated
7. for the last `5` queries only, prefer score-aware active learning by expected information gain; do not replace the full `45`-query cover
8. treat sensor-placement logic as a redundancy-aware window selection problem, not as a Gaussian-process problem
9. only consider weak pairwise regularization on low-confidence cells after the ensemble path is validated live
10. do not spend time on cellular-automata style modeling unless we later obtain explicit trajectory supervision

Success signal after 3 rounds:
- higher floor across seeds and fewer catastrophic weak seeds

Round status:
- round 11 result: `58.0097 pts`, rank `123/171`
- round 11 seed scores: `61.9860`, `58.2812`, `55.0643`, `57.5882`, `57.1286`
- round 13 result: `78.2571 pts`, rank `114/186`
- round 13 seed scores: `78.4480`, `77.3383`, `78.1995`, `77.5400`, `79.7598`
- round 14 result: `55.8597 pts`, rank `146/244`
- round 14 seed scores: `53.9839`, `58.1836`, `56.4144`, `56.4328`, `54.2840`
- direction after round 14: mixed; round 13 remains the strongest result, but the 3-round trajectory is not clearly strengthening enough to justify a 4th unchanged live round
- round 12 was missed
- evaluation decision: stop unchanged live deployments of Approach 4 here and move to the offline-first improvement plan below
- a safe ensemble of empirical posterior plus learned posterior is now implemented and tuned offline
- latest offline bundle uses `posterior_mode=ensemble` with `cv_prequential_logloss=0.5350`
- this improved over the previous learned-gate stack (`0.5541`) on historical prequential validation
- refinement selection now uses an expected-information-gain approximation plus overlap penalty for the last `5` windows
- do not deploy graph-heavy smoothing by default; require clear offline evidence first

Next offline change plan after the Approach 4 evaluation window closes:
1. run an offline backend bake-off: `LightGBM`, `XGBoost`, and `CatBoost` against the current `sklearn-hgbt` bundle
2. gate condition for backend replacement: require a clear and stable leave-one-round-out prequential logloss improvement over `0.5350`, not a symbolic improvement
3. on the winning backend only, add feature expansion with `radius=3,4` context plus coastal / water proximity features
4. gate condition for feature expansion: do not deploy unless it preserves or improves the post-bake-off offline result
5. then run `Optuna` with `50+` trials on the surviving stack
6. live deployment is allowed only after all prior gate conditions pass; do not mix these offline experiments into the active live track prematurely

Current offline bake-off status:
- `libomp` is installed, so `LightGBM`, `XGBoost`, and `CatBoost` are now available in the project `.venv`
- the first coarse screen (`optuna=0`, `mixer=0`) produced:
- `sklearn-hgbt`: `0.7177`
- `lightgbm`: `0.7256`
- `xgboost`: `0.7181`
- `catboost`: `0.7123`
- coarse-screen winner: `CatBoost`
- coarse-screen decision: shortlist `CatBoost` for the next full-tuning gate run, but do not replace the current live backend yet
- first `CatBoost + FEATURE_VERSION=4` gate run:
- `base_model_cv_prequential_logloss`: `0.7140`
- `ensemble-tuned cv_prequential_logloss`: `0.6127`
- gate decision after the first `CatBoost v4` run: still well above the current live baseline (`0.5350`), so `CatBoost` has not earned backend replacement
- first `sklearn-hgbt + FEATURE_VERSION=4` gate run:
- `base_model_cv_prequential_logloss`: `0.7225`
- `ensemble-tuned cv_prequential_logloss`: `0.5676`
- comparison to `CatBoost v4`: current backend is materially better on the new feature set
- current offline decision: stop pursuing backend replacement for now and continue the next optimization cycle on `sklearn-hgbt` with `FEATURE_VERSION=4`
- round `15` emergency fallback result: `20.9269 pts`, rank `244/262`
- round `15` decision: frequency fallback is an emergency-only path and must not replace the stable GBDT deployment strategy
- operational compatibility decision: keep old stable bundles runnable by using the bundle's own `feature_version` during prediction while `FEATURE_VERSION=4` tuning continues offline
- round `16` restored-stable-bundle result: `70.8183 pts`, rank `174/272`
- round `16` seed scores: `71.4041`, `70.6278`, `72.7148`, `68.5201`, `70.8244`
- round `16` decision: the restored `feature_version=3` stable bundle remains the live path until a new candidate actually beats it offline
- completed `sklearn_v4_tuned1` result: `cv_prequential_logloss=0.559430506272988`
- offline gate decision after `sklearn_v4_tuned1`: failed; do not deploy this bundle to live
- round `17` orchestration: launch a full `feature_version=3` retrain as a one-shot job now, and schedule a one-shot deadline submit at `+150` minutes
- round `17` deployment rule: if the new `v3` report beats the stable report by a real margin before the deadline, it may submit; otherwise the deadline job submits the stable bundle automatically
- round `17` live result: `64.8617 pts`, rank `193/283`
- round `17` seed scores: `65.8869`, `66.6285`, `64.6511`, `64.8430`, `62.2989`
- round `17` operational conclusion: the restored stable `feature_version=3` bundle remains the live path; no newer offline candidate has earned deployment
- round `18` live result: `45.3033 pts`, rank `187/265`
- round `18` seed scores: `47.5590`, `42.4233`, `44.0190`, `51.5642`, `40.9508`
- round `18` operational conclusion: the stable bundle had a weak round, but it still remains the best live option because the fresh offline candidates are worse
- round `19` live result: `62.8855 pts`, rank `152/228`
- round `19` seed scores: `64.8612`, `61.6158`, `62.0661`, `62.0324`, `63.8520`
- round `19` operational conclusion: the old stable bundle remained workable, but not strong enough to outweigh the recent-history benchmark that favored the new candidate
- offline retrain optimization: historical feature matrices and observation maps are now cached per run in `gbdt.py` to reduce repeated work during `v3` tuning and cross-validation
- completed `gbdt_v3_retrain2` result: `cv_prequential_logloss=0.6615468842343689`, `feature_version=3`, `training_examples=135000`
- offline gate correction: raw `cv_prequential_logloss` values from models trained and evaluated on different historical round sets are not directly comparable and must not be used alone for deployment decisions
- benchmark-aware gate: compare candidate and stable bundles head-to-head on the same recent historical rounds before promotion
- benchmark result for `gbdt_v3_retrain2` vs previous stable bundle on rounds `14-19`: candidate `0.6778`, stable `0.7142`, delta `-0.0364`
- deployment decision after benchmark: promote `gbdt_v3_retrain2` as the live bundle for the next round, while preserving the old stable bundle as a backup at `models/gbdt_prior_stable_backup_20260322T031440Z.joblib`
- round `21` live result on the promoted candidate: `71.7469 pts`, rank `177/225`
- round `21` seed scores: `72.4649`, `74.0669`, `69.5372`, `73.7905`, `68.8751`
- round `21` operational conclusion: the benchmark-promoted candidate validated positively in live play and should remain the live bundle unless a newer candidate beats it on the same benchmark protocol
