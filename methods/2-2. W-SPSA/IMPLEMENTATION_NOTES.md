# W-SPSA Implementation Notes

- baseline family:
  - weighted simultaneous perturbation stochastic approximation
- method name used in this project:
  - `W-SPSA`
- current adaptation:
  - two-sided SPSA updates under the shared sequential step-lock protocol
  - per-link measurement-error vector instead of a single scalar mismatch
  - OD-specific correlation weights built from realized current-step OD-to-link influence
  - weighted gradient estimate following the Lu et al. idea of replacing scalar aggregation with correlation-weighted measurement errors
  - zero warm start under the shared optimizer protocol unless `warm_start` is explicitly changed
  - the gain update is applied in raw OD units; the finite-difference denominator already contains the raw OD perturbation scale
  - uses link-flow observations only, with no stored OD-demand labels or explicit OD prior injection

Because the project protocol is myopic step-lock calibration, this adaptation only uses the current measurement row rather than the full multi-interval objective used in the original paper. The weighted SPSA core, however, now follows the paper's main mechanism more closely than the earlier heuristic version.
