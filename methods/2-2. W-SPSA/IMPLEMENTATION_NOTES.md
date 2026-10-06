# W-SPSA implementation

This implementation applies weighted simultaneous perturbation stochastic approximation to sequential OD demand estimation.

- Candidate demands are evaluated from copies of the current simulator state.
- OD-specific weights are computed from current-step OD-to-link propagation and link-flow errors.
- Two-sided perturbations estimate the search direction using the realised, bounded demand perturbation.
- The gain update is applied in vehicle units.
- Each update uses the current observed link-flow row, with earlier demand decisions fixed.

Method settings are defined in `params/test_params.json`.

Reference: L. Lu, Y. Xu, C. Antoniou, and M. Ben-Akiva (2015), "An enhanced SPSA algorithm for the calibration of dynamic traffic assignment models," Transportation Research Part C, 51, 149–166.
