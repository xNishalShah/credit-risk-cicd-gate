# Quality gate and fairness defense

**Recommended rule:** accept a candidate only when PR-AUC is no more than 0.01 below the approved baseline and recall at 50 percent precision is at least 0.45. PR-AUC is the primary signal because only about 19 percent of customers are high risk; accuracy could reward a model that mostly predicts the majority class.

| Candidate rule | Expected behavior | Trade-off |
|---|---|---|
| No PR-AUC drop | Conservative | Rejects harmless sampling variation. |
| PR-AUC drop no larger than 0.01 plus recall floor | Recommended | Allows limited variance but preserves useful risk capture. |
| PR-AUC drop no larger than 0.03 | Permissive | Risks accepting a material degradation. |

Before registration, report false-negative rate, false-positive rate, PR-AUC, and support by `Income_Bracket`. The project should not assert fairness until that output is generated from the final registered model. We would revisit this rule if three-seed comparisons show normal PR-AUC variation above 0.01, or if an income group has meaningfully worse false-negative rate with enough samples to make that result stable.
