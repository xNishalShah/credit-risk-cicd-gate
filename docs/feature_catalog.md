# Feature catalog

| Feature | Formula | Reason |
|---|---|---|
| Credit Utilization | Current Balance / Credit Limit | High outstanding balance relative to approved credit can signal repayment stress. |
| High Utilization Flag | Utilization at or above YAML cutoff | Makes a potentially non-linear utilization risk boundary explicit. |
| Payment Delay Band | On time, short delay, or long delay from YAML cutoff | Separates habitual late payment from small operational delays. |
| Monthly Spend to Limit | Monthly Spend / Credit Limit | Sustained spending pressure relative to capacity can signal financial strain. |

`Customer_ID` and the target are excluded from predictors. A zero credit limit is converted to a safe zero ratio rather than infinity. Missing values are imputed inside the fitted pipeline, avoiding leakage from the test set.
