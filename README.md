# SPVC-FTS

Reproduction in progress of Chen L., Lu Q. J. (2026), “Forecasting of Fuzzy Time Series Based on Semiparametric Varying Coefficient Models”, *Mathematical Modeling and Its Applications*, 15(1), 10–20. DOI: 10.19943/j.2095-3070.jmmia.2026.01.02.

**Status: incomplete.** Only the preprocessing formulas and MSE/MAE are implemented; the two-stage model, bandwidth selection, MSM and the paper's experiments are not yet reproduced. No reported numerical result has been independently verified. / **复现进行中：目前仅实现预处理及 MSE、MAE；两阶段模型、带宽选择、MSM 和原文实验尚未完成。**

## Installation and verification

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`financial_returns(low, high)` returns centre and equal left/right spreads in natural-log units using the formulas on paper p. 15. The paper does not specify a multiplier in these formulas, while reported financial metrics should be checked against the original data and scaling.

`temperature_fuzzy(values)` implements the normal probability equations on p. 17 with population variance. If a data point has less than 0.05 probability in either tail, the equation has no finite solution; the function raises an error rather than silently inventing a workaround. No claim is made that this resolves the paper's data preparation.

`errors(actual, fitted)` evaluates p. 14 MSE and MAE as the mean of the sum of component errors, not the mean across all individual components.

## Reproduction issues

- The paper specifies Shenzhen index high/low values for 2023-02-24 to 2023-12-15, gold high/low values for 2023-07-19 to 2024-09-20, and Shanghai daily mean temperature for 2024-01-01 to 2024-09-30, but does not include the original series in the PDF. Exact providers, price conventions, and the series used after differencing need confirmation.
- The paper does not provide an explicit finite bandwidth search limit or all numerical optimization details. Leave-one-out handling of lagged inputs also needs to be specified and tested; full-series fitting must not be reported as genuine out-of-sample forecasting.
- Section 3.2 text says the final MAE is 0.6602, but Table 2 says 0.6620. Both values must be kept distinct pending verification.
- The model, comparison methods, plots, data provenance, and paper-to-code result tables remain to be implemented.
