## 10a — Data Card / Feature Dictionary

| Feature | Semantic type | % missing | Univariate notes | Bivariate signal (stat / effect size) | MI | Multicoll. flag | Leakage flag | Recommended treatment | Encoding | Scaling | Keep / Drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ncat | target (severity/count) | 0% | extreme zero inflation, heavy tail | target variable | — | — | — | log1p target candidate | none | none | KEEP |
| mod140 | numeric underwriting score | low | heavy right-skew | strongest Pearson/Spearman signal | low MI but high tree importance | low | POSSIBLE | leakage review + log1p | numeric | robust scaler | KEEP (review) |
| mod140_year | temporal / ordinal | low | temporal drift observed | strong tree importance | low | low | POSSIBLE | treat as ordinal/time feature | ordinal | none | KEEP |
| risk_score1 | numeric risk score | low | mild skew | weak linear but nonlinear importance | low | low | possible proxy | preserve + monitor | numeric | robust scaler | KEEP |
| property_value_800 | numeric property valuation | ~94% | heavy-tail monetary variable | weak linear signal | strongest MI feature | low | no | median impute + log1p | numeric | robust scaler | KEEP |
| mortgage_available_he_800 | binary / enrichment flag | ~94% | sparse / mostly missing | weak standalone effect | low | low | no | preserve missingness | binary | none | KEEP |
| est_hh_income_200_prev | binary / enrichment flag | ~94% | sparse | missingness informative | low | low | no | missing category retention | binary | none | KEEP |
| wealth_rating_07 | ordinal wealth rating | ~94% | skewed socioeconomic variable | weak linear signal | strong MI | low | fairness proxy risk | preserve + monitor fairness | ordinal | robust scaler | KEEP |
| length_of_residence | numeric / ordinal | moderate | low variance | weak signal | low | low | no | median imputation | numeric | standard scaler | KEEP |
| property_has_pool | binary property feature | ~94% | sparse binary feature | weak signal | low | low | no | preserve missingness | binary | none | KEEP |
| vendor2_fire_raw | vendor severity feature | moderate | moderate skew | nonlinear monotonic pattern | low | moderate vendor cluster | no | monitor overlap | numeric | robust scaler | KEEP |
| vendor2_eqpm | vendor severity feature | moderate | moderate skew | moderate interaction structure | low | moderate vendor cluster | no | possible feature consolidation | numeric | robust scaler | KEEP |
| vendor3_pipemlc | vendor severity feature | low | moderate skew | monotonic nonlinear signal | low | moderate vendor cluster | no | preserve for nonlinear models | numeric | robust scaler | KEEP |
| vendor3_compemlc | vendor severity feature | low | moderate skew | weak linear but useful tree signal | low | moderate vendor cluster | no | preserve for ensemble models | numeric | robust scaler | KEEP |
| state | categorical geographic | low | low cardinality | significant ANOVA effect | low | low | proxy risk possible | rare-level grouping if needed | one-hot | none | KEEP |
| dwelling_type | categorical property type | low | low cardinality | weak ANOVA effect | near-zero MI | low | no | one-hot encoding | one-hot | none | KEEP |
| homeowner_renter | categorical occupancy | low | low cardinality | weak ANOVA effect | near-zero MI | low | no | one-hot encoding | one-hot | none | KEEP |
| record_id | identifier | 0% | unique identifier | no predictive meaning | none | none | YES | remove before modeling | none | none | DROP |
| qpid | identifier | high uniqueness | identifier-like | no meaningful signal | none | none | YES | remove before modeling | none | none | DROP |

---

### 10a — Data Card / Feature Dictionary

One row per feature. Internally consistent with Stages 2–9.

| Feature | Semantic type | % missing | Univariate notes | Bivariate signal (stat, effect size) | MI | Multicoll. flag | Leakage flag | Recommended treatment | Encoding | Scaling | Keep / Drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |
### 10b — Train / Validation / Test Split Recommendation

Choose **one** and justify against this dataset's specific risks:
- ☑ Stratified (by target)
- ☑ Time-based (cutoff: vendor_year)
- ☐ Random
- ☐ Grouped
- ☐ Nested

**Fractions:**  
train 70 / val 15 / test 15

**Justification:**

- target (`ncat`) is extremely zero-inflated (~95% zeros)
- stratification required to preserve catastrophic-loss frequency across splits
- temporal drift observed across `vendor_year`
- average NCAT changed meaningfully between years
- random split may leak future underwriting patterns into training
- time-aware validation better reflects real insurance deployment setting
- catastrophic-loss observations are sparse and unstable
- validation/test sets should preserve tail-event representation
- leakage-sensitive variables (`mod140`, `mod140_year`) should be evaluated carefully before final modeling
## 10c — Preprocessing Pipeline Sketch

| Feature | Imputation | Transform | Encoding | Scaling | Notes (fit-on-train-only?) |
|---|---|---|---|---|---|
| mod140 | median | log1p candidate | none | robust scaler | scaler fit on train only |
| mod140_year | none | ordinal handling | ordinal | none | preserve temporal order |
| risk_score1 | median | none | none | robust scaler | fit scaler on train only |
| property_value_800 | median + missing flag | log1p | none | robust scaler | fit transform on train only |
| mortgage_available_he_800 | explicit missing category | none | binary | none | preserve informative missingness |
| est_hh_income_200_prev | explicit missing category | none | binary | none | preserve sparse missing pattern |
| wealth_rating_07 | median + missing flag | ordinal preserve | ordinal | robust scaler | fairness review recommended |
| length_of_residence | median | optional clipping | none | standard scaler | fit scaler on train only |
| property_has_pool | explicit missing category | none | binary | none | sparse binary feature |
| vendor2_fire_raw | median | optional log transform | none | robust scaler | vendor overlap monitoring |
| vendor2_eqpm | median | optional log transform | none | robust scaler | possible consolidation candidate |
| vendor3_pipemlc | median | optional log transform | none | robust scaler | nonlinear interaction candidate |
| vendor3_compemlc | median | optional log transform | none | robust scaler | preserve for tree models |
| state | most frequent | rare-level grouping | one-hot | none | grouping fit on train only |
| dwelling_type | most frequent | none | one-hot | none | low-cardinality categorical |
| homeowner_renter | most frequent | none | one-hot | none | low-cardinality categorical |
| ncat (target) | none | log1p target candidate | none | none | transform fit using train only |
| record_id | none | none | none | none | remove before modeling |
| qpid | none | none | none | none | remove before modeling |

---
### 10c — Preprocessing Pipeline Sketch

Map each kept feature to its transformer chain. Specific enough that someone else could implement it as a `sklearn.pipeline.Pipeline` / `ColumnTransformer`. Mark fit-on-train-only operations clearly.

| Feature | Imputation | Transform | Encoding | Scaling | Notes (fit-on-train-only?) |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
### 10d — Top-10 Candidate Predictors + Known Risks

#### Top 10 (with evidence from Stages 7 + 8 + 9):

1. `mod140`
   - strongest linear + tree-based importance signal
   - possible underwriting composite score / leakage risk

2. `mod140_year`
   - strong temporal underwriting signal
   - may capture pricing regime shifts

3. `risk_score1`
   - weak linear effect but consistent nonlinear importance

4. `vendor3_pipemlc`
   - nonlinear vendor severity relationship detected

5. `vendor2_eqpm`
   - moderate vendor interaction signal

6. `vendor3_compemlc`
   - moderate monotonic relationship with severity

7. `state`
   - geographic underwriting / catastrophe exposure signal

8. `wealth_rating_07`
   - high MI score suggests nonlinear socioeconomic signal

9. `property_value_800`
   - nonlinear property exposure relationship

10. `length_of_residence`
    - weak standalone effect but potential interaction feature

---

#### Known Risks register:

- **Leakage suspects retained or removed:**
  - `mod140`
  - `mod140_year`
  - possible internally engineered underwriting scores
  - require business validation before production use

- **Multicollinearity clusters:**
  - vendor2 / vendor3 severity-related feature families
  - moderate correlation structure observed
  - possible consolidation candidate

- **Missingness assumptions:**
  - missingness likely informative
  - many enrichment variables contain dominant missing categories
  - underwriting missingness may itself encode risk

- **Drift / temporal-stability concerns:**
  - target drift observed across `vendor_year`
  - catastrophic-loss severity varies over time
  - temporal validation recommended

- **Fairness-sensitive features and proxies:**
  - wealth / property valuation variables
  - geographic variables (`state`, ZIP-related variables)
  - socioeconomic proxy risk possible

- **Other:**
  - extreme zero inflation (~95%)
  - catastrophic-loss tail dominates modeling behavior
  - nonlinear + interaction-heavy structure
  - robust / outlier-aware modeling likely required
---