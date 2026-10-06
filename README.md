# Loan Portfolio Dashboard v4

## Pages

1. Data Cleaning
2. Portfolio Summary
3. Statistical Analysis
4. Workflow

## Statistical Analysis

### Logistic Regression

Outcome:
- `Higher Risk = 1`: Watchlist or Non-Performing
- `Higher Risk = 0`: Performing

Two views are provided:

1. **Core inferential logistic model**
   - Risk Rating
   - LTV
   - Tenor
   - Floating vs Fixed
   - Odds ratio
   - 95% confidence interval
   - p-value

   Numeric predictors are standardized, so the odds ratio represents a 1-SD increase.

2. **Regularized contributor ranking**
   - L2 logistic regression
   - selectable predictors:
     Risk Rating, LTV, Tenor, Rate Type, Region, Industry,
     Borrower Type, Facility Type, Relationship Manager
   - cross-validated ROC AUC
   - permutation importance by original variable
   - detailed regularized coefficients / model-based odds ratios

   Industry and Relationship Manager are not selected by default because their
   high cardinality can overfit a small sample.

### Clustering

K-means uses numeric risk characteristics only:
- Risk Rating
- LTV
- Tenor
- Log Exposure

Steps:
- median imputation
- standardization
- silhouette-score comparison for k
- K-means segmentation
- cluster risk/performance profile
- exposure-weighted Loan Status mix
- PCA used only as a 2D visualization of cluster space

Loan Status is NOT used to form clusters.

## Interpretation

All statistical outputs are exploratory associations / segmentation, not causal estimates.
Portfolio Summary continues to use only `Include in Analysis = TRUE`.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run loan_portfolio_dashboard.py
```

## Model interpretation guardrails

- Cross-validated ROC AUC is displayed to show whether the logistic model generalizes.
- Permutation importance can be near zero or negative; this indicates weak or unstable
  predictive contribution rather than evidence that a factor is protective.
- If ROC AUC is near or below 0.5, the dashboard should be interpreted as showing that
  the selected predictors do not provide stable predictive discrimination in this sample.
- Clusters are unsupervised descriptive segments, not credit grades or causal groups.
