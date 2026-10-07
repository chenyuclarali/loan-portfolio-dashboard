# Loan Portfolio Dashboard v6

Pages:
1. Data Cleaning
2. Portfolio Summary
3. Performance Analysis
4. Workflow

## Performance Analysis

### Panel 1 — Logistic Regression
- User-selectable factor bar (multiselect)
- Numeric predictors standardized
- Categorical predictors dummy-coded
- Odds-ratio dot plot with 95% confidence intervals
- Compact OR / CI / p-value table
- No dashboard-written conclusion

### Panel 2 — Clustering
- Fixed 4-cluster K-means
- Features: Risk Rating, LTV, Tenor, Log Exposure
- Table shows each cluster's key contributing factor
- Key factor = centroid dimension with largest absolute standardized deviation
- Cluster factor dot plot
- 100% colored stacked bar for Performing / Watchlist / Non-Performing exposure mix
- No dashboard-written conclusion

Run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run loan_portfolio_dashboard.py
```
