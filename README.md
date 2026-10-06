# Loan Portfolio Dashboard v5

Pages:
1. Data Cleaning
2. Portfolio Summary
3. Performance Analysis
4. Workflow

Performance Analysis has only two panels:

## Panel 1 — Logistic Regression
- Outcome: Higher Risk = Watchlist or Non-Performing
- Predictors: Risk Rating, LTV, Tenor, Floating vs Fixed
- Displays one Odds Ratio chart with 95% confidence intervals
- Ends with one concise conclusion

## Panel 2 — Clustering
- K-means with Risk Rating, LTV, Tenor, and Log Exposure
- Uses four clusters for interpretability
- Displays one compact cluster summary table
- Displays one Higher-Risk Exposure by Cluster chart
- Ends with one concise conclusion

Run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run loan_portfolio_dashboard.py
```
