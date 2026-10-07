Loan Portfolio Cleaning & Credit Review Dashboard
=================================================

Dashboard Instructions
------------------
https://loan-portfolio-dashboard.streamlit.app/
  1. Upload the case study excel file.
  2. Review flagged records in Manual Review and update Include in Analysis as needed.
  3. Use Portfolio Summary and Performance Analysis to review results.
  4. Click Download Full Results Excel to export the current analysis.
  5. Results update based on the current Include in Analysis selections.

Python Instructions
------------------
Sidebar navigation
Case Study
  - Data Cleaning
  - Portfolio Summary
  - Workflow

Files required in the same folder:
  1. loan_portfolio_dashboard_with_workflow.py
  2. clean_loan_portfolio_FINAL.py
  3. requirements.txt

Install dependencies:
  python -m pip install -r requirements.txt

Run the dashboard:
  python -m streamlit run loan_portfolio_dashboard_with_workflow.py

Data Cleaning
- Upload the raw case-study Excel workbook.
- Select Loan_Portfolio_Data.
- Run automated cleaning.
- Review Cleaned Data and the validation log.
- Edit records with Manual Review Required = TRUE.
- Use "Verified - keep as-is" only after confirming an unusual value is correct.
- Apply revisions and rerun checks.
- Download the current cleaned workbook.

Portfolio Summary
- Portfolio composition by geography, industry, borrower type, and facility type.
- Fixed vs. floating exposure.
- Risk rating and loan-status distributions.
- Borrower/category concentrations.
- Assumptions and data-treatment table.

Workflow
- Interactive workflow chart showing the full cleaning process.
- Green nodes = deterministic corrections.
- Orange nodes = validation / flag-only checks.
- Red nodes = manual review and revalidation loop.
- RapidFuzz is shown explicitly as a flag-only secondary QA control.
- Current open-review metrics appear when a workbook has been loaded.

Notes
-----
- Summary percentages are exposure-weighted using Outstanding Balance.
- Open manual-review rows are included in the base-case summary by default.
- A toggle provides a sensitivity view excluding open review rows.
