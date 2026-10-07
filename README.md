Loan Portfolio Cleaning & Review Dashboard
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

<img width="1460" height="943" alt="image" src="https://github.com/user-attachments/assets/605159e2-3398-4045-a667-fab84c0f0024" />
<img width="1462" height="945" alt="image" src="https://github.com/user-attachments/assets/01d7ebdb-8956-44b1-9949-8b6169d08f99" />
<img width="1462" height="867" alt="image" src="https://github.com/user-attachments/assets/718a7db6-e8b8-4956-9a5c-1db139c9923f" />
<img width="1458" height="880" alt="image" src="https://github.com/user-attachments/assets/ff0795ce-a726-45c8-a38f-8559f111c57c" />
<img width="1457" height="765" alt="image" src="https://github.com/user-attachments/assets/bbb5abcc-dc58-4f77-a1a1-c496f92d21da" />
<img width="1461" height="883" alt="image" src="https://github.com/user-attachments/assets/328906e1-122b-4d39-8485-3736b9734d31" />



