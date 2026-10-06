#!/usr/bin/env python3
"""
Loan portfolio data-cleaning script.

Usage:
    python clean_loan_portfolio.py input.xlsx output.xlsx

Example:
    python clean_loan_portfolio.py "Data Cleaning and Analysis Case Study .xlsx" "Loan_Portfolio_Cleaned.xlsx"

Dependencies:
    pip install pandas openpyxl rapidfuzz

Design principle:
    - One function per major data-quality issue.
    - Correct deterministic issues.
    - Flag uncertain issues rather than guessing.
    - Preserve an audit trail in a Data_Quality_Log sheet.
    - Separate resolved corrections from unresolved items requiring manual review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


DATA_SHEET = "Loan_Portfolio_Data"


# ---------------------------------------------------------------------------
# Helper: add a row-level or portfolio-level issue to the audit log
# ---------------------------------------------------------------------------
def add_issue(
    issues: list[dict],
    issue_type: str,
    description: str,
    action: str,
    loan_id=None,
    original_value=None,
    cleaned_value=None,
) -> None:
    issues.append(
        {
            "Loan ID": loan_id,
            "Issue Type": issue_type,
            "Description": description,
            "Original Value": original_value,
            "Cleaned Value": cleaned_value,
            "Action / Assumption": action,
        }
    )


# ---------------------------------------------------------------------------
# 1. File / row-count validation
# ---------------------------------------------------------------------------
def validate_row_count(
    df: pd.DataFrame,
    issues: list[dict],
    expected_rows: int = 100,
) -> pd.DataFrame:
    """Flag if the extract contains a different number of records than expected."""
    actual_rows = len(df)
    if actual_rows != expected_rows:
        add_issue(
            issues,
            issue_type="Record Count",
            description=f"Instructions state {expected_rows} records, but extract contains {actual_rows}.",
            action="Retained all available records; source-system reconciliation required.",
            original_value=actual_rows,
            cleaned_value=actual_rows,
        )
    return df


# ---------------------------------------------------------------------------
# 2. Text cleanup
# ---------------------------------------------------------------------------
def clean_text_fields(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """Trim leading/trailing whitespace and collapse repeated internal spaces."""
    df = df.copy()

    text_columns = [
        "Loan ID",
        "Borrower Name",
        "Country",
        "Region",
        "Industry",
        "Borrower Type",
        "Facility Type",
        "Interest Rate Type",
        "Base Rate Index",
        "Loan Status",
        "Relationship Manager",
    ]

    for col in text_columns:
        if col not in df.columns:
            continue

        original = df[col].copy()
        cleaned = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        df[col] = cleaned

        changed = original.notna() & (original.astype("string") != cleaned)
        for idx in df.index[changed]:
            add_issue(
                issues,
                issue_type="Text Formatting",
                description=f"Whitespace standardized in {col}.",
                action="Trimmed leading/trailing whitespace and collapsed repeated spaces.",
                loan_id=df.at[idx, "Loan ID"],
                original_value=original.at[idx],
                cleaned_value=df.at[idx, col],
            )

    return df


# ---------------------------------------------------------------------------
# 3. Categorical standardization
# ---------------------------------------------------------------------------
def standardize_categories(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """Map obvious spelling/case variants to one canonical category."""
    df = df.copy()

    mappings = {
        "Country": {
            "USA": "United States",
            "U.S.A.": "United States",
            "US": "United States",
            "UAE": "United Arab Emirates",
            "UK": "United Kingdom",
        },
        "Borrower Type": {
            "corporate": "Corporate",
            "Corp.": "Corporate",
            "sme": "SME",
            "Small Business": "SME",
        },
        "Loan Status": {
            "performing": "Performing",
            "Perfroming": "Performing",
            "watch list": "Watchlist",
            "Non Performing": "Non-Performing",
            "DEFAULT": "Non-Performing",
        },
    }

    for col, mapping in mappings.items():
        if col not in df.columns:
            continue

        original = df[col].copy()
        df[col] = df[col].replace(mapping)

        changed = original.notna() & (original.astype("string") != df[col].astype("string"))
        for idx in df.index[changed]:
            add_issue(
                issues,
                issue_type="Category Standardization",
                description=f"Standardized {col}.",
                action="Mapped obvious spelling/case/abbreviation variant to canonical category.",
                loan_id=df.at[idx, "Loan ID"],
                original_value=original.at[idx],
                cleaned_value=df.at[idx, col],
            )

    return df


# ---------------------------------------------------------------------------
# 4. Outstanding-balance cleanup
# ---------------------------------------------------------------------------
def clean_outstanding_balance(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Convert balance to numeric.
    Negative balances are treated as sign-entry errors for this case study
    and converted to absolute values, with the assumption explicitly logged.
    """
    df = df.copy()

    col = "Outstanding Balance"
    df[col] = pd.to_numeric(df[col], errors="coerce")

    negative_mask = df[col] < 0
    for idx in df.index[negative_mask]:
        original = df.at[idx, col]
        cleaned = abs(original)
        df.at[idx, col] = cleaned

        add_issue(
            issues,
            issue_type="Outstanding Balance",
            description="Negative outstanding loan balance.",
            action=(
                "Assumed a sign-entry error and used absolute value for analysis; "
                "verify against source system before Credit Committee use."
            ),
            loan_id=df.at[idx, "Loan ID"],
            original_value=original,
            cleaned_value=cleaned,
        )

    return df


# ---------------------------------------------------------------------------
# 5. Interest-rate consistency
# ---------------------------------------------------------------------------
def clean_interest_rate_fields(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Enforce the case-study field logic:
      - Fixed loans use Fixed Rate (%).
      - Floating loans use Base Rate Index + Margin (bps).
    """
    df = df.copy()

    # Parse Fixed Rate (%), including values such as "525bps".
    fixed_col = "Fixed Rate (%)"
    original_fixed = df[fixed_col].copy()

    def parse_fixed_rate(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, str):
            text = value.strip().lower()
            if text.endswith("bps"):
                number = pd.to_numeric(text[:-3].strip(), errors="coerce")
                return number / 100 if pd.notna(number) else np.nan

        return pd.to_numeric(value, errors="coerce")

    df[fixed_col] = df[fixed_col].apply(parse_fixed_rate)

    converted_bps = (
        original_fixed.astype("string").str.strip().str.lower().str.endswith("bps", na=False)
    )
    for idx in df.index[converted_bps]:
        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Fixed-rate value stored in basis points.",
            action="Converted basis points to percentage points (bps / 100).",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original_fixed.at[idx],
            cleaned_value=df.at[idx, fixed_col],
        )

    rate_type = df["Interest Rate Type"].astype("string").str.lower()

    # Floating loans should not have Fixed Rate (%).
    floating_with_fixed = (rate_type == "floating") & df[fixed_col].notna()
    for idx in df.index[floating_with_fixed]:
        original = df.at[idx, fixed_col]
        df.at[idx, fixed_col] = np.nan

        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Floating-rate loan had a value in Fixed Rate (%).",
            action="Cleared Fixed Rate (%) because the field is not applicable to floating loans.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original,
            cleaned_value=np.nan,
        )

    # Fixed loans should not have Base Rate Index or Margin (bps).
    fixed_mask = rate_type == "fixed"

    fixed_with_base = fixed_mask & df["Base Rate Index"].notna()
    for idx in df.index[fixed_with_base]:
        original = df.at[idx, "Base Rate Index"]
        df.at[idx, "Base Rate Index"] = pd.NA

        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Fixed-rate loan had Base Rate Index populated.",
            action="Cleared Base Rate Index because it is not applicable to fixed-rate loans.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original,
            cleaned_value=None,
        )

    df["Margin (bps)"] = pd.to_numeric(df["Margin (bps)"], errors="coerce")

    fixed_with_margin = fixed_mask & df["Margin (bps)"].notna()
    for idx in df.index[fixed_with_margin]:
        original = df.at[idx, "Margin (bps)"]
        df.at[idx, "Margin (bps)"] = np.nan

        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Fixed-rate loan had Margin (bps) populated.",
            action="Cleared Margin (bps) because it is not applicable to fixed-rate loans.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original,
            cleaned_value=np.nan,
        )

    # Missing required fields: flag, do not impute.
    floating_mask = rate_type == "floating"

    missing_base = floating_mask & df["Base Rate Index"].isna()
    for idx in df.index[missing_base]:
        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Floating-rate loan is missing Base Rate Index.",
            action="Left missing; cannot infer reliably without source-system data.",
            loan_id=df.at[idx, "Loan ID"],
        )

    missing_margin = floating_mask & df["Margin (bps)"].isna()
    for idx in df.index[missing_margin]:
        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Floating-rate loan is missing Margin (bps).",
            action="Left missing; cannot infer reliably without source-system data.",
            loan_id=df.at[idx, "Loan ID"],
        )

    missing_fixed = fixed_mask & df[fixed_col].isna()
    for idx in df.index[missing_fixed]:
        add_issue(
            issues,
            issue_type="Interest Rate",
            description="Fixed-rate loan is missing Fixed Rate (%).",
            action="Left missing; cannot infer reliably without source-system data.",
            loan_id=df.at[idx, "Loan ID"],
        )

    return df


# ---------------------------------------------------------------------------
# 6. Internal risk-rating cleanup
# ---------------------------------------------------------------------------
def clean_risk_rating(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Valid scale is 1 (strongest) through 10 (weakest).
    Invalid values such as 0, 12, or NR are converted to missing and logged.
    """
    df = df.copy()

    col = "Internal Risk Rating (1-10)"
    original = df[col].copy()
    numeric = pd.to_numeric(original, errors="coerce")

    invalid = original.notna() & (~numeric.between(1, 10))
    for idx in df.index[invalid]:
        add_issue(
            issues,
            issue_type="Risk Rating",
            description="Internal risk rating is outside the documented 1-10 scale.",
            action="Set to missing; do not force invalid value into a valid rating bucket.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original.at[idx],
            cleaned_value=None,
        )

    numeric = numeric.where(numeric.between(1, 10), np.nan)
    df[col] = numeric.astype("Int64")

    return df


# ---------------------------------------------------------------------------
# 7. Date parsing and tenor validation
# ---------------------------------------------------------------------------
def clean_dates_and_validate_tenor(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Parse dates, flag missing/illogical dates, and calculate implied tenor.
    Do not invent a corrected maturity date when source information is unavailable.
    """
    df = df.copy()

    orig_col = "Origination Date"
    mat_col = "Maturity Date"

    original_orig = df[orig_col].copy()
    original_mat = df[mat_col].copy()

    df[orig_col] = pd.to_datetime(df[orig_col], errors="coerce", dayfirst=True)
    df[mat_col] = pd.to_datetime(df[mat_col], errors="coerce", dayfirst=True)

    bad_orig_parse = original_orig.notna() & df[orig_col].isna()
    for idx in df.index[bad_orig_parse]:
        add_issue(
            issues,
            issue_type="Date",
            description="Origination Date could not be parsed.",
            action="Set to missing; verify against source system.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original_orig.at[idx],
        )

    bad_mat_parse = original_mat.notna() & df[mat_col].isna()
    for idx in df.index[bad_mat_parse]:
        add_issue(
            issues,
            issue_type="Date",
            description="Maturity Date could not be parsed.",
            action="Set to missing; verify against source system.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original_mat.at[idx],
        )

    missing_maturity = df[mat_col].isna()
    for idx in df.index[missing_maturity]:
        add_issue(
            issues,
            issue_type="Date",
            description="Maturity Date is missing.",
            action="Left missing; cannot infer reliably.",
            loan_id=df.at[idx, "Loan ID"],
        )

    reversed_dates = (
        df[orig_col].notna()
        & df[mat_col].notna()
        & (df[mat_col] < df[orig_col])
    )
    for idx in df.index[reversed_dates]:
        add_issue(
            issues,
            issue_type="Date",
            description="Maturity Date is earlier than Origination Date.",
            action="Retained original dates and flagged for source-system verification.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=f"{df.at[idx, orig_col]} -> {df.at[idx, mat_col]}",
        )

    # Calculate implied tenor only when date order is logical.
    valid_date_order = (
        df[orig_col].notna()
        & df[mat_col].notna()
        & (df[mat_col] >= df[orig_col])
    )

    implied_tenor = pd.Series(np.nan, index=df.index, dtype="float64")
    implied_tenor.loc[valid_date_order] = (
        (df.loc[valid_date_order, mat_col] - df.loc[valid_date_order, orig_col]).dt.days
        / 365.25
    )
    df["Implied Tenor (Yrs)"] = implied_tenor.round(2)

    df["Stated Tenor (Yrs)"] = pd.to_numeric(
        df["Stated Tenor (Yrs)"], errors="coerce"
    )

    # Difference > 0.5 year is treated as a material mismatch.
    tenor_mismatch = (
        valid_date_order
        & df["Stated Tenor (Yrs)"].notna()
        & ((df["Implied Tenor (Yrs)"] - df["Stated Tenor (Yrs)"]).abs() > 0.5)
    )

    for idx in df.index[tenor_mismatch]:
        add_issue(
            issues,
            issue_type="Tenor",
            description="Stated tenor does not agree with origination/maturity dates.",
            action="Retained stated tenor and added Implied Tenor (Yrs) for comparison.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=df.at[idx, "Stated Tenor (Yrs)"],
            cleaned_value=df.at[idx, "Implied Tenor (Yrs)"],
        )

    return df


# ---------------------------------------------------------------------------
# 8. LTV validation
# ---------------------------------------------------------------------------
def clean_and_validate_ltv(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Convert LTV to numeric.
    Negative LTV is invalid and is set to missing.
    LTV above 100% is retained but flagged because it can be economically possible
    (e.g., collateral value deterioration) and should not be automatically corrected.
    """
    df = df.copy()

    col = "LTV (%)"
    df[col] = pd.to_numeric(df[col], errors="coerce")

    negative_ltv = df[col] < 0
    for idx in df.index[negative_ltv]:
        original = df.at[idx, col]
        df.at[idx, col] = np.nan

        add_issue(
            issues,
            issue_type="LTV",
            description="Negative LTV is not economically meaningful.",
            action="Set to missing; verify collateral and exposure values.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=original,
            cleaned_value=np.nan,
        )

    high_ltv = df[col] > 100
    for idx in df.index[high_ltv]:
        add_issue(
            issues,
            issue_type="LTV",
            description="LTV exceeds 100%.",
            action=(
                "Retained value because >100% LTV can be possible; "
                "flagged for verification rather than automatically correcting it."
            ),
            loan_id=df.at[idx, "Loan ID"],
            original_value=df.at[idx, col],
            cleaned_value=df.at[idx, col],
        )

    return df


# ---------------------------------------------------------------------------
# 9. Duplicate checks
# ---------------------------------------------------------------------------
def flag_duplicates(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Flag duplicate Loan IDs and potential duplicate facilities.
    Do not delete records when the evidence is ambiguous.
    """
    df = df.copy()

    duplicate_id = df["Loan ID"].duplicated(keep=False)
    df["Duplicate Loan ID Flag"] = duplicate_id

    for idx in df.index[duplicate_id]:
        add_issue(
            issues,
            issue_type="Duplicate",
            description="Loan ID appears more than once.",
            action="Retained both rows because the underlying loan details differ; verify unique key.",
            loan_id=df.at[idx, "Loan ID"],
        )

    # Conservative potential-duplicate rule:
    # same borrower + facility + balance + origination + maturity.
    duplicate_keys = [
        "Borrower Name",
        "Facility Type",
        "Outstanding Balance",
        "Origination Date",
        "Maturity Date",
    ]

    potential_duplicate = df.duplicated(subset=duplicate_keys, keep=False)
    df["Potential Duplicate Facility Flag"] = potential_duplicate

    for idx in df.index[potential_duplicate]:
        add_issue(
            issues,
            issue_type="Duplicate",
            description=(
                "Possible duplicate facility: borrower, facility type, balance, "
                "origination date, and maturity date match another row."
            ),
            action="Retained record; differences in Loan ID/rate terms may indicate separate facilities.",
            loan_id=df.at[idx, "Loan ID"],
        )

    return df


# ---------------------------------------------------------------------------
# 10. Exposure outlier check
# ---------------------------------------------------------------------------
def flag_exposure_outliers(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Flag extreme balance outliers using a conservative 3*IQR rule.
    Outliers are not changed; this is a review flag, not a correction.
    """
    df = df.copy()

    balance = df["Outstanding Balance"]
    q1 = balance.quantile(0.25)
    q3 = balance.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 3 * iqr

    outlier = balance > threshold
    df["Exposure Outlier Flag"] = outlier

    for idx in df.index[outlier]:
        portfolio_total = balance.sum()
        share = (
            df.at[idx, "Outstanding Balance"] / portfolio_total
            if portfolio_total
            else np.nan
        )

        add_issue(
            issues,
            issue_type="Exposure Outlier",
            description=(
                f"Outstanding balance exceeds 3*IQR threshold "
                f"({threshold:,.0f}) and represents {share:.1%} of portfolio exposure."
            ),
            action="Retained value; verify large exposure before Credit Committee presentation.",
            loan_id=df.at[idx, "Loan ID"],
            original_value=df.at[idx, "Outstanding Balance"],
            cleaned_value=df.at[idx, "Outstanding Balance"],
        )

    return df



# ---------------------------------------------------------------------------
# 11. Additional fuzzy-text quality check (RapidFuzz)
# ---------------------------------------------------------------------------
def flag_fuzzy_text_anomalies(
    df: pd.DataFrame,
    issues: list[dict],
    category_threshold: float = 85.0,
    borrower_threshold: float = 93.0,
) -> pd.DataFrame:
    """
    Run AFTER the deterministic cleaning functions.

    This function is deliberately flag-only:
      - It does NOT automatically revise values.
      - It uses RapidFuzz to identify remaining text values that may be typos.
      - Controlled categorical fields are compared with expected canonical values.
      - Borrower names are checked for highly similar near-duplicates.

    Rationale:
      Fuzzy matching is useful for finding issues that were not anticipated in the
      explicit mapping dictionary, but similarity alone is not sufficient evidence
      to overwrite investment data.
    """
    df = df.copy()

    expected_values = {
        "Region": [
            "Africa", "Asia Pacific", "Europe", "Latin America",
            "Middle East", "North America",
        ],
        "Borrower Type": [
            "Corporate", "Financial Institution", "SME",
            "Sovereign / Public Sector",
        ],
        "Facility Type": [
            "Bridge Loan", "Overdraft Facility", "Project Finance Loan",
            "Revolving Credit Facility", "Syndicated Loan", "Term Loan",
            "Trade Finance Facility", "Working Capital Facility",
        ],
        "Interest Rate Type": ["Fixed", "Floating"],
        "Base Rate Index": [
            "BBSY", "CDI", "CORRA", "EIBOR", "EURIBOR", "JIBAR", "KOFR",
            "LPR", "MIBOR", "NIBOR", "SARON", "SOFR", "SONIA", "SORA",
            "TIIE", "TONA",
        ],
        "Loan Status": ["Performing", "Watchlist", "Non-Performing"],
        "Industry": [
            "Agriculture", "Automotive", "Chemicals", "Construction", "Energy",
            "Financial Services", "Food & Beverage", "Healthcare", "Hospitality",
            "Manufacturing", "Mining & Metals", "Pharmaceuticals", "Real Estate",
            "Retail & Consumer", "Technology", "Telecommunications",
            "Transportation & Logistics", "Utilities",
        ],
    }

    # A. Controlled-category fuzzy check.
    for col, valid_values in expected_values.items():
        if col not in df.columns:
            continue

        for idx, value in df[col].items():
            if pd.isna(value):
                continue

            text_value = str(value).strip()

            if text_value in valid_values:
                continue

            match = process.extractOne(
                text_value,
                valid_values,
                scorer=fuzz.WRatio,
            )

            if match is None:
                continue

            suggested_value, score, _ = match

            if score >= category_threshold:
                add_issue(
                    issues,
                    issue_type="Fuzzy Text Review",
                    description=(
                        f"RapidFuzz found a possible text inconsistency in {col}: "
                        f"'{text_value}' is {score:.1f}% similar to "
                        f"'{suggested_value}'."
                    ),
                    action=(
                        "Flag only. No automatic replacement was made because "
                        "fuzzy similarity is not sufficient evidence to revise "
                        "investment data; verify against the data dictionary or source system."
                    ),
                    loan_id=df.at[idx, "Loan ID"],
                    original_value=text_value,
                    cleaned_value=f"Suggested review: {suggested_value}",
                )
            else:
                add_issue(
                    issues,
                    issue_type="Unexpected Category",
                    description=(
                        f"{col} contains '{text_value}', which is not in the "
                        "expected controlled vocabulary."
                    ),
                    action=(
                        "Flag only. No automatic replacement was made; verify "
                        "whether this is a valid new category or a data-entry issue."
                    ),
                    loan_id=df.at[idx, "Loan ID"],
                    original_value=text_value,
                    cleaned_value=None,
                )

    # B. Borrower-name near-duplicate check.
    # Use a high threshold because different legal entities can have similar names.
    if "Borrower Name" in df.columns:
        borrower_values = (
            df["Borrower Name"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_borrowers = sorted(set(borrower_values))
        flagged_pairs: set[tuple[str, str]] = set()

        for borrower in unique_borrowers:
            candidates = [x for x in unique_borrowers if x != borrower]
            if not candidates:
                continue

            match = process.extractOne(
                borrower,
                candidates,
                scorer=fuzz.WRatio,
            )

            if match is None:
                continue

            similar_name, score, _ = match
            pair = tuple(sorted((borrower, similar_name)))

            if score >= borrower_threshold and pair not in flagged_pairs:
                flagged_pairs.add(pair)

                affected_rows = df.index[
                    df["Borrower Name"].astype("string").isin(pair)
                ]

                for idx in affected_rows:
                    current_name = str(df.at[idx, "Borrower Name"])
                    other_name = pair[1] if current_name == pair[0] else pair[0]

                    add_issue(
                        issues,
                        issue_type="Fuzzy Borrower Review",
                        description=(
                            f"Borrower name '{current_name}' is {score:.1f}% "
                            f"similar to another borrower name '{other_name}'."
                        ),
                        action=(
                            "Flag only. Do not merge borrowers automatically; "
                            "verify legal entity identity, Loan ID, and source-system records."
                        ),
                        loan_id=df.at[idx, "Loan ID"],
                        original_value=current_name,
                        cleaned_value=f"Possible near-duplicate: {other_name}",
                    )

    return df


# ---------------------------------------------------------------------------
# 12. Classify unresolved issues that require manual review
# ---------------------------------------------------------------------------
def classify_manual_review_issues(
    issues: list[dict],
) -> list[dict]:
    """
    Add a Manual Review Required field to every audit-log issue.

    The distinction is intentional:
      - Deterministic corrections (whitespace, known category mappings, unit
        conversion, clearing fields that are not applicable) are logged but
        do not require further review.
      - Values that cannot be reliably inferred, ambiguous duplicates,
        unusual exposures, fuzzy matches, and source-data inconsistencies
        require manual/source-system review.

    This function runs after all cleaning and quality checks, so the final
    review decision is based on the complete set of detected issues.
    """
    unresolved_issue_types = {
        "Outstanding Balance",
        "Risk Rating",
        "Date",
        "Tenor",
        "LTV",
        "Duplicate",
        "Exposure Outlier",
        "Fuzzy Text Review",
        "Unexpected Category",
        "Fuzzy Borrower Review",
        "Record Count",
    }

    for item in issues:
        issue_type = str(item.get("Issue Type") or "")
        description = str(item.get("Description") or "")

        manual_review = issue_type in unresolved_issue_types

        # Interest-rate issues are mixed: some are deterministic corrections,
        # while missing required fields cannot be safely inferred.
        if issue_type == "Interest Rate":
            manual_review = "missing" in description.lower()

        # Formatting and known category mappings are already resolved.
        if issue_type in {"Text Formatting", "Category Standardization"}:
            manual_review = False

        item["Manual Review Required"] = bool(manual_review)

    return issues


# ---------------------------------------------------------------------------
# 13. Add final row-level manual-review flag and reason
# ---------------------------------------------------------------------------
def add_manual_review_flags(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """
    Add final decision fields to Cleaned_Data:
      - Manual Review Required: TRUE only for unresolved issues.
      - Manual Review Reason: semicolon-separated unresolved descriptions.

    Portfolio-level issues without a Loan ID remain in Data_Quality_Log only.
    """
    df = df.copy()

    review_map: dict[str, list[str]] = {}

    for item in issues:
        if not item.get("Manual Review Required", False):
            continue

        loan_id = item.get("Loan ID")
        if loan_id is None or pd.isna(loan_id):
            continue

        key = str(loan_id)
        reason = str(item.get("Description") or item.get("Issue Type") or "Review required")

        review_map.setdefault(key, [])
        if reason not in review_map[key]:
            review_map[key].append(reason)

    loan_ids = df["Loan ID"].astype("string")
    df["Manual Review Required"] = loan_ids.map(
        lambda x: str(x) in review_map
    ).fillna(False)

    df["Manual Review Reason"] = loan_ids.map(
        lambda x: "; ".join(review_map.get(str(x), []))
    ).fillna("")

    return df


# ---------------------------------------------------------------------------
# 14. Consolidated row-level issue flag
# ---------------------------------------------------------------------------
def add_issue_summary(
    df: pd.DataFrame,
    issues: list[dict],
) -> pd.DataFrame:
    """Add one readable Issue Summary field to each cleaned loan record."""
    df = df.copy()

    issue_map: dict[str, list[str]] = {}

    for item in issues:
        loan_id = item.get("Loan ID")
        if pd.isna(loan_id) or loan_id is None:
            continue

        issue_map.setdefault(str(loan_id), [])
        label = str(item["Issue Type"])
        if label not in issue_map[str(loan_id)]:
            issue_map[str(loan_id)].append(label)

    df["Issue Summary"] = (
        df["Loan ID"]
        .astype("string")
        .map(lambda x: "; ".join(issue_map.get(str(x), [])))
        .fillna("")
    )

    return df


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_loan_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the cleaning functions in a transparent, auditable order."""
    issues: list[dict] = []

    df = validate_row_count(df, issues)
    df = clean_text_fields(df, issues)
    df = standardize_categories(df, issues)
    df = clean_outstanding_balance(df, issues)
    df = clean_interest_rate_fields(df, issues)
    df = clean_risk_rating(df, issues)
    df = clean_dates_and_validate_tenor(df, issues)
    df = clean_and_validate_ltv(df, issues)
    df = flag_duplicates(df, issues)
    df = flag_exposure_outliers(df, issues)
    df = flag_fuzzy_text_anomalies(df, issues)

    # Final classification: distinguish resolved corrections from unresolved items.
    issues = classify_manual_review_issues(issues)
    df = add_manual_review_flags(df, issues)
    df = add_issue_summary(df, issues)

    quality_log = pd.DataFrame(issues)

    return df, quality_log


# ---------------------------------------------------------------------------
# Excel input / output
# ---------------------------------------------------------------------------
def read_input_excel(path: Path, sheet_name: str = DATA_SHEET) -> pd.DataFrame:
    """Read the raw loan data sheet."""
    return pd.read_excel(path, sheet_name=sheet_name)


def write_output_excel(
    cleaned_df: pd.DataFrame,
    quality_log: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the clean dataset, final review flags, and audit log to a new workbook."""
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        cleaned_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        quality_log.to_excel(writer, sheet_name="Data_Quality_Log", index=False)

        # Basic usability formatting.
        for sheet_name in ["Cleaned_Data", "Data_Quality_Log"]:
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for column_cells in ws.columns:
                header = str(column_cells[0].value or "")
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells[:200]
                )
                width = min(max(max_length + 2, len(header) + 2), 35)
                ws.column_dimensions[column_cells[0].column_letter].width = width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean commercial-bank loan portfolio Excel data."
    )
    parser.add_argument("input_excel", type=Path, help="Raw input .xlsx file")
    parser.add_argument("output_excel", type=Path, help="Clean output .xlsx file")
    parser.add_argument(
        "--sheet",
        default=DATA_SHEET,
        help=f"Input sheet name (default: {DATA_SHEET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = read_input_excel(args.input_excel, sheet_name=args.sheet)
    cleaned_df, quality_log = clean_loan_portfolio(raw_df)
    write_output_excel(cleaned_df, quality_log, args.output_excel)

    print(f"Input rows:   {len(raw_df):,}")
    print(f"Output rows:  {len(cleaned_df):,}")
    print(f"Logged issues:{len(quality_log):,}")
    print(f"Saved: {args.output_excel}")


if __name__ == "__main__":
    main()
