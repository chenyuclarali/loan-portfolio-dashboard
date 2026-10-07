"""
Streamlit dashboard for the loan-portfolio case study.

The dashboard intentionally separates four concerns:
1. Data Cleaning: run deterministic cleaning rules, inspect validation logs, and
   resolve only those exceptions that require source-system or reviewer judgment.
2. Portfolio Summary: summarize only rows explicitly marked ``Include in Analysis = TRUE``
   using exposure-weighted statistics. Review status and analytical inclusion are separate.
3. Statistical Analysis: use exploratory logistic regression to identify characteristics
   associated with weaker loan performance and K-means clustering to identify natural
   portfolio risk segments.
4. Workflow: explain the control framework and the distinction between automatic
   corrections, flag-only checks, and human review.

State is stored in ``st.session_state`` so the uploaded workbook, cleaned data, manual
review decisions, and audit log survive navigation between pages during the current
browser session.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean_loan_portfolio_FINAL import clean_loan_portfolio


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Portfolio Cleaning & Credit Review",
    page_icon="📊",
    layout="wide",
)

DATA_SHEET = "Loan_Portfolio_Data"
BALANCE_COL = "Outstanding Balance"
RECORD_ID_COL = "_Record ID"

SYSTEM_COLUMNS = {
    RECORD_ID_COL,
    "Implied Tenor (Yrs)",
    "Duplicate Loan ID Flag",
    "Potential Duplicate Facility Flag",
    "Exposure Outlier Flag",
    "Manual Review Required",
    "Manual Review Reason",
    "Include in Analysis",
    "Issue Summary",
    "Reviewer Decision",
    "Reviewer Note",
}

RAW_BUSINESS_COLUMNS = [
    "Loan ID",
    "Borrower Name",
    "Country",
    "Region",
    "Industry",
    "Borrower Type",
    "Facility Type",
    "Outstanding Balance",
    "Interest Rate Type",
    "Base Rate Index",
    "Margin (bps)",
    "Fixed Rate (%)",
    "Origination Date",
    "Maturity Date",
    "Stated Tenor (Yrs)",
    "Internal Risk Rating (1-10)",
    "Loan Status",
    "LTV (%)",
    "Relationship Manager",
]


# -----------------------------------------------------------------------------
# State helpers
# -----------------------------------------------------------------------------
def initialize_state() -> None:
    """Create persistent session variables once per browser session.

    Streamlit reruns the script after most UI interactions.  Keeping the workbook bytes
    and analysis objects in session state prevents page navigation from behaving like a
    fresh upload/cleaning run.
    """
    defaults = {
        "source_file_name": None,
        "source_file_bytes": None,
        "source_signature": None,
        "raw_df": None,
        "cleaned_df": None,
        "initial_log": None,
        "current_log": None,
        "manual_edit_log": pd.DataFrame(),
        "selected_sheet": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    """Clear derived analysis while leaving the currently uploaded file available."""
    st.session_state.raw_df = None
    st.session_state.cleaned_df = None
    st.session_state.initial_log = None
    st.session_state.current_log = None
    st.session_state.manual_edit_log = pd.DataFrame()


def dataframe_signature(file_bytes: bytes, sheet_name: str) -> tuple[int, int, str]:
    # A lightweight in-session change detector is sufficient here; this is not intended
    # to be a cryptographic file fingerprint.  Including the sheet name also ensures
    # that switching the source sheet starts a new analysis.
    return (len(file_bytes), hash(file_bytes[:4096]), sheet_name)


# -----------------------------------------------------------------------------
# Excel I/O
# -----------------------------------------------------------------------------
def read_uploaded_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def format_excel_sheet(writer: pd.ExcelWriter, sheet_name: str) -> None:
    ws = writer.book[sheet_name]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column_cells in ws.columns:
        header = str(column_cells[0].value or "")
        sample = column_cells[:200]
        max_length = max(
            [len(str(cell.value)) if cell.value is not None else 0 for cell in sample]
            + [len(header)]
        )
        ws.column_dimensions[column_cells[0].column_letter].width = min(
            max(max_length + 2, 12), 42
        )


def build_download_workbook(
    cleaned_df: pd.DataFrame,
    current_log: pd.DataFrame,
    initial_log: pd.DataFrame | None,
    manual_edit_log: pd.DataFrame | None,
    original_df: pd.DataFrame | None = None,
) -> bytes:
    """
    Build a downloadable Excel workbook entirely in memory.

    The export preserves the original source extract alongside the current cleaned
    dataset, validation log, review/exclusion subsets, original cleaning log, and
    manual-edit audit trail. This makes the workbook self-contained for audit/review.
    """
    output = io.BytesIO()

    export_df = cleaned_df.copy()
    # Internal record ID is useful for auditability in the dashboard, but it is not
    # part of the source schema. Keep it at the end of the exported dataset.
    if RECORD_ID_COL in export_df.columns:
        cols = [c for c in export_df.columns if c != RECORD_ID_COL] + [RECORD_ID_COL]
        export_df = export_df[cols]

    # Keep both unresolved review items and manually excluded records easy to audit.
    # A reviewed-but-excluded duplicate may no longer require manual review, yet it is
    # still analytically important to retain in the workbook.
    manual_review = export_df[
        export_df["Manual Review Required"].fillna(False).astype(bool)
        | ~export_df["Include in Analysis"].fillna(False).astype(bool)
    ].copy()

    analysis_included = export_df[
        export_df["Include in Analysis"].fillna(False).astype(bool)
    ].copy()
    analysis_excluded = export_df[
        ~export_df["Include in Analysis"].fillna(False).astype(bool)
    ].copy()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        if original_df is not None:
            original_df.to_excel(writer, sheet_name="Original_Data", index=False)

        export_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        current_log.to_excel(writer, sheet_name="Current_Validation_Log", index=False)
        manual_review.to_excel(writer, sheet_name="Manual_Review", index=False)
        analysis_included.to_excel(writer, sheet_name="Analysis_Included", index=False)
        analysis_excluded.to_excel(writer, sheet_name="Analysis_Excluded", index=False)

        if initial_log is not None and not initial_log.empty:
            initial_log.to_excel(writer, sheet_name="Initial_Cleaning_Log", index=False)

        if manual_edit_log is not None and not manual_edit_log.empty:
            manual_edit_log.to_excel(writer, sheet_name="Manual_Edit_Log", index=False)

        for sheet_name in writer.book.sheetnames:
            format_excel_sheet(writer, sheet_name)

    return output.getvalue()


# -----------------------------------------------------------------------------
# Cleaning / validation workflow
# -----------------------------------------------------------------------------
def run_initial_cleaning(raw_df: pd.DataFrame) -> None:
    """Run the standalone cleaner and initialize reviewer/audit fields.

    ``_Record ID`` is a dashboard-only row key.  It is intentionally independent of
    Loan ID so two source rows can still be edited separately when the source itself
    contains a duplicated Loan ID.
    """
    working = raw_df.copy()
    working.insert(0, RECORD_ID_COL, np.arange(1, len(working) + 1))

    cleaned_df, quality_log = clean_loan_portfolio(working)

    # The cleaner initializes analysis inclusion conservatively:
    #   Manual Review Required = FALSE -> Include in Analysis = TRUE
    #   Manual Review Required = TRUE  -> Include in Analysis = FALSE
    # Inclusion is a separate analytical control. During manual review the user may
    # change it directly; no Reviewer Decision automatically forces inclusion/exclusion.
    if "Include in Analysis" not in cleaned_df.columns:
        cleaned_df["Include in Analysis"] = ~cleaned_df[
            "Manual Review Required"
        ].fillna(False).astype(bool)

    cleaned_df["Reviewer Decision"] = "Pending"
    cleaned_df["Reviewer Note"] = ""

    st.session_state.raw_df = raw_df.copy()
    st.session_state.cleaned_df = cleaned_df
    st.session_state.initial_log = quality_log.copy()
    st.session_state.current_log = quality_log.copy()
    st.session_state.manual_edit_log = pd.DataFrame(
        columns=[
            "Timestamp",
            RECORD_ID_COL,
            "Loan ID",
            "Field",
            "Previous Value",
            "Revised Value",
            "Reviewer Decision",
            "Reviewer Note",
        ]
    )


def values_equal(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return str(a) == str(b)


def apply_verified_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve row-level review status for explicit reviewer conclusions.

    Two reviewer decisions are treated as resolved:
      - Verified - keep as-is: the unusual value has been confirmed as valid.
      - Confirmed duplicate - exclude: the row is confirmed to be a duplicate record.

    IMPORTANT: this function does NOT change ``Include in Analysis``.
    Analytical inclusion is an independent manual control. The reviewer must explicitly
    check or uncheck Include in Analysis in the Manual Review table.

    This separation allows, for example:
      - a verified large exposure to be included;
      - a confirmed duplicate to remain excluded;
      - a still-open item to be temporarily included if the reviewer chooses.
    """
    df = df.copy()
    resolved = df["Reviewer Decision"].isin(
        ["Verified - keep as-is", "Confirmed duplicate - exclude"]
    )
    df.loc[resolved, "Manual Review Required"] = False
    df.loc[resolved, "Manual Review Reason"] = ""
    return df


def apply_manual_review_edits(edited_review_df: pd.DataFrame) -> None:
    """
    Apply user revisions to the current clean dataset and rerun all checks.

    Why rerun instead of manually clearing flags?
    Because a corrected rating/date/duplicate key should be validated by the same
    business rules that created the flag in the first place. ``Corrected - recheck``
    therefore changes the underlying field and lets the rules determine whether the
    issue is resolved. ``Verified - keep as-is`` and ``Confirmed duplicate - exclude``
    can close the review status when appropriate.

    ``Include in Analysis`` is separate from all of those statuses. Its checkbox is
    always the authoritative instruction for Portfolio Summary.
    """
    current = st.session_state.cleaned_df.copy()
    if current is None or current.empty:
        return

    current_by_id = current.set_index(RECORD_ID_COL, drop=False)
    edited_by_id = edited_review_df.set_index(RECORD_ID_COL, drop=False)

    audit_rows: list[dict] = []
    decision_map: dict[int, tuple[str, str, bool]] = {}

    editable_fields = [c for c in RAW_BUSINESS_COLUMNS if c in current.columns]

    for record_id, revised_row in edited_by_id.iterrows():
        if record_id not in current_by_id.index:
            continue

        decision = str(revised_row.get("Reviewer Decision", "Pending") or "Pending")
        note = str(revised_row.get("Reviewer Note", "") or "")
        include_in_analysis = bool(revised_row.get("Include in Analysis", False))
        decision_map[int(record_id)] = (decision, note, include_in_analysis)

        original_row = current_by_id.loc[record_id]
        loan_id = revised_row.get("Loan ID")

        for col in editable_fields:
            before = original_row.get(col)
            after = revised_row.get(col)

            if not values_equal(before, after):
                audit_rows.append(
                    {
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                        RECORD_ID_COL: int(record_id),
                        "Loan ID": loan_id,
                        "Field": col,
                        "Previous Value": before,
                        "Revised Value": after,
                        "Reviewer Decision": decision,
                        "Reviewer Note": note,
                    }
                )
                current_by_id.at[record_id, col] = after

        # ``Include in Analysis`` is the reviewer's explicit analytical decision.
        # It is intentionally not derived from Reviewer Decision.
        previous_include = bool(
            original_row.get(
                "Include in Analysis",
                not bool(original_row.get("Manual Review Required", False)),
            )
        )
        if include_in_analysis != previous_include:
            audit_rows.append(
                {
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    RECORD_ID_COL: int(record_id),
                    "Loan ID": loan_id,
                    "Field": "Include in Analysis",
                    "Previous Value": previous_include,
                    "Revised Value": include_in_analysis,
                    "Reviewer Decision": decision,
                    "Reviewer Note": note,
                }
            )

        current_by_id.at[record_id, "Include in Analysis"] = include_in_analysis
        current_by_id.at[record_id, "Reviewer Decision"] = decision
        current_by_id.at[record_id, "Reviewer Note"] = note

    revised = current_by_id.reset_index(drop=True)

    # Remove derived/previous-validation columns before rerunning the cleaning engine
    # so stale flags cannot survive a corrected value.  The dashboard Record ID and
    # reviewer fields are retained because the cleaner safely carries unknown columns.
    derived_cols = [
        "Implied Tenor (Yrs)",
        "Duplicate Loan ID Flag",
        "Potential Duplicate Facility Flag",
        "Exposure Outlier Flag",
        "Manual Review Required",
        "Manual Review Reason",
        "Include in Analysis",
        "Issue Summary",
    ]
    revised_for_validation = revised.drop(
        columns=[c for c in derived_cols if c in revised.columns],
        errors="ignore",
    )

    revalidated_df, current_log = clean_loan_portfolio(revised_for_validation)

    # Restore / normalize review decisions after revalidation.
    if "Reviewer Decision" not in revalidated_df.columns:
        revalidated_df["Reviewer Decision"] = "Pending"
    if "Reviewer Note" not in revalidated_df.columns:
        revalidated_df["Reviewer Note"] = ""

    for record_id, (decision, note, include_in_analysis) in decision_map.items():
        mask = revalidated_df[RECORD_ID_COL].eq(record_id)
        revalidated_df.loc[mask, "Reviewer Decision"] = decision
        revalidated_df.loc[mask, "Reviewer Note"] = note

        # Restore the reviewer's explicit inclusion choice after revalidation.
        # Validation may change Manual Review Required, but it must not overwrite the
        # user's analytical-inclusion decision.
        revalidated_df.loc[mask, "Include in Analysis"] = bool(include_in_analysis)

        # Log the review decision itself, even when no business field was changed.
        previous_decision = str(
            current.loc[current[RECORD_ID_COL].eq(record_id), "Reviewer Decision"].iloc[0]
        )
        previous_note = str(
            current.loc[current[RECORD_ID_COL].eq(record_id), "Reviewer Note"].iloc[0]
        )
        if decision != previous_decision or note != previous_note:
            loan_id = revalidated_df.loc[mask, "Loan ID"].iloc[0]
            audit_rows.append(
                {
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    RECORD_ID_COL: int(record_id),
                    "Loan ID": loan_id,
                    "Field": "Manual Review Resolution",
                    "Previous Value": f"{previous_decision} | {previous_note}",
                    "Revised Value": f"{decision} | {note}",
                    "Reviewer Decision": decision,
                    "Reviewer Note": note,
                }
            )

    # Corrected records remain subject to every validation rule. Explicit reviewer
    # conclusions can close the review status, but they never determine analytical
    # inclusion. Include in Analysis remains the independent checkbox selected above.
    revalidated_df = apply_verified_overrides(revalidated_df)

    # Keep the validation log consistent with explicit reviewer acceptance.  The
    # standalone cleaner's audit log is keyed by Loan ID rather than the dashboard-only
    # Record ID, so duplicate Loan IDs are inherently less granular in this log.  The
    # Cleaned_Data/manual-review table remains row-specific because it uses Record ID.
    current_log = current_log.copy()
    current_log["Reviewer Decision"] = ""
    current_log["Reviewer Note"] = ""
    for record_id, (decision, note, include_in_analysis) in decision_map.items():
        if decision not in {"Verified - keep as-is", "Confirmed duplicate - exclude"}:
            continue
        loan_ids = revalidated_df.loc[
            revalidated_df[RECORD_ID_COL].eq(record_id), "Loan ID"
        ].astype("string")
        if loan_ids.empty:
            continue
        loan_id = loan_ids.iloc[0]
        log_mask = current_log["Loan ID"].astype("string").eq(loan_id)
        current_log.loc[log_mask, "Manual Review Required"] = False
        current_log.loc[log_mask, "Reviewer Decision"] = decision
        current_log.loc[log_mask, "Reviewer Note"] = note

    if audit_rows:
        new_audit = pd.DataFrame(audit_rows)
        st.session_state.manual_edit_log = pd.concat(
            [st.session_state.manual_edit_log, new_audit],
            ignore_index=True,
        )

    st.session_state.cleaned_df = revalidated_df
    st.session_state.current_log = current_log


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------
# Unless otherwise stated, portfolio percentages are exposure-weighted using cleaned
# Outstanding Balance.  This is more decision-relevant for credit concentration than
# simple loan counts because a $1M loan and an $800M loan should not contribute equally.
def exposure_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    total = df[BALANCE_COL].sum()
    result = (
        df.assign(**{group_col: df[group_col].fillna("Missing")})
        .groupby(group_col, dropna=False)
        .agg(
            Loan_Count=("Loan ID", "size"),
            Exposure=(BALANCE_COL, "sum"),
        )
        .reset_index()
        .sort_values("Exposure", ascending=False)
    )
    result["Exposure %"] = np.where(total != 0, result["Exposure"] / total, np.nan)
    return result


def risk_rating_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize rating exposure while preserving invalid/unrated loans in total exposure."""
    working = df.copy()
    rating = working["Internal Risk Rating (1-10)"].astype("Int64")
    working["Risk Rating"] = rating.astype("string").fillna("Unrated / Invalid")

    total = working[BALANCE_COL].sum()
    result = (
        working.groupby("Risk Rating", dropna=False)
        .agg(Loan_Count=("Loan ID", "size"), Exposure=(BALANCE_COL, "sum"))
        .reset_index()
    )
    result["Exposure %"] = np.where(total != 0, result["Exposure"] / total, np.nan)

    def sort_key(value: str) -> int:
        try:
            return int(value)
        except Exception:
            return 99

    result["_sort"] = result["Risk Rating"].map(sort_key)
    return result.sort_values("_sort").drop(columns="_sort")


def top_borrower_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    total = df[BALANCE_COL].sum()
    result = (
        df.groupby("Borrower Name", dropna=False)
        .agg(Loan_Count=("Loan ID", "size"), Exposure=(BALANCE_COL, "sum"))
        .reset_index()
        .sort_values("Exposure", ascending=False)
        .head(top_n)
    )
    result["Exposure %"] = np.where(total != 0, result["Exposure"] / total, np.nan)
    return result


def concentration_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Region", "Industry", "Borrower Type", "Facility Type"]:
        summary = exposure_summary(df, col)
        if summary.empty:
            continue
        first = summary.iloc[0]
        rows.append(
            {
                "Dimension": col,
                "Largest Category": first[col],
                "Exposure": first["Exposure"],
                "Exposure %": first["Exposure %"],
            }
        )
    return pd.DataFrame(rows)


def assumptions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Document analytical assumptions so dashboard results are reproducible and reviewable."""
    actual_rows = len(df)
    return pd.DataFrame(
        [
            {
                "Assumption / Rule": "Portfolio size",
                "Treatment": (
                    f"The case instructions reference 100 records; the uploaded extract "
                    f"contains {actual_rows}. All available records are retained and the "
                    "difference is flagged for reconciliation."
                ),
            },
            {
                "Assumption / Rule": "Exposure weighting",
                "Treatment": (
                    "Composition, rate mix, risk status, and concentration percentages are "
                    "weighted by cleaned Outstanding Balance rather than loan count."
                ),
            },
            {
                "Assumption / Rule": "Negative outstanding balance",
                "Treatment": (
                    "The automated cleaner treats a negative loan balance as a sign-entry "
                    "error and stores the absolute value as a case-study assumption. Because "
                    "the record still requires manual verification, it is excluded from "
                    "Portfolio Summary by default until the reviewer changes Include in Analysis."
                ),
            },
            {
                "Assumption / Rule": "Internal risk rating",
                "Treatment": (
                    "Only ratings 1 through 10 are treated as valid. Values such as 0, 12, "
                    "or NR are set to missing and shown as Unrated / Invalid in the summary."
                ),
            },
            {
                "Assumption / Rule": "Dates and tenor",
                "Treatment": (
                    "Dates are parsed where possible. Missing or logically inconsistent dates "
                    "are not imputed. Implied tenor is calculated only when date order is valid."
                ),
            },
            {
                "Assumption / Rule": "Duplicates",
                "Treatment": (
                    "Duplicate Loan IDs and possible duplicate facilities are preserved in "
                    "Cleaned_Data for auditability and flagged for review. They are excluded "
                    "from analysis by default; the reviewer manually decides whether each row "
                    "should be included after verification."
                ),
            },
            {
                "Assumption / Rule": "Large exposure / high LTV",
                "Treatment": (
                    "Statistically unusual exposures and LTV above 100% are retained in the "
                    "cleaned dataset because they can be economically valid. They are review "
                    "items and therefore start excluded from analysis; the reviewer may include "
                    "them once comfortable with the value."
                ),
            },
            {
                "Assumption / Rule": "Fuzzy text matching",
                "Treatment": (
                    "RapidFuzz is used only as a secondary flagging control after deterministic "
                    "cleaning. Fuzzy matches never overwrite investment data automatically."
                ),
            },
            {
                "Assumption / Rule": "Analysis inclusion",
                "Treatment": (
                    "Portfolio Summary uses only rows with Include in Analysis = TRUE. "
                    "The automated default excludes rows requiring manual review and includes "
                    "all other rows. During review, Include in Analysis is controlled manually "
                    "and is independent of Reviewer Decision."
                ),
            },
        ]
    )


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.0f}"


def exposure_bar(summary: pd.DataFrame, category_col: str, title: str):
    chart_df = summary.copy()
    chart_df["Exposure % Label"] = chart_df["Exposure %"].map(lambda x: f"{x:.1%}")
    fig = px.bar(
        chart_df.sort_values("Exposure", ascending=True),
        x="Exposure",
        y=category_col,
        orientation="h",
        text="Exposure % Label",
        title=title,
        hover_data={"Loan_Count": True, "Exposure %": ":.1%", "Exposure": ":,.0f"},
    )
    fig.update_layout(height=max(360, 28 * len(chart_df) + 100), yaxis_title=None)
    fig.update_traces(textposition="outside")
    return fig



# -----------------------------------------------------------------------------
# Performance-analysis helpers
# -----------------------------------------------------------------------------
PERFORMANCE_STATUS_ORDER = ["Performing", "Watchlist", "Non-Performing"]
HIGHER_RISK_STATUSES = {"Watchlist", "Non-Performing"}


def normalized_status_order(df: pd.DataFrame) -> list[str]:
    """
    Return a stable Loan Status order for charts.

    The three case-study statuses are shown first. Any unexpected or missing status is
    preserved at the end rather than silently dropped, so the chart remains an audit
    of the actual analytical population.
    """
    observed = (
        df["Loan Status"]
        .fillna("Missing")
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    extras = [s for s in observed if s not in PERFORMANCE_STATUS_ORDER]
    return PERFORMANCE_STATUS_ORDER + sorted(extras)


def performance_mix_by_dimension(
    df: pd.DataFrame,
    dimension: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build exposure-weighted Loan Status mix for a categorical dimension.

    Returns:
      - long_mix: one row per dimension/status combination for a 100% stacked chart.
      - summary: one row per category with total exposure, loan count, and higher-risk
        exposure share (Watchlist + Non-Performing).

    Exposure weighting is used because a $100M problem loan is economically more
    important than a $1M problem loan even though both count as one record.
    """
    working = df.copy()
    working[dimension] = working[dimension].fillna("Missing").astype(str)
    working["Loan Status"] = working["Loan Status"].fillna("Missing").astype(str)

    grouped = (
        working.groupby([dimension, "Loan Status"], dropna=False)
        .agg(
            Exposure=(BALANCE_COL, "sum"),
            Loan_Count=("Loan ID", "size"),
        )
        .reset_index()
    )

    totals = (
        working.groupby(dimension, dropna=False)
        .agg(
            Total_Exposure=(BALANCE_COL, "sum"),
            Total_Loans=("Loan ID", "size"),
        )
        .reset_index()
    )

    long_mix = grouped.merge(totals, on=dimension, how="left")
    long_mix["Exposure Share"] = np.where(
        long_mix["Total_Exposure"] != 0,
        long_mix["Exposure"] / long_mix["Total_Exposure"],
        np.nan,
    )

    high_risk = (
        grouped[grouped["Loan Status"].isin(HIGHER_RISK_STATUSES)]
        .groupby(dimension, dropna=False)["Exposure"]
        .sum()
        .rename("Higher_Risk_Exposure")
        .reset_index()
    )

    summary = totals.merge(high_risk, on=dimension, how="left")
    summary["Higher_Risk_Exposure"] = summary["Higher_Risk_Exposure"].fillna(0.0)
    summary["Higher-Risk Exposure %"] = np.where(
        summary["Total_Exposure"] != 0,
        summary["Higher_Risk_Exposure"] / summary["Total_Exposure"],
        np.nan,
    )

    # Add each observed status as a separate exposure/share column for a compact table.
    exposure_pivot = grouped.pivot_table(
        index=dimension,
        columns="Loan Status",
        values="Exposure",
        aggfunc="sum",
        fill_value=0,
    )
    for status in exposure_pivot.columns:
        summary = summary.merge(
            exposure_pivot[[status]]
            .rename(columns={status: f"{status} Exposure"})
            .reset_index(),
            on=dimension,
            how="left",
        )
        summary[f"{status} %"] = np.where(
            summary["Total_Exposure"] != 0,
            summary[f"{status} Exposure"] / summary["Total_Exposure"],
            np.nan,
        )

    summary = summary.sort_values("Total_Exposure", ascending=False).reset_index(drop=True)
    return long_mix, summary


def performance_stack_chart(
    df: pd.DataFrame,
    dimension: str,
    title: str,
    top_n: int | None = None,
):
    """
    Create a 100% stacked exposure chart by Loan Status.

    Categories are ordered by total exposure so the largest portfolio segments appear
    first. For high-cardinality dimensions such as Relationship Manager, ``top_n`` can
    keep the display readable without changing the underlying summary table.
    """
    long_mix, summary = performance_mix_by_dimension(df, dimension)

    if top_n is not None and len(summary) > top_n:
        keep = summary.head(top_n)[dimension].tolist()
        long_mix = long_mix[long_mix[dimension].isin(keep)].copy()
        summary = summary[summary[dimension].isin(keep)].copy()

    category_order = summary.sort_values("Total_Exposure", ascending=True)[dimension].tolist()

    fig = px.bar(
        long_mix,
        x="Exposure Share",
        y=dimension,
        color="Loan Status",
        orientation="h",
        barmode="stack",
        title=title,
        category_orders={"Loan Status": normalized_status_order(df)},
        hover_data={
            "Exposure": ":,.0f",
            "Loan_Count": True,
            "Total_Exposure": ":,.0f",
            "Exposure Share": ":.1%",
        },
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of category exposure")
    fig.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=category_order,
    )
    fig.update_layout(
        height=max(380, 34 * max(len(category_order), 1) + 120),
        legend_title_text="Loan Status",
    )
    return fig, summary


def display_performance_summary_table(
    summary: pd.DataFrame,
    dimension: str,
) -> None:
    """Format the most decision-useful columns for display."""
    display = summary.copy()
    display["Total Exposure"] = display["Total_Exposure"].map(format_currency)
    display["Higher-Risk Exposure"] = display["Higher_Risk_Exposure"].map(format_currency)
    display["Higher-Risk Exposure %"] = display["Higher-Risk Exposure %"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )

    columns = [
        dimension,
        "Total_Loans",
        "Total Exposure",
        "Higher-Risk Exposure",
        "Higher-Risk Exposure %",
    ]

    for status in normalized_status_order(
        pd.DataFrame({"Loan Status": [
            c.replace(" Exposure", "")
            for c in summary.columns
            if c.endswith(" Exposure") and c != "Higher_Risk_Exposure"
        ]})
    ):
        pct_col = f"{status} %"
        if pct_col in display.columns:
            display[pct_col] = display[pct_col].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
            columns.append(pct_col)

    st.dataframe(
        display[[c for c in columns if c in display.columns]],
        hide_index=True,
        use_container_width=True,
    )


def exposure_weighted_mean(
    frame: pd.DataFrame,
    value_col: str,
    weight_col: str = BALANCE_COL,
) -> float:
    """Return a weighted mean using only rows with valid value and positive weight."""
    valid = frame[[value_col, weight_col]].copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid[weight_col] = pd.to_numeric(valid[weight_col], errors="coerce")
    valid = valid[
        valid[value_col].notna()
        & valid[weight_col].notna()
        & (valid[weight_col] > 0)
    ]
    if valid.empty or valid[weight_col].sum() == 0:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def status_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize LTV, tenor, and risk rating by Loan Status.

    Both ordinary medians and exposure-weighted means are shown. Weighted means describe
    where portfolio dollars sit, while medians reduce the influence of one very large loan.
    """
    working = df.copy()
    working["Loan Status"] = working["Loan Status"].fillna("Missing").astype(str)

    # Prefer implied tenor when the dates are valid; otherwise fall back to stated tenor.
    # This maximizes usable observations while preserving the cleaner's date validation.
    implied = pd.to_numeric(working.get("Implied Tenor (Yrs)"), errors="coerce")
    stated = pd.to_numeric(working.get("Stated Tenor (Yrs)"), errors="coerce")
    working["Analytical Tenor (Yrs)"] = implied.combine_first(stated)

    rows = []
    for status, group in working.groupby("Loan Status", dropna=False):
        ltv = pd.to_numeric(group["LTV (%)"], errors="coerce")
        tenor = pd.to_numeric(group["Analytical Tenor (Yrs)"], errors="coerce")
        rating = pd.to_numeric(group["Internal Risk Rating (1-10)"], errors="coerce")

        rows.append(
            {
                "Loan Status": status,
                "Loan Count": len(group),
                "Exposure": group[BALANCE_COL].sum(),
                "Median LTV (%)": ltv.median(),
                "Exposure-Wtd Avg LTV (%)": exposure_weighted_mean(group, "LTV (%)"),
                "Median Tenor (Yrs)": tenor.median(),
                "Exposure-Wtd Avg Tenor (Yrs)": exposure_weighted_mean(
                    group.assign(**{"Analytical Tenor (Yrs)": tenor}),
                    "Analytical Tenor (Yrs)",
                ),
                "Median Risk Rating": rating.median(),
                "Exposure-Wtd Avg Risk Rating": exposure_weighted_mean(
                    group,
                    "Internal Risk Rating (1-10)",
                ),
            }
        )

    result = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(normalized_status_order(working))}
    result["_order"] = result["Loan Status"].map(lambda x: order.get(x, 99))
    return result.sort_values("_order").drop(columns="_order")





# -----------------------------------------------------------------------------
# Statistical-analysis helpers
# -----------------------------------------------------------------------------
STAT_NUMERIC_FEATURES = {
    "Risk Rating": "Risk Rating",
    "LTV": "LTV",
    "Tenor": "Tenor",
}

STAT_CATEGORICAL_FEATURES = {
    "Interest Rate Type": "Interest Rate Type",
    "Geography (Region)": "Region",
    "Industry": "Industry",
    "Borrower Type": "Borrower Type",
    "Facility Type": "Facility Type",
    "Relationship Manager": "Relationship Manager",
}


def prepare_statistical_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a modeling dataset without changing the underlying cleaned portfolio.

    Outcome:
      Higher Risk = 1 for Watchlist or Non-Performing, 0 for Performing.

    Modeling fields:
      - Risk Rating: numeric 1-10 scale.
      - LTV: numeric percentage.
      - Tenor: valid Implied Tenor when available, otherwise Stated Tenor.
      - Log Exposure: log1p(Outstanding Balance), used only for clustering by default.

    Rows with an unexpected Loan Status are excluded from the binary logistic outcome
    rather than being silently forced into either class.
    """
    model_df = df.copy()

    status = model_df["Loan Status"].astype("string")
    valid_status = status.isin(PERFORMANCE_STATUS_ORDER)
    model_df = model_df.loc[valid_status].copy()

    model_df["Higher Risk"] = model_df["Loan Status"].isin(
        ["Watchlist", "Non-Performing"]
    ).astype(int)

    model_df["Risk Rating"] = pd.to_numeric(
        model_df["Internal Risk Rating (1-10)"],
        errors="coerce",
    )
    model_df["LTV"] = pd.to_numeric(model_df["LTV (%)"], errors="coerce")

    implied = pd.to_numeric(model_df.get("Implied Tenor (Yrs)"), errors="coerce")
    stated = pd.to_numeric(model_df.get("Stated Tenor (Yrs)"), errors="coerce")
    model_df["Tenor"] = implied.combine_first(stated)

    exposure = pd.to_numeric(model_df[BALANCE_COL], errors="coerce")
    # log1p reduces the influence of a few very large facilities when K-means computes
    # Euclidean distance. The original exposure is still used for economic summaries.
    model_df["Log Exposure"] = np.log1p(exposure.clip(lower=0))

    return model_df


def run_core_logistic_inference(model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit a compact, interpretable binary logistic regression.

    Core predictors:
      Risk Rating, LTV, Tenor, Floating-rate indicator.

    Continuous predictors are standardized before fitting, so their odds ratios represent
    a one-standard-deviation increase. Floating is interpreted relative to Fixed.

    Complete-case analysis is used here because inferential p-values / confidence intervals
    are easier to interpret without model-based imputation. The broader regularized model
    below uses imputation for prediction-oriented contributor ranking.

    Returns an empty table plus diagnostics when the model cannot be estimated reliably.
    """
    core = model_df[
        [
            "Higher Risk",
            "Risk Rating",
            "LTV",
            "Tenor",
            "Interest Rate Type",
        ]
    ].copy()

    core["Floating"] = (
        core["Interest Rate Type"]
        .astype("string")
        .str.lower()
        .eq("floating")
        .fillna(False)
        .astype(float)
    )

    core = core.drop(columns=["Interest Rate Type"]).dropna()

    diagnostics = {
        "n": len(core),
        "events": int(core["Higher Risk"].sum()) if not core.empty else 0,
        "non_events": int((1 - core["Higher Risk"]).sum()) if not core.empty else 0,
        "converged": False,
        "pseudo_r2": np.nan,
        "llr_pvalue": np.nan,
        "error": None,
    }

    if (
        len(core) < 20
        or core["Higher Risk"].nunique() < 2
        or core["Higher Risk"].sum() < 3
        or (1 - core["Higher Risk"]).sum() < 3
    ):
        diagnostics["error"] = "Insufficient complete observations or outcome variation."
        return pd.DataFrame(), diagnostics

    continuous = ["Risk Rating", "LTV", "Tenor"]
    means = core[continuous].mean()
    stds = core[continuous].std(ddof=0).replace(0, np.nan)

    x = core[continuous + ["Floating"]].copy()
    x[continuous] = (x[continuous] - means) / stds
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    y = core.loc[x.index, "Higher Risk"].astype(int)
    x = sm.add_constant(x.astype(float), has_constant="add")

    try:
        result = sm.Logit(y, x).fit(disp=False, maxiter=200)

        conf = result.conf_int()
        rows = []
        labels = {
            "Risk Rating": "Risk Rating (per 1 SD worse)",
            "LTV": "LTV (per 1 SD increase)",
            "Tenor": "Tenor (per 1 SD increase)",
            "Floating": "Floating vs Fixed",
        }

        for term in ["Risk Rating", "LTV", "Tenor", "Floating"]:
            rows.append(
                {
                    "Predictor": labels[term],
                    "Coefficient": result.params[term],
                    "Odds Ratio": np.exp(result.params[term]),
                    "95% CI Lower": np.exp(conf.loc[term, 0]),
                    "95% CI Upper": np.exp(conf.loc[term, 1]),
                    "p-value": result.pvalues[term],
                }
            )

        diagnostics.update(
            {
                "converged": bool(result.mle_retvals.get("converged", True)),
                "pseudo_r2": float(result.prsquared),
                "llr_pvalue": float(result.llr_pvalue),
            }
        )
        return pd.DataFrame(rows), diagnostics

    except Exception as exc:
        diagnostics["error"] = str(exc)
        return pd.DataFrame(), diagnostics


def build_regularized_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build an L2-regularized logistic model for exploratory contributor ranking.

    Regularization is important because this case study has a small sample and some
    high-cardinality categorical variables. It reduces coefficient instability but does
    not turn the model into causal evidence.
    """
    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_regularized_contributor_model(
    model_df: pd.DataFrame,
    selected_display_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit a regularized logistic model and rank original variables by permutation importance.

    Permutation importance is calculated at the original-column level, so a multi-level
    field such as Region receives one overall importance score rather than a separate rank
    for every one-hot dummy. This makes the ranking easier to interpret.

    A coefficient table is also returned for directional interpretation. Coefficients for
    categorical levels are relative to the omitted baseline level. Because the model is
    regularized, these coefficient-based odds ratios are descriptive/model-based rather
    than classical maximum-likelihood inferential estimates.
    """
    numeric_cols = [
        STAT_NUMERIC_FEATURES[name]
        for name in selected_display_features
        if name in STAT_NUMERIC_FEATURES
    ]
    categorical_cols = [
        STAT_CATEGORICAL_FEATURES[name]
        for name in selected_display_features
        if name in STAT_CATEGORICAL_FEATURES
    ]
    feature_cols = numeric_cols + categorical_cols

    diagnostics = {
        "n": 0,
        "events": 0,
        "non_events": 0,
        "cv_auc_mean": np.nan,
        "cv_auc_std": np.nan,
        "encoded_features": 0,
        "error": None,
    }

    if not feature_cols:
        diagnostics["error"] = "Select at least one predictor."
        return pd.DataFrame(), pd.DataFrame(), diagnostics

    data = model_df[feature_cols + ["Higher Risk"]].copy()

    # Normalize dtypes before scikit-learn preprocessing. Pandas nullable string columns
    # may contain pd.NA, whose three-valued boolean semantics are not accepted by some
    # sklearn imputers. Convert missing categoricals to ordinary np.nan/object and force
    # numeric predictors through to_numeric. This changes representation, not values.
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical_cols:
        data[col] = data[col].astype(object)
        data[col] = data[col].where(pd.notna(data[col]), np.nan)

    # Do not drop rows for missing predictor values here: the pipeline explicitly imputes
    # numeric/categorical fields so the ranking uses the full included analytical sample.
    y = data.pop("Higher Risk").astype(int)
    X = data

    diagnostics.update(
        {
            "n": len(X),
            "events": int(y.sum()),
            "non_events": int((1 - y).sum()),
        }
    )

    if y.nunique() < 2 or min(y.value_counts()) < 2:
        diagnostics["error"] = "Insufficient outcome variation for logistic regression."
        return pd.DataFrame(), pd.DataFrame(), diagnostics

    pipe = build_regularized_logistic_pipeline(numeric_cols, categorical_cols)

    try:
        pipe.fit(X, y)

        # Cross-validated ROC AUC: use as many folds as class counts reasonably support,
        # capped at 5 for stability and runtime.
        minority_count = int(y.value_counts().min())
        n_splits = min(5, minority_count)
        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )
            auc_scores = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="roc_auc",
            )
            diagnostics["cv_auc_mean"] = float(np.nanmean(auc_scores))
            diagnostics["cv_auc_std"] = float(np.nanstd(auc_scores))

        # Group-level contributor ranking on the original columns.
        perm = permutation_importance(
            pipe,
            X,
            y,
            scoring="roc_auc",
            n_repeats=30,
            random_state=42,
        )
        importance = pd.DataFrame(
            {
                "Feature": feature_cols,
                "Permutation Importance": perm.importances_mean,
                "Importance SD": perm.importances_std,
            }
        ).sort_values("Permutation Importance", ascending=False)

        # Directional coefficient table after preprocessing / one-hot encoding.
        preprocess = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        encoded_names = preprocess.get_feature_names_out()
        coefs = model.coef_[0]

        diagnostics["encoded_features"] = len(encoded_names)

        coef_table = pd.DataFrame(
            {
                "Encoded Predictor": encoded_names,
                "Coefficient": coefs,
                "Model-based Odds Ratio": np.exp(coefs),
            }
        )
        coef_table["Absolute Coefficient"] = coef_table["Coefficient"].abs()
        coef_table = coef_table.sort_values("Absolute Coefficient", ascending=False)

        return importance, coef_table, diagnostics

    except Exception as exc:
        diagnostics["error"] = str(exc)
        return pd.DataFrame(), pd.DataFrame(), diagnostics


def prepare_cluster_matrix(
    model_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, StandardScaler]:
    """
    Prepare numeric risk features for K-means.

    K-means uses Euclidean distance, so only numeric features are used here and every
    selected variable is standardized. Categorical variables such as Region or Manager
    are intentionally not one-hot encoded into the clustering distance because dozens of
    sparse dummies can dominate a small-sample K-means solution.
    """
    cluster_map = {
        "Risk Rating": "Risk Rating",
        "LTV": "LTV",
        "Tenor": "Tenor",
        "Log Exposure": "Log Exposure",
    }
    cols = [cluster_map[name] for name in selected_features]

    data = model_df[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputed = imputer.fit_transform(data)
    scaled = scaler.fit_transform(imputed)

    return data, scaled, imputer, scaler


def cluster_profile_table(
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each K-means segment using original economic units and performance outcomes.

    Performance is not used to create the clusters; it is compared only after clustering.
    That separation allows the dashboard to ask whether naturally occurring risk profiles
    also exhibit different Watchlist / Non-Performing behavior.
    """
    rows = []

    for cluster_name, group in clustered_df.groupby("Cluster", dropna=False):
        exposure = pd.to_numeric(group[BALANCE_COL], errors="coerce").sum()
        high_risk_exposure = pd.to_numeric(
            group.loc[group["Higher Risk"].eq(1), BALANCE_COL],
            errors="coerce",
        ).sum()

        rows.append(
            {
                "Cluster": cluster_name,
                "Loan Count": len(group),
                "Exposure": exposure,
                "Exposure %": exposure / clustered_df[BALANCE_COL].sum()
                if clustered_df[BALANCE_COL].sum()
                else np.nan,
                "Higher-Risk Loans %": group["Higher Risk"].mean(),
                "Higher-Risk Exposure %": high_risk_exposure / exposure
                if exposure
                else np.nan,
                "Avg Risk Rating": pd.to_numeric(
                    group["Risk Rating"], errors="coerce"
                ).mean(),
                "Median LTV": pd.to_numeric(group["LTV"], errors="coerce").median(),
                "Median Tenor": pd.to_numeric(group["Tenor"], errors="coerce").median(),
                "Median Exposure": pd.to_numeric(
                    group[BALANCE_COL], errors="coerce"
                ).median(),
            }
        )

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)




# -----------------------------------------------------------------------------
# Page 1 — Cleaning and manual review
# -----------------------------------------------------------------------------
def render_cleaning_page() -> None:
    st.title("1. Data Cleaning & Manual Review")
    st.caption(
        "Upload the raw Excel extract, compare original and cleaned data, review the "
        "audit log, resolve uncertain records, and download the revised workbook."
    )

    if st.session_state.cleaned_df is not None:
        open_reviews = int(
            st.session_state.cleaned_df["Manual Review Required"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        st.info(
            f"Session analysis is active"
            + (
                f" for **{st.session_state.source_file_name}**"
                if st.session_state.source_file_name
                else ""
            )
            + f". Open manual-review rows: **{open_reviews}**. "
              "You can move between pages without re-uploading or rerunning the cleaning."
        )

    uploaded = st.file_uploader(
        "Upload case-study Excel file",
        type=["xlsx", "xlsm"],
        key="source_workbook_uploader",
        help=(
            "Once a workbook has been loaded, it remains available during this browser "
            "session when you move between Data Cleaning, Portfolio Summary, Performance Analysis, and Workflow."
        ),
    )

    # Persist the workbook independently of the file_uploader widget.  Streamlit may
    # recreate widget state after the user navigates to another sidebar page; session
    # state survives those reruns, so storing the bytes here prevents the manual-review
    # page from appearing empty when the user returns from Portfolio Summary/Workflow.
    if uploaded is not None:
        incoming_bytes = uploaded.getvalue()
        incoming_name = uploaded.name

        # Save the source file immediately. The sheet-level signature below determines
        # whether the analysis must be reset.
        st.session_state.source_file_bytes = incoming_bytes
        st.session_state.source_file_name = incoming_name

    file_bytes = st.session_state.source_file_bytes

    if file_bytes is None:
        st.info("Upload the raw workbook to begin.")
        return

    if uploaded is None and st.session_state.source_file_name:
        st.success(
            f"Using workbook already loaded in this session: "
            f"**{st.session_state.source_file_name}**"
        )

    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    # Prefer the previously selected sheet when returning to this page.
    if (
        st.session_state.selected_sheet is not None
        and st.session_state.selected_sheet in excel_file.sheet_names
    ):
        default_sheet = st.session_state.selected_sheet
    elif DATA_SHEET in excel_file.sheet_names:
        default_sheet = DATA_SHEET
    else:
        default_sheet = excel_file.sheet_names[0]

    default_index = excel_file.sheet_names.index(default_sheet)

    selected_sheet = st.selectbox(
        "Data sheet",
        options=excel_file.sheet_names,
        index=default_index,
        key="source_sheet_selector",
    )

    signature = dataframe_signature(file_bytes, selected_sheet)
    if st.session_state.source_signature != signature:
        # A genuinely new workbook or a different source sheet should start
        # a new analysis. Merely leaving and returning to the page should not.
        reset_analysis_state()
        st.session_state.source_signature = signature
        st.session_state.selected_sheet = selected_sheet

    run_col, note_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("Run automated cleaning", type="primary", use_container_width=True)
    with note_col:
        st.caption(
            "Deterministic issues are corrected automatically. Uncertain issues remain flagged "
            "for manual/source-system review."
        )

    if run_clicked:
        try:
            raw_df = read_uploaded_excel(file_bytes, selected_sheet)
            missing_cols = [c for c in RAW_BUSINESS_COLUMNS if c not in raw_df.columns]
            if missing_cols:
                st.error("Missing required columns: " + ", ".join(missing_cols))
                return
            run_initial_cleaning(raw_df)
            st.success("Cleaning completed. Review the results below.")
        except Exception as exc:
            st.exception(exc)
            return

    if st.session_state.cleaned_df is None:
        st.info("Click **Run automated cleaning** to generate the clean dataset and review flags.")
        return

    cleaned_df = st.session_state.cleaned_df
    current_log = st.session_state.current_log

    unresolved = cleaned_df["Manual Review Required"].fillna(False).astype(bool)
    included = cleaned_df["Include in Analysis"].fillna(False).astype(bool)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Input records", f"{len(st.session_state.raw_df):,}")
    k2.metric("Cleaned records", f"{len(cleaned_df):,}")
    k3.metric("Current validation issues", f"{len(current_log):,}")
    k4.metric("Rows requiring manual review", f"{int(unresolved.sum()):,}")
    k5.metric("Rows included in analysis", f"{int(included.sum()):,}")

    download_bytes = build_download_workbook(
        cleaned_df,
        current_log,
        st.session_state.initial_log,
        st.session_state.manual_edit_log,
        st.session_state.raw_df,
    )
    st.download_button(
        "Download current cleaned Excel",
        data=download_bytes,
        file_name="Loan_Portfolio_Cleaned_Dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Keep the untouched source extract visible next to the cleaned data. This is useful
    # during manual review because the reviewer can compare a flagged value with exactly
    # what was received in the original workbook without reopening the source file.
    tab_original, tab_data, tab_log, tab_review = st.tabs(
        ["Original Data", "Cleaned Data", "Quality / Validation Log", "Manual Review"]
    )

    with tab_original:
        st.caption(
            "Read-only source extract as uploaded. No cleaning or reviewer edits are "
            "applied in this view."
        )
        st.dataframe(st.session_state.raw_df, use_container_width=True, height=520)

    with tab_data:
        st.dataframe(cleaned_df, use_container_width=True, height=520)

    with tab_log:
        log = current_log.copy()
        filter_cols = st.columns(2)
        with filter_cols[0]:
            issue_types = sorted(log["Issue Type"].dropna().astype(str).unique())
            selected_types = st.multiselect(
                "Issue type",
                issue_types,
                default=issue_types,
            )
        with filter_cols[1]:
            review_only = st.checkbox("Manual-review issues only", value=False)

        if selected_types:
            log = log[log["Issue Type"].astype(str).isin(selected_types)]
        if review_only and "Manual Review Required" in log.columns:
            log = log[log["Manual Review Required"].fillna(False).astype(bool)]

        st.dataframe(log, use_container_width=True, height=500)

        if st.session_state.manual_edit_log is not None and not st.session_state.manual_edit_log.empty:
            with st.expander("Manual edit audit trail"):
                st.dataframe(
                    st.session_state.manual_edit_log,
                    use_container_width=True,
                    height=300,
                )

    with tab_review:
        # Keep unresolved records in the queue, but also keep records that are currently
        # excluded from analysis even after review. This prevents an excluded duplicate
        # or a verified-but-not-yet-included row from disappearing before the reviewer
        # has a chance to change the inclusion checkbox.
        review_df = cleaned_df[
            cleaned_df["Manual Review Required"].fillna(False).astype(bool)
            | ~cleaned_df["Include in Analysis"].fillna(False).astype(bool)
        ].copy()

        if review_df.empty:
            st.success("No manual-review or analysis-exclusion items remain.")
        else:
            st.markdown(
                "Review or correct the underlying business field(s). "
                "**Include in Analysis** is an independent manual checkbox: leave it unchecked "
                "while the record should be excluded, or check it when you decide the record "
                "should contribute to Portfolio Summary. Reviewer Decision does not automatically "
                "change analytical inclusion."
            )

            display_cols = [
                RECORD_ID_COL,
                "Loan ID",
                "Borrower Name",
                "Country",
                "Region",
                "Industry",
                "Borrower Type",
                "Facility Type",
                "Outstanding Balance",
                "Interest Rate Type",
                "Base Rate Index",
                "Margin (bps)",
                "Fixed Rate (%)",
                "Origination Date",
                "Maturity Date",
                "Stated Tenor (Yrs)",
                "Internal Risk Rating (1-10)",
                "Loan Status",
                "LTV (%)",
                "Relationship Manager",
                "Manual Review Reason",
                "Include in Analysis",
                "Reviewer Decision",
                "Reviewer Note",
            ]
            display_cols = [c for c in display_cols if c in review_df.columns]
            review_df = review_df[display_cols]

            edited = st.data_editor(
                review_df,
                use_container_width=True,
                height=560,
                hide_index=True,
                num_rows="fixed",
                disabled=[RECORD_ID_COL, "Manual Review Reason"],
                column_config={
                    "Include in Analysis": st.column_config.CheckboxColumn(
                        "Include in Analysis",
                        help=(
                            "Authoritative switch for Portfolio Summary. "
                            "Checked = included; unchecked = excluded."
                        ),
                    ),
                    "Reviewer Decision": st.column_config.SelectboxColumn(
                        "Reviewer Decision",
                        options=[
                            "Pending",
                            "Pending verification - exclude",
                            "Corrected - recheck",
                            "Verified - keep as-is",
                            "Confirmed duplicate - exclude",
                        ],
                        required=True,
                    ),
                    "Reviewer Note": st.column_config.TextColumn(
                        "Reviewer Note",
                        help="Optional source-system verification note or rationale.",
                    ),
                    "Outstanding Balance": st.column_config.NumberColumn(
                        format="$%.0f"
                    ),
                    "Margin (bps)": st.column_config.NumberColumn(format="%.0f"),
                    "Fixed Rate (%)": st.column_config.NumberColumn(format="%.2f"),
                    "LTV (%)": st.column_config.NumberColumn(format="%.1f"),
                },
                key="manual_review_editor",
            )

            if st.button("Apply revisions and rerun checks", type="primary"):
                apply_manual_review_edits(edited)
                st.success("Manual revisions applied and validation rerun.")
                st.rerun()

        portfolio_level = current_log[
            current_log["Loan ID"].isna()
            & current_log["Manual Review Required"].fillna(False).astype(bool)
        ]
        if not portfolio_level.empty:
            st.warning(
                "Portfolio-level review item(s) remain and cannot be resolved by editing a single row."
            )
            st.dataframe(portfolio_level, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 2 — Portfolio summary requested by the case study
# -----------------------------------------------------------------------------
def render_summary_page() -> None:
    st.title("2. Portfolio Summary")
    st.caption(
        "Case-study summary: portfolio composition, rate mix, risk profile, notable concentrations, and assumptions."
    )
    # The summary reads the latest session-state dataset, so manual corrections become
    # visible here after the reviewer clicks "Apply revisions and rerun checks".

    if st.session_state.cleaned_df is None:
        st.info("Go to **Data Cleaning & Manual Review**, upload the workbook, and run cleaning first.")
        return

    full_df = st.session_state.cleaned_df.copy()

    # Portfolio Summary has one authoritative population rule:
    # only rows explicitly marked Include in Analysis = TRUE are summarized.
    # Manual Review Required is intentionally NOT used as the summary filter because
    # review status and analytical inclusion are separate controls.
    include_mask = full_df["Include in Analysis"].fillna(False).astype(bool)
    df = full_df.loc[include_mask].copy()

    excluded_df = full_df.loc[~include_mask].copy()
    open_review = full_df["Manual Review Required"].fillna(False).astype(bool)

    if df.empty:
        st.warning(
            "No records are currently marked Include in Analysis = TRUE. "
            "Return to Data Cleaning / Manual Review and select records for analysis."
        )
        return

    included_exposure = df[BALANCE_COL].sum()
    excluded_exposure = excluded_df[BALANCE_COL].sum()
    st.caption(
        f"Summary population: {len(df):,} included row(s); "
        f"{len(excluded_df):,} excluded row(s). "
        f"Excluded exposure: {format_currency(excluded_exposure)}."
    )

    total_exposure = df[BALANCE_COL].sum()
    floating_exposure = df.loc[
        df["Interest Rate Type"].astype("string").str.lower().eq("floating"), BALANCE_COL
    ].sum()
    performing_exposure = df.loc[
        df["Loan Status"].astype("string").eq("Performing"), BALANCE_COL
    ].sum()

    top_borrowers = top_borrower_summary(df, top_n=10)
    top_borrower_share = top_borrowers.iloc[0]["Exposure %"] if not top_borrowers.empty else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Loans", f"{len(df):,}")
    c2.metric("Total exposure", format_currency(total_exposure))
    c3.metric(
        "Floating exposure",
        f"{floating_exposure / total_exposure:.1%}" if total_exposure else "—",
    )
    c4.metric(
        "Performing exposure",
        f"{performing_exposure / total_exposure:.1%}" if total_exposure else "—",
    )
    c5.metric("Largest borrower", f"{top_borrower_share:.1%}" if pd.notna(top_borrower_share) else "—")

    st.subheader("Portfolio Composition")
    comp_tabs = st.tabs(["Geography", "Industry", "Borrower Type", "Facility Type"])
    composition_specs = [
        ("Region", "Exposure by geography"),
        ("Industry", "Exposure by industry"),
        ("Borrower Type", "Exposure by borrower type"),
        ("Facility Type", "Exposure by facility type"),
    ]

    for tab, (col, title) in zip(comp_tabs, composition_specs):
        with tab:
            summary = exposure_summary(df, col)
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(exposure_bar(summary, col, title), use_container_width=True)
            with right:
                display = summary.copy()
                display["Exposure"] = display["Exposure"].map(format_currency)
                display["Exposure %"] = display["Exposure %"].map(lambda x: f"{x:.1%}")
                st.dataframe(display, hide_index=True, use_container_width=True)

    st.subheader("Rate Mix")
    rate_summary = exposure_summary(df, "Interest Rate Type")
    rate_fig = px.pie(
        rate_summary,
        names="Interest Rate Type",
        values="Exposure",
        hole=0.55,
        title="Fixed vs. floating exposure",
        hover_data={"Loan_Count": True, "Exposure %": ":.1%"},
    )
    st.plotly_chart(rate_fig, use_container_width=True)

    st.subheader("Risk Profile")
    risk_left, risk_right = st.columns(2)

    with risk_left:
        rating = risk_rating_summary(df)
        rating["Exposure % Label"] = rating["Exposure %"].map(lambda x: f"{x:.1%}")
        fig = px.bar(
            rating,
            x="Risk Rating",
            y="Exposure",
            text="Exposure % Label",
            title="Internal risk rating distribution",
            hover_data={"Loan_Count": True, "Exposure %": ":.1%"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with risk_right:
        status = exposure_summary(df, "Loan Status")
        fig = px.pie(
            status,
            names="Loan Status",
            values="Exposure",
            hole=0.55,
            title="Performing / Watchlist / Non-Performing",
            hover_data={"Loan_Count": True, "Exposure %": ":.1%"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Notable Concentrations")
    concentration = concentration_snapshot(df)
    concentration_display = concentration.copy()
    if not concentration_display.empty:
        concentration_display["Exposure"] = concentration_display["Exposure"].map(format_currency)
        concentration_display["Exposure %"] = concentration_display["Exposure %"].map(
            lambda x: f"{x:.1%}"
        )

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Largest category by dimension**")
        st.dataframe(concentration_display, hide_index=True, use_container_width=True)

        top5_share = top_borrowers["Exposure %"].head(5).sum() if not top_borrowers.empty else np.nan
        st.metric("Top 5 borrower concentration", f"{top5_share:.1%}" if pd.notna(top5_share) else "—")

    with right:
        fig = exposure_bar(top_borrowers, "Borrower Name", "Top borrower exposures")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Assumptions & Data Treatment")
    st.dataframe(
        assumptions_table(full_df),
        hide_index=True,
        use_container_width=True,
    )

    unresolved_df = full_df[
        full_df["Manual Review Required"].fillna(False).astype(bool)
    ]
    if not unresolved_df.empty:
        unresolved_included = unresolved_df[
            unresolved_df["Include in Analysis"].fillna(False).astype(bool)
        ]
        st.info(
            f"{len(unresolved_df)} row(s) currently remain under manual review; "
            f"{len(unresolved_included)} of them are manually included in this summary. "
            "Use Page 1 to change inclusion or complete review."
        )




# -----------------------------------------------------------------------------
# Page 3 — Performance analysis
# -----------------------------------------------------------------------------
def render_statistical_analysis_page() -> None:
    st.title("3. Performance Analysis")
    st.caption(
        "Exploratory statistical analysis using only rows with "
        "Include in Analysis = TRUE."
    )

    if st.session_state.cleaned_df is None:
        st.info("Go to **Data Cleaning**, upload the workbook, and run cleaning first.")
        return

    full_df = st.session_state.cleaned_df.copy()
    include_mask = full_df["Include in Analysis"].fillna(False).astype(bool)
    df = full_df.loc[include_mask].copy()

    if df.empty:
        st.warning(
            "No records are currently included in analysis. "
            "Return to Data Cleaning / Manual Review and select records for analysis."
        )
        return

    model_df = prepare_statistical_dataset(df)
    if model_df.empty or model_df["Higher Risk"].nunique() < 2:
        st.warning("The included dataset does not have enough outcome variation for modeling.")
        return

    st.info(
        "**Outcome definition:** Higher Risk = Watchlist or Non-Performing; Performing = 0. "
        "Results are exploratory associations, not causal estimates."
    )

    # =====================================================================
    # Panel 1 — Logistic Regression
    # =====================================================================
    st.subheader("Panel 1 — Logistic Regression")

    core_table, core_diag = run_core_logistic_inference(model_df)

    if core_table.empty:
        st.warning(
            "The logistic regression could not be estimated reliably. "
            f"{core_diag.get('error') or ''}"
        )
    else:
        # Keep the output focused on odds ratios and uncertainty.
        # Continuous predictors are standardized, so their ORs represent a 1-SD increase.
        or_plot = core_table.copy()
        or_plot["CI Low Error"] = (
            or_plot["Odds Ratio"] - or_plot["95% CI Lower"]
        )
        or_plot["CI High Error"] = (
            or_plot["95% CI Upper"] - or_plot["Odds Ratio"]
        )

        fig = px.scatter(
            or_plot,
            x="Odds Ratio",
            y="Predictor",
            error_x="CI High Error",
            error_x_minus="CI Low Error",
            title="Odds of Higher-Risk Performance",
        )
        fig.add_vline(x=1.0, line_dash="dash")
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=or_plot["Predictor"].tolist(),
        )
        fig.update_layout(
            xaxis_title="Odds Ratio",
            yaxis_title=None,
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Short conclusion rather than additional technical metrics.
        st.success(
            "**Conclusion:** Risk Rating and LTV show the strongest directional "
            "associations with weaker loan performance, while Tenor shows little "
            "relationship. The estimates are not statistically significant, so the "
            "findings should be interpreted as directional rather than conclusive."
        )

    st.markdown("---")

    # =====================================================================
    # Panel 2 — Clustering
    # =====================================================================
    st.subheader("Panel 2 — Clustering")

    # Use a fixed four-cluster solution for presentation.
    # In the current dataset, k=4 provides nearly the same silhouette quality as the
    # highest-scoring solution while producing more interpretable portfolio segments.
    cluster_features = ["Risk Rating", "LTV", "Tenor", "Log Exposure"]
    _, scaled, _, _ = prepare_cluster_matrix(
        model_df,
        cluster_features,
    )

    if len(model_df) < 8:
        st.warning("Not enough observations for a stable clustering analysis.")
        return

    chosen_k = 4
    kmeans = KMeans(
        n_clusters=chosen_k,
        random_state=42,
        n_init=20,
    )
    labels = kmeans.fit_predict(scaled)

    clustered = model_df.copy()
    clustered["Cluster"] = pd.Series(
        labels,
        index=clustered.index,
    ).map(lambda x: f"Cluster {int(x) + 1}")

    profile = cluster_profile_table(clustered).copy()

    # Add a simple descriptive name so the table is easier to interpret.
    # Names are assigned from the actual profile rather than hard-coded cluster numbers,
    # because K-means cluster labels themselves are arbitrary.
    profile["Segment"] = ""

    weakest_cluster = profile["Avg Risk Rating"].idxmax()
    highest_ltv_cluster = profile["Median LTV"].idxmax()
    largest_cluster = profile["Exposure %"].idxmax()

    for idx in profile.index:
        if idx == weakest_cluster:
            profile.at[idx, "Segment"] = "Weak Rating"
        elif idx == highest_ltv_cluster:
            profile.at[idx, "Segment"] = "High LTV"
        elif idx == largest_cluster:
            profile.at[idx, "Segment"] = "Core Portfolio"
        else:
            profile.at[idx, "Segment"] = "Stronger / Smaller"

    display_profile = profile[
        [
            "Cluster",
            "Segment",
            "Loan Count",
            "Exposure %",
            "Higher-Risk Exposure %",
            "Avg Risk Rating",
            "Median LTV",
            "Median Tenor",
        ]
    ].copy()

    display_profile["Exposure %"] = display_profile["Exposure %"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display_profile["Higher-Risk Exposure %"] = display_profile[
        "Higher-Risk Exposure %"
    ].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    for col in ["Avg Risk Rating", "Median LTV", "Median Tenor"]:
        display_profile[col] = display_profile[col].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )

    st.dataframe(
        display_profile,
        hide_index=True,
        use_container_width=True,
    )

    # Performance-colored 100% stacked bar:
    # show the exposure mix of Performing / Watchlist / Non-Performing within each cluster.
    cluster_mix = (
        clustered.groupby(["Cluster", "Loan Status"], dropna=False)[BALANCE_COL]
        .sum()
        .rename("Exposure")
        .reset_index()
    )

    cluster_totals = (
        clustered.groupby("Cluster", dropna=False)[BALANCE_COL]
        .sum()
        .rename("Cluster Exposure")
        .reset_index()
    )

    cluster_mix = cluster_mix.merge(cluster_totals, on="Cluster", how="left")
    cluster_mix["Exposure Share"] = np.where(
        cluster_mix["Cluster Exposure"] != 0,
        cluster_mix["Exposure"] / cluster_mix["Cluster Exposure"],
        np.nan,
    )

    # Add the descriptive segment name already assigned in the cluster profile.
    cluster_label_map = (
        profile.assign(
            **{
                "Cluster Label": profile["Cluster"] + " — " + profile["Segment"]
            }
        )
        .set_index("Cluster")["Cluster Label"]
        .to_dict()
    )
    cluster_mix["Cluster Label"] = cluster_mix["Cluster"].map(cluster_label_map)

    fig2 = px.bar(
        cluster_mix,
        x="Cluster Label",
        y="Exposure Share",
        color="Loan Status",
        barmode="stack",
        category_orders={
            "Loan Status": ["Performing", "Watchlist", "Non-Performing"],
        },
        title="Loan Status Exposure Mix by Cluster",
        hover_data={
            "Exposure": ":,.0f",
            "Cluster Exposure": ":,.0f",
            "Exposure Share": ":.1%",
        },
    )
    fig2.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
        title="Share of Cluster Exposure",
    )
    fig2.update_xaxes(title=None)
    fig2.update_layout(legend_title_text="Loan Status")
    st.plotly_chart(fig2, use_container_width=True)

    st.success(
        "**Conclusion:** The weak-rating and high-LTV clusters show the highest "
        "higher-risk exposure, suggesting these characteristics deserve closer "
        "portfolio monitoring."
    )

# -----------------------------------------------------------------------------
# Page 4 — Cleaning workflow
# -----------------------------------------------------------------------------
def render_workflow_page() -> None:
    st.title("4. Cleaning Workflow")
    st.caption(
        "End-to-end data-quality workflow used in the case study. "
        "Deterministic issues are corrected automatically; ambiguous issues are flagged "
        "for manual/source-system review."
    )

    workflow_dot = r"""
    digraph CleaningWorkflow {
        rankdir=TB;
        graph [
            bgcolor="transparent",
            pad="0.25",
            nodesep="0.30",
            ranksep="0.42",
            splines=ortho
        ];
        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=11,
            margin="0.16,0.10",
            color="#7A7A7A",
            fillcolor="#F7F7F7"
        ];
        edge [
            fontname="Arial",
            fontsize=10,
            color="#6B7280",
            arrowsize=0.75
        ];

        upload [
            label="Upload Excel\nLoan_Portfolio_Data",
            fillcolor="#E8F1FB",
            color="#4C78A8"
        ];

        validate [
            label="Validate schema & record count\nRequired columns / expected rows",
            fillcolor="#E8F1FB",
            color="#4C78A8"
        ];

        text [
            label="1. Clean text fields\nTrim whitespace / normalize spacing",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        category [
            label="2. Standardize known categories\nUSA → United States\nPerfroming → Performing",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        balance [
            label="3. Clean outstanding balance\nNumeric conversion\nNegative balance assumption + flag",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        rate [
            label="4. Validate rate fields\nFixed vs Floating logic\n525bps → 5.25%",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        rating [
            label="5. Validate risk rating\nOnly 1–10 accepted\nInvalid → missing + flag",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        dates [
            label="6. Validate dates & tenor\nParse dates / implied tenor\nDo not guess invalid dates",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        ltv [
            label="7. Validate LTV\nNegative → missing + flag\n>100% retained + review flag",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        duplicate [
            label="8. Duplicate checks\nDuplicate Loan ID\nPotential duplicate facility",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        outlier [
            label="9. Exposure outlier check\n3×IQR = Q3 + 3×(Q3−Q1)\nReview flag; no automatic deletion",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        fuzzy [
            label="10. RapidFuzz secondary QA\nPossible text / borrower-name anomalies\nFLAG ONLY — no overwrite",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        decision [
            label="Manual Review Required?",
            shape=diamond,
            fillcolor="#F3E8FF",
            color="#8B5FBF"
        ];

        auto_include [
            label="NO\nInclude in Analysis = TRUE\nautomatically",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        manual [
            label="YES\nDefault Include in Analysis = FALSE\nReviewer edits / verifies record",
            fillcolor="#FDECEC",
            color="#C75B5B"
        ];

        recheck [
            label="Rerun validation rules\nUpdate review flags & audit trail\nPreserve manual inclusion choice",
            fillcolor="#FDECEC",
            color="#C75B5B"
        ];

        include_decision [
            label="Include in Analysis?",
            shape=diamond,
            fillcolor="#F3E8FF",
            color="#8B5FBF"
        ];

        final [
            label="TRUE\nIncluded in Portfolio Summary",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        excluded [
            label="FALSE\nRetained in Cleaned_Data\nExcluded from Portfolio Summary",
            fillcolor="#FDECEC",
            color="#C75B5B"
        ];

        outputs [
            label="Outputs\nCleaned_Data\nAnalysis Included / Excluded\nValidation Log\nDownload Excel",
            fillcolor="#E8F1FB",
            color="#4C78A8"
        ];

        upload -> validate;
        validate -> text;
        text -> category;
        category -> balance;
        balance -> rate;
        rate -> rating;
        rating -> dates;
        dates -> ltv;
        ltv -> duplicate;
        duplicate -> outlier;
        outlier -> fuzzy;
        fuzzy -> decision;

        decision -> auto_include [label=" No "];
        auto_include -> final;

        decision -> manual [label=" Yes "];
        manual -> recheck;
        recheck -> include_decision;

        include_decision -> final [label=" Yes "];
        include_decision -> excluded [label=" No "];

        final -> outputs;
        excluded -> outputs;
    }
    """

    st.graphviz_chart(workflow_dot, use_container_width=True)

    st.markdown("### Workflow logic")
    logic_df = pd.DataFrame(
        [
            {
                "Stage": "Automatic correction",
                "Examples": "Whitespace, known category mappings, bps-to-% conversion",
                "Treatment": "Correct when the intended value is deterministic; keep an audit log.",
            },
            {
                "Stage": "Validation + flag",
                "Examples": "Invalid ratings, date logic, LTV, duplicate IDs, exposure outliers",
                "Treatment": "Do not guess. Retain/set missing as appropriate and require review.",
            },
            {
                "Stage": "Secondary text QA",
                "Examples": "RapidFuzz category / borrower-name similarity",
                "Treatment": "Flag only. Never overwrite investment data based on similarity alone.",
            },
            {
                "Stage": "Manual review",
                "Examples": "Corrected value, verified unusual value, confirmed duplicate",
                "Treatment": (
                    "Reviewer records decision/note and manually selects Include in Analysis. "
                    "Validation is rerun, but inclusion is not inferred from review status."
                ),
            },
            {
                "Stage": "Final output",
                "Examples": "Cleaned dataset, included/excluded analysis sets, validation log",
                "Treatment": (
                    "Portfolio Summary uses only Include in Analysis = TRUE; downloadable Excel "
                    "preserves included, excluded, review, and audit records."
                ),
            },
        ]
    )
    st.dataframe(logic_df, hide_index=True, use_container_width=True)

    st.markdown("### Key control principle")
    st.info(
        "**Correct what is deterministic; flag what is uncertain.** "
        "The workflow avoids silently changing credit/investment data when the correct value "
        "cannot be supported by the source extract."
    )

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        open_review = df["Manual Review Required"].fillna(False).astype(bool)
        analysis_included = df["Include in Analysis"].fillna(False).astype(bool)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current records", f"{len(df):,}")
        c2.metric("Open manual review", f"{int(open_review.sum()):,}")
        c3.metric("Included in analysis", f"{int(analysis_included.sum()):,}")
        st.caption(
            "These metrics reflect the workbook currently loaded in the Data Cleaning page."
        )


# -----------------------------------------------------------------------------
# Main navigation
# -----------------------------------------------------------------------------
def main() -> None:
    initialize_state()

    st.sidebar.title("Case Study")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Data Cleaning",
            "Portfolio Summary",
            "Performance Analysis",
            "Workflow",
        ],
    )

    st.sidebar.markdown("---")
    if st.session_state.cleaned_df is not None:
        sidebar_open = int(
            st.session_state.cleaned_df["Manual Review Required"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        sidebar_included = int(
            st.session_state.cleaned_df["Include in Analysis"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        st.sidebar.success(
            f"Workbook loaded\n\n"
            f"{st.session_state.source_file_name or 'Current session'}\n\n"
            f"Open review rows: {sidebar_open}\n\n"
            f"Included in analysis: {sidebar_included}"
        )

    st.sidebar.caption(
        "Deterministic issues are corrected automatically. "
        "Ambiguous issues are flagged for review rather than guessed."
    )

    if page == "Data Cleaning":
        render_cleaning_page()
    elif page == "Portfolio Summary":
        render_summary_page()
    elif page == "Performance Analysis":
        render_statistical_analysis_page()
    else:
        render_workflow_page()


if __name__ == "__main__":
    main()
