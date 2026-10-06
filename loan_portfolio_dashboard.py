"""
Streamlit dashboard for the loan-portfolio case study.

The dashboard intentionally separates three concerns:
1. Data Cleaning: run deterministic cleaning rules, inspect validation logs, and
   resolve only those exceptions that require source-system or reviewer judgment.
2. Portfolio Summary: summarize only rows explicitly marked ``Include in Analysis = TRUE``
   using exposure-weighted statistics. Review status and analytical inclusion are separate.
3. Workflow: explain the control framework and the distinction between automatic
   corrections, flag-only checks, and human review.

State is stored in ``st.session_state`` so the uploaded workbook, cleaned data, manual
review decisions, and audit log survive navigation between the three pages during the
current browser session.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
            "session when you move between Data Cleaning, Portfolio Summary, and Workflow."
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
# Page 3 — Cleaning workflow
# -----------------------------------------------------------------------------
def render_workflow_page() -> None:
    st.title("3. Cleaning Workflow")
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
    else:
        render_workflow_page()


if __name__ == "__main__":
    main()
