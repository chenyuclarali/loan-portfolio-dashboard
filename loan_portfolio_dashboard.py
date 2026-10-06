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
    defaults = {
        "source_file_name": None,
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
    st.session_state.raw_df = None
    st.session_state.cleaned_df = None
    st.session_state.initial_log = None
    st.session_state.current_log = None
    st.session_state.manual_edit_log = pd.DataFrame()


def dataframe_signature(file_bytes: bytes, sheet_name: str) -> tuple[int, int, str]:
    # A lightweight signature is enough to detect a different upload in one session.
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
) -> bytes:
    """Build a downloadable Excel workbook entirely in memory."""
    output = io.BytesIO()

    export_df = cleaned_df.copy()
    # Internal record ID is useful for auditability in the dashboard, but it is not
    # part of the source schema. Keep it at the end of the exported dataset.
    if RECORD_ID_COL in export_df.columns:
        cols = [c for c in export_df.columns if c != RECORD_ID_COL] + [RECORD_ID_COL]
        export_df = export_df[cols]

    manual_review = export_df[
        export_df["Manual Review Required"].fillna(False).astype(bool)
    ].copy()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        export_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        current_log.to_excel(writer, sheet_name="Current_Validation_Log", index=False)
        manual_review.to_excel(writer, sheet_name="Manual_Review", index=False)

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
    working = raw_df.copy()
    working.insert(0, RECORD_ID_COL, np.arange(1, len(working) + 1))

    cleaned_df, quality_log = clean_loan_portfolio(working)
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
    """A reviewer may explicitly accept an unusual value after verification."""
    df = df.copy()
    verified = df["Reviewer Decision"].eq("Verified - keep as-is")
    df.loc[verified, "Manual Review Required"] = False
    df.loc[verified, "Manual Review Reason"] = ""
    return df


def apply_manual_review_edits(edited_review_df: pd.DataFrame) -> None:
    """
    Apply user revisions to the current clean dataset and rerun all checks.

    Why rerun instead of manually clearing flags?
    Because a corrected rating/date/duplicate key should be validated by the same
    business rules that created the flag in the first place.
    """
    current = st.session_state.cleaned_df.copy()
    if current is None or current.empty:
        return

    current_by_id = current.set_index(RECORD_ID_COL, drop=False)
    edited_by_id = edited_review_df.set_index(RECORD_ID_COL, drop=False)

    audit_rows: list[dict] = []
    decision_map: dict[int, tuple[str, str]] = {}

    editable_fields = [c for c in RAW_BUSINESS_COLUMNS if c in current.columns]

    for record_id, revised_row in edited_by_id.iterrows():
        if record_id not in current_by_id.index:
            continue

        decision = str(revised_row.get("Reviewer Decision", "Pending") or "Pending")
        note = str(revised_row.get("Reviewer Note", "") or "")
        decision_map[int(record_id)] = (decision, note)

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

        current_by_id.at[record_id, "Reviewer Decision"] = decision
        current_by_id.at[record_id, "Reviewer Note"] = note

    revised = current_by_id.reset_index(drop=True)

    # Remove derived columns before rerunning the cleaning engine. The record ID and
    # reviewer fields are retained because the cleaner safely carries unknown columns.
    derived_cols = [
        "Implied Tenor (Yrs)",
        "Duplicate Loan ID Flag",
        "Potential Duplicate Facility Flag",
        "Exposure Outlier Flag",
        "Manual Review Required",
        "Manual Review Reason",
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

    for record_id, (decision, note) in decision_map.items():
        mask = revalidated_df[RECORD_ID_COL].eq(record_id)
        revalidated_df.loc[mask, "Reviewer Decision"] = decision
        revalidated_df.loc[mask, "Reviewer Note"] = note

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

    # Corrected records remain subject to the checks. Explicitly verified values may
    # be accepted even if they remain statistical outliers (e.g., >100% LTV or an
    # unusually large exposure).
    revalidated_df = apply_verified_overrides(revalidated_df)

    # Keep the validation log consistent with explicit reviewer acceptance.
    current_log = current_log.copy()
    current_log["Reviewer Decision"] = ""
    current_log["Reviewer Note"] = ""
    for record_id, (decision, note) in decision_map.items():
        if decision != "Verified - keep as-is":
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


def assumptions_table(df: pd.DataFrame, include_open_reviews: bool) -> pd.DataFrame:
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
                    "error and uses the absolute value for analysis, while keeping the item "
                    "open for manual/source-system verification."
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
                    "Duplicate Loan IDs and possible duplicate facilities are retained unless "
                    "the reviewer corrects them; they are flagged rather than automatically removed."
                ),
            },
            {
                "Assumption / Rule": "Large exposure / high LTV",
                "Treatment": (
                    "Statistically unusual exposures and LTV above 100% are retained because "
                    "they can be real. They require verification rather than automatic correction."
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
                "Assumption / Rule": "Open manual-review records",
                "Treatment": (
                    "Open manual-review records are included in the displayed summary."
                    if include_open_reviews
                    else "Open manual-review records are excluded from the displayed summary."
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
        "Upload the raw Excel extract, run the cleaning rules, review the audit log, "
        "resolve uncertain records, and download the revised clean workbook."
    )

    uploaded = st.file_uploader("Upload case-study Excel file", type=["xlsx", "xlsm"])
    if uploaded is None:
        st.info("Upload the raw workbook to begin.")
        return

    file_bytes = uploaded.getvalue()
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    default_index = (
        excel_file.sheet_names.index(DATA_SHEET)
        if DATA_SHEET in excel_file.sheet_names
        else 0
    )

    selected_sheet = st.selectbox(
        "Data sheet",
        options=excel_file.sheet_names,
        index=default_index,
    )

    signature = dataframe_signature(file_bytes, selected_sheet)
    if st.session_state.source_signature != signature:
        reset_analysis_state()
        st.session_state.source_signature = signature
        st.session_state.source_file_name = uploaded.name
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
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Input records", f"{len(st.session_state.raw_df):,}")
    k2.metric("Cleaned records", f"{len(cleaned_df):,}")
    k3.metric("Current validation issues", f"{len(current_log):,}")
    k4.metric("Rows requiring manual review", f"{int(unresolved.sum()):,}")

    download_bytes = build_download_workbook(
        cleaned_df,
        current_log,
        st.session_state.initial_log,
        st.session_state.manual_edit_log,
    )
    st.download_button(
        "Download current cleaned Excel",
        data=download_bytes,
        file_name="Loan_Portfolio_Cleaned_Dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tab_data, tab_log, tab_review = st.tabs(
        ["Cleaned Data", "Quality / Validation Log", "Manual Review"]
    )

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
        review_df = cleaned_df[
            cleaned_df["Manual Review Required"].fillna(False).astype(bool)
        ].copy()

        if review_df.empty:
            st.success("No unresolved row-level manual review flags remain.")
        else:
            st.markdown(
                "Edit the underlying business field(s) where you know the correct value. "
                "For a value that is unusual but verified as correct, choose **Verified - keep as-is**. "
                "Then apply revisions to rerun the validation rules."
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
                    "Reviewer Decision": st.column_config.SelectboxColumn(
                        "Reviewer Decision",
                        options=[
                            "Pending",
                            "Corrected - recheck",
                            "Verified - keep as-is",
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

    if st.session_state.cleaned_df is None:
        st.info("Go to **Data Cleaning & Manual Review**, upload the workbook, and run cleaning first.")
        return

    full_df = st.session_state.cleaned_df.copy()

    open_review = full_df["Manual Review Required"].fillna(False).astype(bool)
    include_open_reviews = st.toggle(
        "Include open manual-review records in summary",
        value=True,
        help=(
            "The base case keeps unresolved records unless the data is unusable. "
            "Turn this off for a sensitivity view excluding open review rows."
        ),
    )

    df = full_df if include_open_reviews else full_df.loc[~open_review].copy()
    if df.empty:
        st.warning("No records remain under the selected summary filter.")
        return

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
        assumptions_table(full_df, include_open_reviews),
        hide_index=True,
        use_container_width=True,
    )

    unresolved_df = full_df[
        full_df["Manual Review Required"].fillna(False).astype(bool)
    ]
    if not unresolved_df.empty:
        st.info(
            f"{len(unresolved_df)} row(s) currently remain under manual review. "
            "Use Page 1 to revise or explicitly verify them."
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
            label="9. Exposure outlier check\n3×IQR review flag\nNo automatic deletion",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        fuzzy [
            label="10. RapidFuzz secondary QA\nPossible text / borrower-name anomalies\nFLAG ONLY — no overwrite",
            fillcolor="#FFF4E5",
            color="#D98C20"
        ];

        decision [
            label="Any unresolved issue?",
            shape=diamond,
            fillcolor="#F3E8FF",
            color="#8B5FBF"
        ];

        manual [
            label="Manual Review Required = TRUE\nReviewer edits value OR verifies keep-as-is",
            fillcolor="#FDECEC",
            color="#C75B5B"
        ];

        recheck [
            label="Rerun all validation rules\nUpdate flags & audit trail",
            fillcolor="#FDECEC",
            color="#C75B5B"
        ];

        final [
            label="Final Cleaned Dataset\nManual Review Required = FALSE",
            fillcolor="#EAF6EC",
            color="#4E9A5F"
        ];

        outputs [
            label="Outputs\nCleaned_Data\nValidation Log\nManual Review\nDownload Excel",
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

        decision -> manual [label=" Yes "];
        manual -> recheck;
        recheck -> decision [label=" Revalidate "];

        decision -> final [label=" No "];
        final -> outputs;
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
                "Examples": "Corrected value or verified unusual value",
                "Treatment": "Reviewer records decision/note; all validation rules are rerun.",
            },
            {
                "Stage": "Final output",
                "Examples": "Cleaned dataset, current validation log, manual-review sheet",
                "Treatment": "Downloadable Excel preserves both clean data and audit trail.",
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
        c1, c2, c3 = st.columns(3)
        c1.metric("Current records", f"{len(df):,}")
        c2.metric("Open manual review", f"{int(open_review.sum()):,}")
        c3.metric(
            "Resolved / usable rows",
            f"{int((~open_review).sum()):,}",
        )
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
