# app.py — CloudMart Resource Tagging & Cost Governance Dashboard (Week 10 Activity)
# -----------------------------------------------------------------------------
# This Streamlit app loads the CloudMart multi-account dataset and walks through
# Task Sets 1–5: EDA, Cost Visibility, Tagging Compliance, Visualizations,
# and a Tag Remediation workflow with before/after comparisons and downloads.
# -----------------------------------------------------------------------------

import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="CloudMart Tagging Governance Dashboard", layout="wide")
st.title("🧮 CloudMart — Resource Tagging & Cost Governance Dashboard")
st.caption("Week 10 Activity • INFO49971 — Cloud Economics")

# -------------------------------
# Helpers
# -------------------------------
TAG_FIELDS = [
    "Department", "Project", "Owner", "CostCenter", "CreatedBy",
    "Environment", "Region", "Service"
]

ALL_TAG_COLUMNS = TAG_FIELDS  # convenience alias
NUMERIC_COST_COL = "MonthlyCostUSD"


def _coerce_blank_to_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return df


def load_dataset(uploaded_file) -> pd.DataFrame:
    import csv
    if uploaded_file is None:
        # Default path fallback (keep the file next to app.py)
        try:
            df = pd.read_csv("cloudmart_multi_account.csv")
        except Exception:
            st.stop()
    else:
        df = pd.read_csv(uploaded_file)

    # --- Handle CSVs where entire rows are wrapped in quotes (seen in lab file) ---
    if df.shape[1] == 1 or ("MonthlyCostUSD" not in df.columns and 'MonthlyCostUSD"' not in df.columns):
        try:
            df = pd.read_csv(
                uploaded_file if uploaded_file is not None else "cloudmart_multi_account.csv",
                engine="python", quoting=csv.QUOTE_NONE
            )
        except Exception:
            pass

    # Clean up any stray quotes in headers like '"AccountID' / 'Tagged"'
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    # Clean up any stray quotes in string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.strip('"')

    # Basic normalization
    if NUMERIC_COST_COL in df.columns:
        df[NUMERIC_COST_COL] = pd.to_numeric(df[NUMERIC_COST_COL], errors="coerce").fillna(0.0)
    else:
        # If the cost column is still missing, try to find a close match (case/underscore differences)
        candidates = [c for c in df.columns if c.lower().replace(" ", "").replace("_", "") in
                      ["monthlycostusd", "costusd", "monthlycost"]]
        if candidates:
            df[NUMERIC_COST_COL] = pd.to_numeric(df[candidates[0]], errors="coerce").fillna(0.0)
        else:
            df[NUMERIC_COST_COL] = 0.0

    # Normalize Tagged column to Yes/No strings
    if "Tagged" in df.columns:
        df["Tagged"] = df["Tagged"].astype(str).str.strip().str.strip('"').str.title()
    elif 'Tagged"' in df.columns:
        df.rename(columns={'Tagged"': 'Tagged'}, inplace=True)
        df["Tagged"] = df["Tagged"].astype(str).str.strip().str.strip('"').str.title()
    else:
        df["Tagged"] = "No"

    df["Tagged"] = df["Tagged"].replace({
        "True": "Yes", "1": "Yes", "Y": "Yes",
        "False": "No", "0": "No", "N": "No"
    })
    df.loc[~df["Tagged"].isin(["Yes", "No"]), "Tagged"] = "No"

    # Clean tag-like columns
    df = _coerce_blank_to_nan(df, ALL_TAG_COLUMNS)

    # Ensure ResourceID exists
    if "ResourceID" not in df.columns:
        df.insert(0, "ResourceID", [f"res-{i:05d}" for i in range(len(df))])

    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st.sidebar.header("🔎 Filters")

    def multiselect_or_all(label, series):
        choices = sorted([x for x in series.dropna().unique()])
        pick = st.sidebar.multiselect(label, options=choices, default=choices)
        return pick or choices

    svc_pick = multiselect_or_all("Service", df["Service"]) if "Service" in df.columns else []
    reg_pick = multiselect_or_all("Region", df["Region"]) if "Region" in df.columns else []
    dep_pick = multiselect_or_all("Department", df["Department"]) if "Department" in df.columns else []
    # NEW: Project filter
    proj_pick = multiselect_or_all("Project", df["Project"]) if "Project" in df.columns else []

    mask = pd.Series(True, index=df.index)
    if "Service" in df.columns:
        mask &= df["Service"].isin(svc_pick)
    if "Region" in df.columns:
        mask &= df["Region"].isin(reg_pick)
    if "Department" in df.columns:
        mask &= df["Department"].isin(dep_pick)
    if "Project" in df.columns:
        mask &= df["Project"].isin(proj_pick)

    return df[mask].copy(), {
        "Service": svc_pick,
        "Region": reg_pick,
        "Department": dep_pick,
        "Project": proj_pick,          # include in active filter pills
    }


def kpi_row(df: pd.DataFrame, label_prefix: str = ""):
    total_resources = len(df)
    tagged_counts = df["Tagged"].value_counts(dropna=False)
    tagged_yes = int(tagged_counts.get("Yes", 0))
    tagged_no = int(tagged_counts.get("No", 0))
    pct_untagged = (tagged_no / total_resources * 100.0) if total_resources else 0.0

    total_cost = float(df[NUMERIC_COST_COL].sum())
    untag_cost = float(df.loc[df["Tagged"] == "No", NUMERIC_COST_COL].sum())
    pct_untag_cost = (untag_cost / total_cost * 100.0) if total_cost else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label_prefix}Total Resources", f"{total_resources:,}")
    c2.metric(f"{label_prefix}Tagged", f"{tagged_yes:,}")
    c3.metric(f"{label_prefix}Untagged", f"{tagged_no:,}", f"{pct_untagged:.1f}%")
    c4.metric(f"{label_prefix}Untagged Cost", f"${untag_cost:,.2f}", f"{pct_untag_cost:.1f}% of spend")


# -------------------------------
# Load Data
# -------------------------------
upload = st.file_uploader("Upload cloudmart_multi_account.csv", type=["csv"])
raw_df = load_dataset(upload)

# Keep a pristine copy for before/after comparisons
if "original_df" not in st.session_state:
    st.session_state.original_df = raw_df.copy(deep=True)

# Working frame (subject to filters and remediation)
df_filtered, active_filters = apply_filters(raw_df)

# -------------------------------
# TABS LAYOUT
# -------------------------------

st.markdown(
    """
    <style>
      .title-banner {background: linear-gradient(90deg,#2E3192,#1BFFFF); padding: 14px 24px; border-radius: 16px; color: white; font-weight: 700;}
      .subtitle {color:#2E3192; font-weight:600; margin-top:4px;}
      .callout {background:#F4F8FF;border:1px solid #DCE8FF;border-radius:12px;padding:12px 16px;margin-bottom:8px}
      .pill {display:inline-block;background:#e8f5e9;border:1px solid #c8e6c9;color:#1b5e20;padding:4px 10px;border-radius:999px;margin-right:6px;margin-bottom:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-banner">CloudMart — Resource Tagging & Cost Governance Dashboard</div>', unsafe_allow_html=True)
st.caption("Week 10 Activity • INFO49971 — Cloud Economics")

# Tabs: Overview/Filters + each Task Set
oview_tab, t1_tab, t2_tab, t3_tab, t4_tab, t5_tab, refl_tab = st.tabs([
    "🏁 Overview & Filters",
    "🧪 Task 1 — EDA",
    "💰 Task 2 — Cost Visibility",
    "🏷️ Task 3 — Tagging Compliance",
    "📊 Task 4 — Visualizations",
    "🛠️ Task 5 — Remediation",
    "📝 Reflection",
])

with oview_tab:
    st.subheader("Filters")
    st.markdown("Use the sidebar to refine Service / Region / Department / Project. Active selections:")
    st.markdown(
        " ".join([f"<span class='pill'>{k}: {', '.join(v) if v else 'All'}</span>" for k, v in active_filters.items()]),
        unsafe_allow_html=True,
    )
    st.divider()
    st.subheader("Key KPIs (current filters)")
    kpi_row(df_filtered)

with t1_tab:
    st.header("Task Set 1 — Data Exploration")
    exp = st.expander("Show EDA details (Tasks 1.1–1.5)", expanded=True)
    with exp:
        st.subheader("1.1 First 5 rows")
        st.dataframe(df_filtered.head())

        st.subheader("1.2 Missing values per column")

        # Treat these as missing everywhere (in addition to real NaN)
        _MISSING_TOKENS = {"", "none", "nan", "na", "null", "-", "n/a"}

        def count_missing_any(col: pd.Series) -> int:
            # True for real NaN
            na_mask = col.isna()
            # Also treat string-y empties/markers as missing
            str_mask = col.astype(str).str.strip().str.lower().isin(_MISSING_TOKENS)
            # If the dtype is numeric, str_mask may mark valid numbers like "0" as not missing; that's OK.
            return (na_mask | str_mask).sum()

        missing_all_cols = pd.Series(
            {c: count_missing_any(df_filtered[c]) for c in df_filtered.columns},
            name="missing_count"
        ).sort_values(ascending=False)

        st.dataframe(missing_all_cols)


        st.subheader("1.3 Columns with most missing values")
        missing_sorted = df_filtered.isnull().sum().sort_values(ascending=False)
        st.write(missing_sorted)

        st.subheader("1.4 Tagged vs Untagged (count)")
        st.write(df_filtered["Tagged"].value_counts(dropna=False))

        st.subheader("1.5 % of resources untagged")
        total = len(df_filtered)
        untag_cnt = int((df_filtered["Tagged"] == "No").sum())
        pct = (untag_cnt / total * 100.0) if total else 0.0
        st.write(f"{pct:.2f}% untagged ({untag_cnt} of {total})")

with t2_tab:
    st.header("Task Set 2 — Cost Visibility")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("2.1 Cost of Tagged vs Untagged")
        by_tag = df_filtered.groupby("Tagged", dropna=False)[NUMERIC_COST_COL].sum().reset_index()
        st.dataframe(by_tag)

    with colB:
        st.subheader("2.2 % of total cost that is untagged")
        total_cost = df_filtered[NUMERIC_COST_COL].sum()
        untag_cost = df_filtered.loc[df_filtered["Tagged"] == "No", NUMERIC_COST_COL].sum()
        pct_untag_cost = (untag_cost / total_cost * 100.0) if total_cost else 0.0
        st.metric("Untagged Cost %", f"{pct_untag_cost:.2f}%")

    st.subheader("2.3 Department with most untagged cost")
    untag_by_dept = (
        df_filtered.loc[df_filtered["Tagged"] == "No"]
        .groupby("Department", dropna=False)[NUMERIC_COST_COL]
        .sum()
        .sort_values(ascending=False)
    )
    st.write(untag_by_dept.head(10))

    colC, colD = st.columns(2)
    with colC:
        st.subheader("2.4 Project with highest total cost")
        by_proj = df_filtered.groupby("Project", dropna=False)[NUMERIC_COST_COL].sum().sort_values(ascending=False)
        st.write(by_proj.head(10))

    with colD:
        st.subheader("2.5 Prod vs Dev/Test — cost & tagging quality")
        if "Environment" in df_filtered.columns:
            env_tag = df_filtered.groupby(["Environment", "Tagged"], dropna=False)[NUMERIC_COST_COL].sum().reset_index()
            st.dataframe(env_tag)
        else:
            st.info("'Environment' column not found.")

with t3_tab:
    st.header("Task Set 3 — Tagging Compliance")
    work = df_filtered.copy()
    for c in ALL_TAG_COLUMNS:
        if c not in work.columns:
            work[c] = np.nan

    work["TagCompletenessScore"] = work[ALL_TAG_COLUMNS].notna().sum(axis=1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("3.2 Top 5 lowest completeness scores")
        st.dataframe(work.sort_values(["TagCompletenessScore", NUMERIC_COST_COL]).head(5)[
            ["ResourceID", "TagCompletenessScore", NUMERIC_COST_COL] + ALL_TAG_COLUMNS
        ])

    with col2:
        st.subheader("3.3 Most frequently missing tag fields")
        missing_counts = work[ALL_TAG_COLUMNS].isna().sum().sort_values(ascending=False)
        st.write(missing_counts)

    st.subheader("3.4 Untagged resources & their costs")
    untagged_only = work.loc[work["Tagged"] == "No"].copy()
    st.dataframe(untagged_only[["ResourceID", NUMERIC_COST_COL] + ALL_TAG_COLUMNS])

    csv_untagged = untagged_only.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download untagged.csv (Task 3.5)", data=csv_untagged, file_name="untagged.csv", mime="text/csv")

with t4_tab:
    st.header("Task Set 4 — Visualizations")

    colv1, colv2 = st.columns(2)
    with colv1:
        st.subheader("4.1 Tagged vs Untagged (Resources)")
        pie_df = df_filtered["Tagged"].value_counts().reset_index()
        pie_df.columns = ["Tagged", "Count"]
        fig_pie = px.pie(pie_df, names="Tagged", values="Count", hole=0.35, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with colv2:
        st.subheader("4.2 Cost per Department by Tagging Status")
        if "Department" in df_filtered.columns:
            dept_tag = df_filtered.groupby(["Department", "Tagged"], dropna=False)[NUMERIC_COST_COL].sum().reset_index()
            fig = px.bar(dept_tag, x="Department", y=NUMERIC_COST_COL, color="Tagged", barmode="group",
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(xaxis_title="Department", yaxis_title="Total Monthly Cost (USD)",
                              margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("'Department' column not found.")

    st.subheader("4.3 Total Cost per Service (Horizontal Bar)")
    if "Service" in df_filtered.columns:
        svc_cost = df_filtered.groupby("Service", dropna=False)[NUMERIC_COST_COL].sum().reset_index()
        fig_svc = px.bar(svc_cost.sort_values(NUMERIC_COST_COL), x=NUMERIC_COST_COL, y="Service", orientation="h",
                         color_discrete_sequence=px.colors.qualitative.Prism)
        fig_svc.update_layout(xaxis_title="Total Monthly Cost (USD)", yaxis_title="Service",
                              margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_svc, use_container_width=True)
    else:
        st.info("'Service' column not found.")

    st.subheader("4.4 Cost by Environment")
    if "Environment" in df_filtered.columns:
        env_cost = df_filtered.groupby("Environment", dropna=False)[NUMERIC_COST_COL].sum().reset_index()
        fig_env = px.bar(env_cost, x="Environment", y=NUMERIC_COST_COL, color="Environment",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_env.update_layout(xaxis_title="Environment", yaxis_title="Total Monthly Cost (USD)",
                              showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_env, use_container_width=True)
    else:
        st.info("'Environment' column not found.")

with t5_tab:
    st.header("Task Set 5 — Tag Remediation Workflow")

    st.markdown("**5.1–5.2 Edit untagged resources to simulate remediation**")
    edit_cols = ["ResourceID", NUMERIC_COST_COL] + ALL_TAG_COLUMNS
    editable = untagged_only[edit_cols].copy()

    st.info("Edit the blank/missing fields below. When done, click 'Apply Remediation'.")
    edited = st.data_editor(editable, num_rows="dynamic", use_container_width=True, key="untag_editor")

    apply = st.button("✅ Apply Remediation")

    if apply:
        # Merge user edits back into the working dataframe (by ResourceID)
        up = edited.set_index("ResourceID")[ALL_TAG_COLUMNS]
        merged = raw_df.copy()

        # Replace values where user provided non-empty entries
        for rid, row in up.iterrows():
            if rid in merged.set_index("ResourceID").index:
                for col in ALL_TAG_COLUMNS:
                    new_val = row[col]
                    if pd.notna(new_val) and str(new_val).strip() != "":
                        merged.loc[merged["ResourceID"] == rid, col] = new_val

        # Recompute Tagged (resource is 'Yes' if all key tags now exist)
        merged = _coerce_blank_to_nan(merged, ALL_TAG_COLUMNS)
        merged["Tagged"] = np.where(merged[ALL_TAG_COLUMNS].notna().all(axis=1), "Yes", "No")

        st.session_state["remediated_df"] = merged
        st.success("Remediation applied. See comparisons below and download updated files.")

    if "remediated_df" in st.session_state:
        st.subheader("5.4 Compare cost visibility — Before vs After")
        before = st.session_state.original_df.copy()
        after = st.session_state.remediated_df.copy()

        colB1, colB2 = st.columns(2)
        with colB1:
            st.markdown("**Before (Original)**")
            kpi_row(before, label_prefix="Before • ")
        with colB2:
            st.markdown("**After (Remediated)**")
            kpi_row(after, label_prefix="After • ")

        st.markdown("**Departments with missing tags (counts, Before vs After)**")
        def dept_missing(df):
            w = df.copy()
            w = _coerce_blank_to_nan(w, ALL_TAG_COLUMNS)
            if "Department" not in w.columns:
                w["Department"] = "(Unknown)"
            w["MissingTagCount"] = w[ALL_TAG_COLUMNS].isna().sum(axis=1)
            return w.groupby("Department")["MissingTagCount"].sum().sort_values(ascending=False)

        d_before = dept_missing(before).rename("Before")
        d_after = dept_missing(after).rename("After")
        compare = pd.concat([d_before, d_after], axis=1)
        st.dataframe(compare)

        buf_after = io.BytesIO(after.to_csv(index=False).encode("utf-8"))
        st.download_button("⬇️ Download remediated.csv", data=buf_after, file_name="remediated.csv", mime="text/csv")

with refl_tab:
    st.subheader("5.5 Reflection (optional)")
    st.text_area(
        "How does improved tagging affect accountability and cost reports? Write a short reflection.",
        help="This is for your lab submission write-up.",
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
**How to run locally**  
1) Install deps: `pip install streamlit pandas numpy plotly`  
2) Place `app.py` and `cloudmart_multi_account.csv` in the same folder.  
3) Run: `streamlit run app.py`
""")
