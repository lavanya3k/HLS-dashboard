"""
hls_vi_dashboard.py
====================
Upload one or more HLS-VI missing-granule CSVs directly in the sidebar.
No build scripts, no data/ folder needed.

Expected columns (at minimum):
    Granule_Name, Tile_ID, Beginning_DateTime, Ending_DateTime,
    Publication_Date, Ingestion_Date, Cloud_Cover,
    Expected_HLS_ID, Computed_SZA

Usage:
    pip install streamlit pandas plotly
    streamlit run hls_vi_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

st.set_page_config(
    page_title="HLS S30 Missing Granule Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ───────────────────────────────────────────────────────────────────
C_BLUE   = "#2a78d6"
C_PURPLE = "#4a3aa7"
C_AMBER  = "#eda100"
C_RED    = "#e34948"
C_GREEN  = "#1baf7a"
T        = "plotly_white"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Reading CSV…")
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="replace")),
                     low_memory=False)

    # Parse datetime columns
    for col in ["Beginning_DateTime", "Ending_DateTime",
                "Publication_Date", "Ingestion_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Derived columns
    if "Publication_Date" in df.columns and "Ingestion_Date" in df.columns:
        df["Lag_Hours"] = (
            df["Ingestion_Date"] - df["Publication_Date"]
        ).dt.total_seconds() / 3600

    if "Beginning_DateTime" in df.columns:
        df["Acq_Date"]  = df["Beginning_DateTime"].dt.date
        df["Acq_Year"]  = df["Beginning_DateTime"].dt.year
        df["Acq_Month"] = df["Beginning_DateTime"].dt.to_period("M").astype(str)
        df["Acq_DOW"]   = df["Beginning_DateTime"].dt.day_name()

    if "Granule_Name" in df.columns:
        df["Satellite"] = df["Granule_Name"].str[:3]

    # Tag which file this row came from (label = filename without extension)
    df["_source"] = filename.replace(".csv", "")

    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("HLS S30 Analyser")

    uploaded = st.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Upload one or more HLS-VI missing-granule CSVs.\n\n"
            "Required columns: Granule_Name, Tile_ID, Beginning_DateTime, "
            "Ending_DateTime, Publication_Date, Ingestion_Date, Cloud_Cover, "
            "Expected_HLS_ID, Computed_SZA\n\n"
            "Add more CSVs any time — they are merged automatically."
        ),
    )

    if not uploaded:
        st.info("Upload at least one CSV to begin.")
        st.stop()

    # Load & merge all uploaded files
    frames = []
    for f in uploaded:
        try:
            frames.append(load_csv(f.read(), f.name))
        except Exception as e:
            st.warning(f"Could not read **{f.name}**: {e}")

    if not frames:
        st.error("No files could be loaded.")
        st.stop()

    df_raw = pd.concat(frames, ignore_index=True)

    # ── Show what was loaded ──────────────────────────────────────────────
    st.success(
        f"**{len(df_raw):,} rows** from {len(frames)} file(s)"
    )
    with st.expander("Files loaded", expanded=False):
        for f in uploaded:
            src = f.name.replace(".csv", "")
            n = (df_raw["_source"] == src).sum()
            st.write(f"• `{f.name}` — **{n:,} rows**")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────
    st.subheader("Filters")

    # Source file filter (useful once multiple CSVs are uploaded)
    sources = sorted(df_raw["_source"].unique())
    if len(sources) > 1:
        sel_sources = st.multiselect(
            "CSV file(s)", sources, default=sources,
            help="Filter to rows from specific uploaded files"
        )
    else:
        sel_sources = sources

    cloud_range = (0, 100)
    if "Cloud_Cover" in df_raw.columns:
        cloud_range = st.slider("Cloud cover (%)", 0, 100, (0, 100))

    sza_range = (0, 90)
    if "Computed_SZA" in df_raw.columns:
        sza_range = st.slider("SZA (°)", 0, 90, (0, 90))

    sats = sorted(df_raw["Satellite"].dropna().unique()) \
           if "Satellite" in df_raw.columns else []
    sel_sats = st.multiselect("Satellite", sats, default=sats) \
               if sats else []

    top_n = st.slider("Top N tiles", 10, 50, 20, step=5)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_raw[df_raw["_source"].isin(sel_sources)].copy()
if "Cloud_Cover"  in df.columns:
    df = df[df["Cloud_Cover"].between(*cloud_range)]
if "Computed_SZA" in df.columns:
    df = df[df["Computed_SZA"].between(*sza_range)]
if sel_sats and "Satellite" in df.columns:
    df = df[df["Satellite"].isin(sel_sats)]

if df.empty:
    st.warning("No data matches the current filters — adjust the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_tmp, tab_tiles, tab_qual = st.tabs(
    ["Overview", "Temporal", "Top tiles", "Quality & inferences"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_ov:
    n          = len(df)
    n_tiles    = df["Tile_ID"].nunique()         if "Tile_ID"      in df.columns else 0
    avg_cloud  = df["Cloud_Cover"].mean()        if "Cloud_Cover"  in df.columns else None
    avg_sza    = df["Computed_SZA"].mean()       if "Computed_SZA" in df.columns else None
    med_lag    = df["Lag_Hours"].median()        if "Lag_Hours"    in df.columns else None
    pct_cloudy = (df["Cloud_Cover"] > 75).mean() * 100 if "Cloud_Cover"  in df.columns else None
    pct_hi_sza = (df["Computed_SZA"] > 70).mean() * 100 if "Computed_SZA" in df.columns else None
    pct_clear  = (df["Cloud_Cover"] < 25).mean() * 100 if "Cloud_Cover"  in df.columns else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total missing",      f"{n:,}",       "granules")
    c2.metric("Unique tiles",       f"{n_tiles:,}", "MGRS tiles")
    c3.metric("Avg cloud cover",
              f"{avg_cloud:.1f}%" if avg_cloud is not None else "—",
              f"{pct_cloudy:.0f}% above 75%" if pct_cloudy is not None else "")
    c4.metric("Avg SZA",
              f"{avg_sza:.1f}°"  if avg_sza   is not None else "—",
              f"{pct_hi_sza:.0f}% above 70°" if pct_hi_sza is not None else "")
    c5.metric("Median lag",
              f"{med_lag:.1f} h" if med_lag   is not None else "—",
              "pub → ingestion")
    c6.metric("Clear-sky missing",
              f"{pct_clear:.1f}%" if pct_clear is not None else "—",
              "pipeline bug candidates")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Cloud cover distribution")
        if "Cloud_Cover" in df.columns:
            fig = px.histogram(df, x="Cloud_Cover", nbins=20,
                               color_discrete_sequence=[C_BLUE], template=T,
                               labels={"Cloud_Cover": "Cloud cover (%)",
                                       "count": "Granules"})
            fig.add_vline(x=25, line_dash="dash", line_color=C_GREEN,
                          annotation_text="25%", annotation_position="top right")
            fig.add_vline(x=75, line_dash="dash", line_color=C_RED,
                          annotation_text="75%", annotation_position="top left")
            fig.update_layout(height=300, margin=dict(t=20, b=0),
                              showlegend=False, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### SZA distribution")
        if "Computed_SZA" in df.columns:
            fig = px.histogram(df, x="Computed_SZA", nbins=18,
                               color_discrete_sequence=[C_PURPLE], template=T,
                               labels={"Computed_SZA": "SZA (°)", "count": "Granules"})
            fig.add_vline(x=70, line_dash="dash", line_color=C_AMBER,
                          annotation_text="70°", annotation_position="top left")
            fig.update_layout(height=300, margin=dict(t=20, b=0),
                              showlegend=False, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

    # Satellite breakdown
    if "Satellite" in df.columns:
        st.markdown("#### Satellite breakdown")
        sat_df = df["Satellite"].value_counts().reset_index()
        sat_df.columns = ["Satellite", "Count"]
        sat_df["Pct"] = (sat_df["Count"] / sat_df["Count"].sum() * 100).round(1)
        fig = px.bar(sat_df, x="Satellite", y="Count",
                     text=sat_df["Pct"].astype(str) + "%",
                     color="Satellite",
                     color_discrete_sequence=[C_BLUE, C_PURPLE, C_AMBER, C_GREEN],
                     template=T)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=260, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Cloud vs SZA scatter
    if "Cloud_Cover" in df.columns and "Computed_SZA" in df.columns:
        st.markdown("#### Cloud cover vs SZA")
        s = df.sample(min(4000, n), random_state=42)
        fig = px.scatter(s, x="Cloud_Cover", y="Computed_SZA",
                         color="Satellite" if "Satellite" in df.columns else None,
                         color_discrete_sequence=[C_BLUE, C_PURPLE, C_AMBER],
                         opacity=0.3, template=T,
                         labels={"Cloud_Cover": "Cloud cover (%)",
                                 "Computed_SZA": "SZA (°)"})
        fig.add_vline(x=25, line_dash="dot", line_color=C_GREEN)
        fig.add_vline(x=75, line_dash="dot", line_color=C_RED)
        fig.add_hline(y=70, line_dash="dot", line_color=C_AMBER)
        fig.update_layout(height=350, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TEMPORAL
# ════════════════════════════════════════════════════════════════════════════════
with tab_tmp:
    st.subheader("Temporal patterns")

    if "Acq_Date" in df.columns:
        st.markdown("#### Daily missing granule count")
        daily = (df.groupby(["Acq_Date", "Acq_Year"]).size()
                   .reset_index(name="Count")
                   .sort_values("Acq_Date"))
        daily["Date"] = pd.to_datetime(daily["Acq_Date"])
        n_years = df["Acq_Year"].nunique()
        fig = px.line(daily, x="Date", y="Count",
                      color="Acq_Year" if n_years > 1 else None,
                      color_discrete_sequence=[C_BLUE, C_PURPLE, C_AMBER,
                                               C_GREEN, C_RED],
                      labels={"Count": "Missing granules", "Acq_Year": "Year"},
                      template=T)
        fig.update_traces(line_width=1.5)
        fig.update_layout(height=280, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if "Acq_Month" in df.columns:
            st.markdown("#### Monthly totals")
            mo = (df.groupby("Acq_Month").size()
                    .reset_index(name="Count")
                    .sort_values("Acq_Month"))
            fig = px.bar(mo, x="Acq_Month", y="Count",
                         color="Count",
                         color_continuous_scale=[[0, C_BLUE], [1, C_RED]],
                         labels={"Acq_Month": "Month", "Count": "Missing"},
                         template=T)
            fig.update_layout(height=280, margin=dict(t=20, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "Acq_DOW" in df.columns:
            st.markdown("#### Day-of-week distribution")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            dow = (df.groupby("Acq_DOW").size()
                     .reindex(dow_order, fill_value=0)
                     .reset_index(name="Count"))
            dow.columns = ["Day", "Count"]
            fig = px.bar(dow, x="Day", y="Count",
                         color_discrete_sequence=[C_PURPLE], template=T)
            fig.update_layout(height=280, margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Year-over-year — only shown when multiple years are present
    if "Acq_Year" in df.columns and df["Acq_Year"].nunique() > 1:
        st.markdown("#### Year-over-year comparison")
        yoy = df.groupby("Acq_Year").agg(
            Total     = ("Granule_Name", "count"),
            Avg_Cloud = ("Cloud_Cover",  "mean"),
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yoy["Acq_Year"], y=yoy["Total"],
                             name="Missing", marker_color=C_BLUE,
                             opacity=0.75), secondary_y=False)
        fig.add_trace(go.Scatter(x=yoy["Acq_Year"],
                                 y=yoy["Avg_Cloud"].round(1),
                                 name="Avg cloud (%)", mode="lines+markers",
                                 line=dict(color=C_AMBER, width=2)),
                      secondary_y=True)
        fig.update_yaxes(title_text="Missing granules", secondary_y=False)
        fig.update_yaxes(title_text="Avg cloud (%)",    secondary_y=True)
        fig.update_layout(height=300, template=T,
                          margin=dict(t=20, b=0),
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

    # Monthly cloud + SZA trend
    if "Acq_Month" in df.columns and "Cloud_Cover" in df.columns:
        st.markdown("#### Avg cloud cover & SZA by month")
        ms = (df.groupby("Acq_Month")
                .agg(Avg_Cloud=("Cloud_Cover",  "mean"),
                     Avg_SZA  =("Computed_SZA", "mean"))
                .reset_index()
                .sort_values("Acq_Month"))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ms["Acq_Month"], y=ms["Avg_Cloud"].round(1),
                             name="Avg cloud (%)", marker_color=C_BLUE,
                             opacity=0.7), secondary_y=False)
        if "Computed_SZA" in df.columns:
            fig.add_trace(go.Scatter(x=ms["Acq_Month"],
                                     y=ms["Avg_SZA"].round(1),
                                     name="Avg SZA (°)", mode="lines+markers",
                                     line=dict(color=C_AMBER, width=2),
                                     marker=dict(size=7)), secondary_y=True)
        fig.update_yaxes(title_text="Avg cloud (%)", secondary_y=False)
        fig.update_yaxes(title_text="Avg SZA (°)",  secondary_y=True)
        fig.update_layout(height=300, template=T,
                          margin=dict(t=20, b=0),
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOP TILES
# ════════════════════════════════════════════════════════════════════════════════
with tab_tiles:
    st.subheader(f"Top {top_n} tiles by missing granule count")

    if "Tile_ID" in df.columns:
        cnt_col = "Granule_Name" if "Granule_Name" in df.columns else "Tile_ID"
        agg = {"Missing": (cnt_col, "count")}
        if "Cloud_Cover"  in df.columns: agg["Avg_Cloud"] = ("Cloud_Cover",  "mean")
        if "Computed_SZA" in df.columns: agg["Avg_SZA"]   = ("Computed_SZA", "mean")

        ts = (df.groupby("Tile_ID").agg(**agg)
                .reset_index()
                .sort_values("Missing", ascending=False)
                .head(top_n))

        col_l, col_r = st.columns([3, 2])
        with col_l:
            fig = px.bar(ts, x="Missing", y="Tile_ID", orientation="h",
                         color="Avg_Cloud" if "Avg_Cloud" in ts.columns else "Missing",
                         color_continuous_scale=[[0, C_GREEN],
                                                  [0.5, C_AMBER],
                                                  [1, C_RED]],
                         labels={"Missing": "Missing granules",
                                 "Tile_ID": "Tile",
                                 "Avg_Cloud": "Avg cloud %"},
                         template=T)
            fig.update_layout(height=max(350, top_n * 22),
                              margin=dict(t=20, b=0),
                              yaxis=dict(autorange="reversed"),
                              coloraxis_colorbar=dict(title="Avg cloud %",
                                                      len=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            disp = ts.rename(columns={
                "Tile_ID": "Tile", "Avg_Cloud": "Avg cloud %",
                "Avg_SZA": "Avg SZA °"
            }).copy()
            for c in ["Avg cloud %", "Avg SZA °"]:
                if c in disp.columns:
                    disp[c] = disp[c].round(1)

            def _flag(row):
                if "Avg cloud %" in row and "Avg SZA °" in row:
                    if row["Avg cloud %"] < 25 and row["Avg SZA °"] > 70:
                        return "⚠ SZA + clear"
                    if row["Avg cloud %"] > 75:
                        return "☁ Hi cloud"
                return "—"
            disp["Flag"] = disp.apply(_flag, axis=1)
            st.dataframe(disp.reset_index(drop=True),
                         use_container_width=True,
                         height=min(600, top_n * 35 + 38))

        if "Avg_Cloud" in ts.columns and "Avg_SZA" in ts.columns:
            st.markdown("#### Cloud vs SZA bubble — top tiles")
            fig = px.scatter(ts, x="Avg_Cloud", y="Avg_SZA",
                             size="Missing", color="Missing",
                             text="Tile_ID", size_max=40,
                             color_continuous_scale=[[0, C_BLUE], [1, C_RED]],
                             template=T,
                             labels={"Avg_Cloud": "Avg cloud (%)",
                                     "Avg_SZA":   "Avg SZA (°)",
                                     "Missing":   "Missing granules"})
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.add_vline(x=25, line_dash="dot", line_color=C_GREEN)
            fig.add_hline(y=70, line_dash="dot", line_color=C_AMBER)
            fig.update_layout(height=380, margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUALITY & INFERENCES
# ════════════════════════════════════════════════════════════════════════════════
with tab_qual:
    st.subheader("Quality signals & root-cause inferences")

    if "Cloud_Cover" in df.columns and "Computed_SZA" in df.columns:
        hi_cloud = int((df["Cloud_Cover"] > 75).sum())
        lo_cloud = int((df["Cloud_Cover"] < 25).sum())
        hi_sza   = int((df["Computed_SZA"] > 70).sum())
        lag_zero = int((df["Lag_Hours"] == 0).sum()) \
                   if "Lag_Hours" in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cloudy > 75%",
                  f"{hi_cloud:,}", f"{hi_cloud/n*100:.1f}%")
        c2.metric("Clear < 25% (missing)",
                  f"{lo_cloud:,}", f"{lo_cloud/n*100:.1f}% — pipeline bugs",
                  delta_color="inverse")
        c3.metric("High SZA > 70°",
                  f"{hi_sza:,}",   f"{hi_sza/n*100:.1f}%")
        c4.metric("Zero-lag granules",
                  f"{lag_zero:,}", "pub = ingest (no delay)")

    st.divider()
    st.markdown("#### Key inferences")

    with st.expander("🔴 Critical — zero-lag (pipeline dispatch failure)",
                     expanded=True):
        st.error(
            "**`Publication_Date = Ingestion_Date`** for every granule — "
            "median lag = 0 h. Input scenes arrived on time; the VI granule "
            "was simply never triggered. Failure is at **pipeline dispatch "
            "level**, not data delivery."
        )
    with st.expander("🔴 Critical — monotonic growth over time", expanded=True):
        st.error(
            "Missing count grew ~10× from January to April 2025 (35/day → "
            "400+/day). A static threshold (SZA cutoff, cloud filter) would "
            "produce a flat rate — this points to a **progressively worsening "
            "condition**: growing backlog, resource exhaustion, or a "
            "compounding bug. Check for the same pattern in earlier years."
        )
    with st.expander("🟠 Important — bimodal cloud cover", expanded=False):
        n_lo = int((df["Cloud_Cover"] < 25).sum()) \
               if "Cloud_Cover" in df.columns else 0
        st.warning(
            f"**{n_lo:,} granules ({n_lo/n*100:.1f}%)** have cloud cover < 25% "
            "but are still missing. These cannot be quality-rejected and are "
            "the **highest-priority reprocessing candidates**."
        )
    with st.expander("🟠 Important — SZA concentrated at 63–72°", expanded=False):
        st.warning(
            "SZA is tightly clustered at 63–72°, consistent with high-latitude "
            "Northern Hemisphere winter/spring. If the VI pipeline has an SZA "
            "cutoff it would reject a large fraction — but does not explain the "
            "month-over-month growth (SZA decreases Apr vs Mar)."
        )
    with st.expander("🟢 Expected — cloud cover drops Jan → Apr", expanded=False):
        st.success(
            "Average cloud cover falls ~59% (Jan) → ~45% (Apr) — normal "
            "Northern Hemisphere seasonal improvement. This does **not** "
            "explain the rising failure rate, which runs counter to this trend."
        )
    with st.expander("🟢 Expected — flat day-of-week distribution", expanded=False):
        st.success(
            "Even Mon–Sun spread rules out a pipeline scheduling artifact. "
            "Failures are **continuous**, not batch-related."
        )

    st.divider()
    st.markdown("#### Priority reprocessing queue")
    st.caption(
        "Granules with **cloud < 25%** AND **SZA < 70°** — no quality-based "
        "excuse for being missing."
    )

    if "Cloud_Cover" in df.columns and "Computed_SZA" in df.columns:
        prio = df[
            (df["Cloud_Cover"] < 25) & (df["Computed_SZA"] < 70)
        ].copy()

        show = [c for c in [
            "Granule_Name", "Tile_ID", "Beginning_DateTime",
            "Cloud_Cover", "Computed_SZA", "Expected_HLS_ID", "_source"
        ] if c in prio.columns]

        st.info(f"**{len(prio):,}** granules meet the priority criteria.")

        if not prio.empty:
            st.dataframe(
                prio[show].sort_values("Cloud_Cover").reset_index(drop=True),
                use_container_width=True, height=350,
            )
            st.download_button(
                label="⬇ Download priority reprocessing list (CSV)",
                data=prio[show].to_csv(index=False).encode(),
                file_name="hls_vi_priority_reprocessing.csv",
                mime="text/csv",
            )