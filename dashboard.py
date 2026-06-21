import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# Page Config
st.set_page_config(page_title="ParkSight AI", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("ParkSight AI")
st.sidebar.markdown("### AI-Powered Parking Congestion Intelligence Platform")
st.sidebar.markdown("**Detect → Quantify → Predict → Enforce**")

page = st.sidebar.radio("Navigation", ["Executive Summary", "Hotspot Intelligence", "Forecast Center", "Enforcement Command Center"])

# Load Data
@st.cache_data
def load_data():
    base_path = "outputs"
    hotspots = pd.DataFrame()
    station_ranking = pd.DataFrame()
    clean_violations = pd.DataFrame()
    if os.path.exists(f"{base_path}/hotspots.csv"):
        hotspots = pd.read_csv(f"{base_path}/hotspots.csv")
    if os.path.exists(f"{base_path}/station_ranking.csv"):
        station_ranking = pd.read_csv(f"{base_path}/station_ranking.csv")
    return hotspots, station_ranking, clean_violations

hotspots, station_ranking, _ = load_data()

if hotspots.empty:
    st.warning("No precomputed data found in 'outputs/' directory. Please run the pipeline first.")
    st.stop()

# Helper color mapping
def get_color(tier):
    return {"Critical": "red", "High": "orange", "Medium": "gold", "Low": "green"}.get(tier, "blue")

# PAGE 1: Executive Summary
if page == "Executive Summary":
    st.title("Executive Summary")
    
    total_violations = hotspots["violations"].sum()
    total_hotspots = len(hotspots)
    critical_hotspots = len(hotspots[hotspots["risk_tier"] == "Critical"])
    forecasted_violations = hotspots["predicted_violations"].sum()
    
    # Congestion Proxy KPIs
    total_congestion_impact = hotspots["cluster_congestion_proxy"].sum()
    avg_impact_score = hotspots["impact_score"].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Violations Detected", f"{total_violations:,.0f}")
    col2.metric("Total Hotspots", f"{total_hotspots:,}")
    col3.metric("Critical Risk Hotspots", f"{critical_hotspots:,}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Forecasted Violations (Tomorrow)", f"{forecasted_violations:,.0f}")
    col5.metric("Total Congestion Obstruction Proxy", f"{total_congestion_impact:,.0f}")
    col6.metric("Average Impact Score", f"{avg_impact_score:.1f}/100")
    
    st.markdown("---")
    
    # Explainable AI: Impact Scoring System card
    st.subheader("Explainable AI: Impact Scoring System")
    with st.expander("ℹ️ How is the Congestion Impact Score (0-100) Calculated?", expanded=True):
        st.markdown("""
        ParkSight AI uses a multi-factor weighting formula to compute the **Congestion Impact Score** for each hotspot:
        
        *   **35% Traffic Obstruction Proxy**: Raw violation density scaled by vehicle congestion weights, peak hour boosts, and proximity to major junctions.
        *   **20% Peak-Hour Exposure**: Ratio of violations occurring during peak hours (e.g. morning/evening rush hours).
        *   **15% Junction Proximity Risk**: Ratio of violations occurring at or near critical junctions.
        *   **15% Temporal Persistence**: Longevity of the hotspot (number of active days/weeks).
        *   **10% Repeat Offender Ratio**: Share of violations caused by vehicles with history of repeat offenses.
        *   **5% Validation Confidence**: Proportion of violations verified/approved by traffic controllers.
        """)
        
    st.markdown("---")
    st.subheader("Platform Workflow")
    st.markdown("`Detect → Quantify → Predict → Enforce`")
    st.markdown("ParkSight AI answers three core questions for traffic operations:")
    st.markdown("- **Which parking violations actually hurt traffic?** (Quantified via Congestion Obstruction Proxy)")
    st.markdown("- **Where should enforcement go first?** (Prioritized by Risk Tiers)")
    st.markdown("- **What will happen tomorrow?** (Forecasted via XGBoost+LightGBM Ensemble)")

# PAGE 2: Hotspot Intelligence
elif page == "Hotspot Intelligence":
    st.title("Hotspot Intelligence Map")
    
    center_lat = hotspots["center_lat"].mean()
    center_lon = hotspots["center_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
    
    for _, row in hotspots.head(50).iterrows():
        tier = row["risk_tier"]
        color = get_color(tier)
        score = row["impact_score"]
        vol = row["violations"]
        growth = row.get("growth_rate", 0)
        window = row.get("best_window", "Unknown")
        
        # Growth Trend Symbol
        if growth > 0.05:
            trend_icon = "▲ Growing"
        elif growth < -0.05:
            trend_icon = "▼ Shrinking"
        else:
            trend_icon = "▬ Stable"
            
        popup_html = f"""
        <b>Cluster ID:</b> {row['cluster_id']}<br>
        <b>Zone:</b> {row['top_junction']}<br>
        <b>Risk Tier:</b> {tier}<br>
        <b>Impact Score:</b> {score:.1f}/100<br>
        <b>Violations:</b> {vol:,}<br>
        <b>Trend:</b> {trend_icon} ({'+' if growth > 0 else ''}{growth*100:.1f}%)<br>
        <b>Best Patrol Window:</b> {window}<br>
        <b>Action:</b> {row['recommended_resources']}
        """
        
        folium.CircleMarker(
            location=[row["center_lat"], row["center_lon"]],
            radius=max(6, min(20, vol / 500)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Impact: {score:.1f} | Tier: {tier}"
        ).add_to(m)
        
    st_folium(m, width=1200, height=600)

# PAGE 3: Forecast Center
elif page == "Forecast Center":
    st.title("Forecast Center")
    
    col1, col2 = st.columns(2)
    
    # Filter out hotspots without named junctions to provide actionable insights
    named_hotspots = hotspots[hotspots["top_junction"] != "No Junction"].copy()
    
    with col1:
        st.subheader("Top Risk Areas Tomorrow")
        top20 = named_hotspots.sort_values("predicted_violations", ascending=False).head(20).copy()
        top20["expected_increase_pct"] = top20["expected_increase_pct"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
        top20["forecast_confidence"] = top20["forecast_confidence"].apply(lambda x: f"{x:.1f}%")
        
        # Forecast Tomorrow with Confidence Bands (predicted ± margin)
        top20["Forecast Tomorrow"] = top20.apply(
            lambda r: f"{int(r['predicted_violations'])} ± {int(r.get('forecast_margin', 1))}", axis=1
        )
        
        display_df = top20[["top_junction", "Forecast Tomorrow", "forecast_risk", "expected_increase_pct", "forecast_confidence"]].copy()
        display_df.columns = ["Location", "Forecast Tomorrow", "Forecast Risk", "Expected Increase", "Confidence"]
        st.dataframe(display_df, width="stretch")
        
    with col2:
        st.subheader("Top 10 Fastest Growing Hotspots")
        
        # Filter for named hotspots with at least 1 violation in the previous period to show valid trends
        growth_df = named_hotspots[named_hotspots["prev_7_day_violations"] >= 1].sort_values("growth_rate", ascending=False).head(10).copy()
        
        # Real Weekly Growth Numbers
        growth_df["This Week (Real)"] = growth_df["recent_7_day_violations"].astype(int)
        growth_df["Last Week (Real)"] = growth_df["prev_7_day_violations"].astype(int)
        
        # Growth Trend Icons
        def get_trend_icon(g):
            if g > 0.05: return "▲ Growing"
            elif g < -0.05: return "▼ Shrinking"
            return "▬ Stable"
        growth_df["Trend"] = growth_df["growth_rate"].apply(get_trend_icon)
        growth_df["Growth %"] = (growth_df["growth_rate"] * 100).apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
        
        display_growth = growth_df[["top_junction", "Last Week (Real)", "This Week (Real)", "Trend", "Growth %"]].copy()
        display_growth.columns = ["Location", "Last Week Violations", "This Week Violations", "Trend", "Growth %"]
        st.dataframe(display_growth, width="stretch")

# PAGE 4: Enforcement Command Center
elif page == "Enforcement Command Center":
    st.title("Enforcement Command Center")
    
    # Calculate operational requirements per station (Recommended Resource Allocation Engine)
    def extract_officers(score): return 3 if score > 90 else (2 if score > 75 else 1)
    def extract_tow(score): return 2 if score > 90 else (1 if score > 75 else 0)
    
    hotspots["req_officers"] = hotspots["impact_score"].apply(extract_officers)
    hotspots["req_tow"] = hotspots["impact_score"].apply(extract_tow)
    
    ops_ranking = hotspots.groupby("top_police_station").agg(
        forecast_violations=("predicted_violations", "sum"),
        total_officers=("req_officers", "sum"),
        total_tows=("req_tow", "sum"),
        avg_impact=("impact_score", "mean")
    ).reset_index()
    ops_ranking["Resource Requirement"] = ops_ranking.apply(
        lambda r: f"{r['total_officers']} Officers, {r['total_tows']} Tow Vehicles", axis=1
    )
    ops_ranking = ops_ranking.sort_values("forecast_violations", ascending=False).head(10)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Station Ranking")
        display_ops = ops_ranking[["top_police_station", "forecast_violations", "Resource Requirement"]].copy()
        display_ops.columns = ["Police Station", "Forecasted Tomorrow Violations", "Resource Requirement"]
        st.dataframe(display_ops, width="stretch")
            
    with col2:
        st.subheader("Recommended Resource Allocation Engine")
        
        # Exclude "No Junction" to ensure actionable commands
        named_hotspots = hotspots[hotspots["top_junction"] != "No Junction"].copy()
        action_df = named_hotspots.head(15)[["top_junction", "why_ranked", "recommended_resources", "risk_tier", "best_window"]]
        
        for _, row in action_df.iterrows():
            with st.expander(f"{row['top_junction']} [{row['risk_tier']}]"):
                why_split = str(row['why_ranked']).split('\n')
                st.markdown(f"**{why_split[0]}**")
                for w in why_split[1:]:
                    st.markdown(w)
                st.markdown(f"**Recommended Enforcement Window:** `{row.get('best_window', 'N/A')}`")
                st.markdown(f"**Recommended Resources:** `{row['recommended_resources']}`")
