import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bus Captain Risk Dashboard", layout="wide")

def load_data():
    return pd.read_csv("driver_predictions.csv")

df = load_data()

st.title("Bus Captain Safety Risk Prediction Dashboard")
st.caption("lord have mercy on us all")
st.sidebar.header("Filter Controls")
selected_tiers = st.sidebar.multiselect(
    "Select Risk Tier",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)
search_id = st.sidebar.text_input("Search Driver ID")

filtered_df = df[df["Risk_Tier"].isin(selected_tiers)]
if search_id.strip():
    filtered_df = filtered_df[filtered_df["Driver_ID"].astype(str).str.contains(search_id.strip())]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Bus Captains", len(filtered_df))
col2.metric("High-Risk Flagged", int((filtered_df["Risk_Tier"] == "High").sum()))
col3.metric("Avg Risk Probability", f"{filtered_df['Risk_Probability'].mean():.1f}%")
col4.metric("Actual Q1 Incident Drivers", int(filtered_df["At_Risk"].sum()))

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Driver Count by Predicted Risk Tier")
    tier_counts = (
        filtered_df["Risk_Tier"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
        .reset_index()
    )
    tier_counts.columns = ["Risk_Tier", "Count"]
    fig_tier = px.bar(
        tier_counts,
        x="Risk_Tier",
        y="Count",
        color="Risk_Tier",
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        text="Count"
    )
    fig_tier.update_layout(showlegend=False)
    st.plotly_chart(fig_tier, use_container_width=True)

with chart_col2:
    st.subheader("Risk Probability vs. Total Alarms")
    fig_scatter = px.scatter(
        filtered_df,
        x="Total_Alarms",
        y="Risk_Probability",
        color="Risk_Tier",
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        hover_data=["Driver_ID", "At_Risk", "Max_Speed"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Driver Risk Profiles")
display_cols = ["Driver_ID", "Risk_Probability", "Risk_Tier", "At_Risk", "Total_Alarms", "Avg_Speed", "Max_Speed"]
other_cols = [c for c in filtered_df.columns if c not in display_cols]
ordered_df = filtered_df[display_cols + other_cols].sort_values(by="Risk_Probability", ascending=False)
st.dataframe(ordered_df, use_container_width=True, height=400)