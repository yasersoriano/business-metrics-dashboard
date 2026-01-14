import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Business Metrics Dashboard", layout="wide")

# -----------------------------
# Demo data
# -----------------------------
@st.cache_data
def generate_data(days=120, n_users=1500, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)

    users = np.arange(1, n_users + 1)
    channels = ["organic", "ad", "social", "referral", "email"]

    rows = []
    for u in users:
        first_day = start + pd.Timedelta(rng.integers(0, days), unit="D")
        channel = rng.choice(channels)
        n_sessions = rng.integers(1, 10)

        for i in range(n_sessions):
            day = first_day + pd.Timedelta(rng.integers(0, 30), unit="D")
            rows.append({
                "user_id": u,
                "date": day,
                "channel": channel,
                "session": 1,
                "signup": rng.random() < 0.7,
                "purchase": rng.random() < 0.2,
                "revenue": rng.choice([0, 0, 0, 20, 50, 100])
            })

    return pd.DataFrame(rows)

df = generate_data()
df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.to_period("W").astype(str)
df["month"] = df["date"].dt.to_period("M").astype(str)

# -----------------------------
# UI
# -----------------------------
st.title("📊 Business Metrics Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAU", df.groupby("month")["user_id"].nunique().iloc[-1])

with col2:
    st.metric("WAU", df.groupby("week")["user_id"].nunique().iloc[-1])

with col3:
    st.metric("Revenue ($)", int(df["revenue"].sum()))

st.divider()

# -----------------------------
# Funnel
# -----------------------------
funnel = {
    "Sessions": len(df),
    "Signups": df["signup"].sum(),
    "Purchases": df["purchase"].sum()
}

funnel_df = pd.DataFrame({
    "Stage": funnel.keys(),
    "Users": funnel.values()
})

fig_funnel = px.funnel(funnel_df, x="Users", y="Stage")
st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# -----------------------------
# Cohort retention
# -----------------------------
cohort = (
    df.groupby(["month", "week"])["user_id"]
    .nunique()
    .reset_index()
    .pivot(index="month", columns="week", values="user_id")
)

fig_cohort = px.imshow(
    cohort.fillna(0),
    aspect="auto",
    title="Cohort Retention (Users)"
)

st.plotly_chart(fig_cohort, use_container_width=True)
