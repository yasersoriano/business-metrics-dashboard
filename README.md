cd ~/business-metrics-dashboard
cat > app.py << 'EOF'
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Business Metrics Dashboard", layout="wide")

DATA_PATH = "data/events.csv"


# ----------------------------
# Demo data generator
# ----------------------------
def generate_demo_data(path=DATA_PATH, seed=42, days=120, n_users=1500):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)

    users = np.arange(1, n_users + 1)
    channels = np.array(["organic", "ad", "social", "referral", "email"])

    user_first = start + pd.to_timedelta(rng.integers(0, days, size=n_users), unit="D")
    user_channel = rng.choice(channels, size=n_users, p=[0.35, 0.30, 0.15, 0.12, 0.08])

    rows = []
    for uid, first_ts, ch in zip(users, user_first, user_channel):
        n_sessions = int(rng.poisson(6)) + 1
        session_days = rng.integers(0, days, size=n_sessions)
        session_dates = np.sort(start + pd.to_timedelta(session_days, unit="D"))

        for d in session_dates:
            # session starts
            ts = d + pd.to_timedelta(rng.integers(8, 22), unit="h") + pd.to_timedelta(rng.integers(0, 60), unit="m")
            rows.append([uid, ts, ch, "visit", 0.0])

            # click probability by channel
            p_click = {"organic": 0.10, "ad": 0.16, "social": 0.12, "referral": 0.14, "email": 0.18}[ch]
            clicked = rng.random() < p_click
            if clicked:
                rows.append([uid, ts + pd.Timedelta(minutes=int(rng.integers(1, 20))), ch, "click", 0.0])

            # purchase probability conditional on click
            # (makes a nicer funnel)
            p_purchase = {"organic": 0.04, "ad": 0.05, "social": 0.035, "referral": 0.045, "email": 0.06}[ch]
            if clicked and (rng.random() < p_purchase):
                revenue = float(np.round(rng.gamma(shape=2.0, scale=25.0), 2))  # mean ~50
                rows.append([uid, ts + pd.Timedelta(minutes=int(rng.integers(10, 90))), ch, "purchase", revenue])

    df = pd.DataFrame(rows, columns=["user_id", "event_time", "channel", "event", "revenue"])
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["date"] = df["event_time"].dt.date
    df["week"] = df["event_time"].dt.to_period("W").dt.start_time.dt.date
    df["month"] = df["event_time"].dt.to_period("M").dt.start_time.dt.date

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


@st.cache_data
def load_data(path=DATA_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["event_time"] = pd.to_datetime(df["event_time"])
    else:
        df = generate_demo_data(path=path)
    df["date"] = pd.to_datetime(df["event_time"]).dt.date
    df["week"] = pd.to_datetime(df["event_time"]).dt.to_period("W").dt.start_time.dt.date
    df["month"] = pd.to_datetime(df["event_time"]).dt.to_period("M").dt.start_time.dt.date
    return df


def mau(df):
    return df[df["event"] == "visit"].groupby("month")["user_id"].nunique().rename("MAU").reset_index()


def wau(df):
    return df[df["event"] == "visit"].groupby("week")["user_id"].nunique().rename("WAU").reset_index()


def dau(df):
    return df[df["event"] == "visit"].groupby("date")["user_id"].nunique().rename("DAU").reset_index()


def funnel(df):
    base = df[df["event"].isin(["visit", "click", "purchase"])]
    counts = base.groupby("event")["user_id"].nunique().reindex(["visit", "click", "purchase"]).fillna(0).astype(int)
    out = pd.DataFrame({"step": counts.index, "users": counts.values})
    out["conversion_from_prev"] = out["users"].div(out["users"].shift(1)).fillna(1.0)
    out["conversion_from_start"] = out["users"].div(out["users"].iloc[0]).fillna(0.0)
    return out


def cohort_retention(df):
    visits = df[df["event"] == "visit"][["user_id", "event_time"]].copy()
    visits["month"] = visits["event_time"].dt.to_period("M").dt.to_timestamp()

    first = visits.groupby("user_id")["month"].min().rename("cohort_month")
    visits = visits.join(first, on="user_id")

    visits["cohort_index"] = (
        (visits["month"].dt.year - visits["cohort_month"].dt.year) * 12
        + (visits["month"].dt.month - visits["cohort_month"].dt.month)
    )

    cohort_pivot = (
        visits.groupby(["cohort_month", "cohort_index"])["user_id"]
        .nunique()
        .reset_index()
        .pivot(index="cohort_month", columns="cohort_index", values="user_id")
        .fillna(0)
    )

    cohort_size = cohort_pivot[0].replace(0, np.nan)
    retention = cohort_pivot.div(cohort_size, axis=0).fillna(0.0)
    retention.index = retention.index.dt.strftime("%Y-%m")
    return retention


def main():
    st.title("📊 Business Metrics Dashboard (MAU/WAU + Funnel + Cohorts)")
    st.caption("Demo dataset generado localmente (events.csv) para mostrar métricas típicas de negocio.")

    df = load_data(DATA_PATH)

    st.sidebar.header("Filtros")
    channels = ["all"] + sorted(df["channel"].unique().tolist())
    channel = st.sidebar.selectbox("Channel", channels, index=0)

    min_dt = df["event_time"].min().date()
    max_dt = df["event_time"].max().date()
    d1, d2 = st.sidebar.date_input("Rango de fechas", (min_dt, max_dt))

    filtered = df[(df["event_time"].dt.date >= d1) & (df["event_time"].dt.date <= d2)].copy()
    if channel != "all":
        filtered = filtered[filtered["channel"] == channel]

    # KPIs
    st.subheader("KPI Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    visitors = int(filtered[filtered["event"] == "visit"]["user_id"].nunique())
    buyers = int(filtered[filtered["event"] == "purchase"]["user_id"].nunique())
    revenue = float(filtered.loc[filtered["event"] == "purchase", "revenue"].sum())
    conv = (buyers / visitors) if visitors else 0.0

    c1.metric("Unique visitors", f"{visitors:,}")
    c2.metric("Buyers", f"{buyers:,}")
    c3.metric("Revenue", f"${revenue:,.2f}")
    c4.metric("Visitor→Buyer", f"{conv*100:.2f}%")

    # MAU/WAU/DAU charts
    st.subheader("Active Users")
    tab1, tab2, tab3 = st.tabs(["DAU", "WAU", "MAU"])

    with tab1:
        d_dau = dau(filtered)
        fig = px.line(d_dau, x="date", y="DAU", title="DAU over time")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        d_wau = wau(filtered)
        fig = px.line(d_wau, x="week", y="WAU", title="WAU over time")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        d_mau = mau(filtered)
        fig = px.line(d_mau, x="month", y="MAU", title="MAU over time")
        st.plotly_chart(fig, use_container_width=True)

    # Funnel
    st.subheader("Funnel")
    f = funnel(filtered)
    colA, colB = st.columns([1, 2])
    with colA:
        st.dataframe(
            f.assign(
                conversion_from_prev=(f["conversion_from_prev"] * 100).round(2).astype(str) + "%",
                conversion_from_start=(f["conversion_from_start"] * 100).round(2).astype(str) + "%",
            ),
            use_container_width=True,
        )
    with colB:
        fig = px.funnel(f, x="users", y="step", title="Visit → Click → Purchase")
        st.plotly_chart(fig, use_container_width=True)

    # Cohorts
    st.subheader("Cohort Retention (Monthly)")
    retention = cohort_retention(filtered)
    fig = px.imshow(
        (retention * 100).round(1),
        aspect="auto",
        labels=dict(x="Months since first visit", y="Cohort (YYYY-MM)", color="Retention %"),
        title="Retention heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver datos (sample)"):
        st.dataframe(filtered.head(50), use_container_width=True)

    st.caption("Tip: sube tu propio dataset reemplazando data/events.csv con el mismo esquema: user_id, event_time, channel, event, revenue.")


if __name__ == "__main__":
    main()
EOF
