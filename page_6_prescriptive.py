import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    PRESCRIPTIVE_RULES, get_cluster_name, assign_priority,
    compute_ltv_estimate, get_regression_features
)

def run():
    st.header("Prescriptive Analytics — Marketing Playbook")
    st.markdown(
        "Combines classification probability, predicted spend, cluster identity and ARM rules "
        "into one actionable recommendation per customer segment."
    )

    if "df_clean" not in st.session_state:
        st.warning("Please upload data on the Data Hub page first.")
        return
    if "rf_cls" not in st.session_state or "cluster_labels" not in st.session_state:
        st.warning("Please train models on the Data Hub page first.")
        return

    df = st.session_state.get("df_clustered", st.session_state["df_clean"]).copy()
    rf_cls = st.session_state["rf_cls"]
    gbm_reg = st.session_state["gbm_reg"]
    cluster_labels = st.session_state["cluster_labels"]

    from utils import get_classification_features
    X_cls, y_cls, cls_feat_names, cls_imp = get_classification_features(df)
    X_cls["cluster_id"] = cluster_labels[:len(X_cls)]

    X_reg, y_reg, reg_feat_names, reg_imp = get_regression_features(df)
    if "cluster_id" in df.columns:
        X_reg["cluster_id"] = df["cluster_id"].values[:len(X_reg)]

    proba = rf_cls.predict_proba(X_cls)[:, 1]
    predicted_spend = np.clip(gbm_reg.predict(X_reg), 0, None)

    df = df.iloc[:len(proba)].copy()
    df["interest_probability"] = proba
    df["predicted_annual_spend"] = predicted_spend[:len(df)]
    df["cluster_name"] = [get_cluster_name(c) for c in cluster_labels[:len(df)]]
    df["priority_tier"] = [assign_priority(p, s)
                           for p, s in zip(proba, predicted_spend[:len(df)])]
    df["ltv_3yr_estimate"] = [
        compute_ltv_estimate(cn, sp)
        for cn, sp in zip(df["cluster_name"], df["predicted_annual_spend"])
    ]

    # ── KPI summary ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    hot = (df["priority_tier"] == "HOT").sum()
    warm = (df["priority_tier"] == "WARM").sum()
    cold = (df["priority_tier"] == "COLD").sum()
    k1.metric("HOT leads", f"{hot:,}", help="P(visit)>0.7 AND spend>₹9K")
    k2.metric("WARM leads", f"{warm:,}", help="P(visit)>0.4 OR spend>₹4K")
    k3.metric("COLD leads", f"{cold:,}")
    k4.metric("Avg predicted spend", f"₹{df['predicted_annual_spend'].mean():,.0f}")
    k5.metric("Avg 3-yr LTV estimate", f"₹{df['ltv_3yr_estimate'].mean():,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Segment Strategy Cards", "Priority Leaderboard", "Bundle & Discount Matrix", "LTV Analysis"
    ])

    # ── Tab 1: Segment cards ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Marketing strategy per customer cluster")
        st.caption(
            "Each card below gives a complete, ready-to-execute marketing playbook for one cluster. "
            "Product bundles are derived from ARM rules for that cluster."
        )
        for cname, rules in PRESCRIPTIVE_RULES.items():
            subset = df[df["cluster_name"] == cname]
            if len(subset) == 0:
                continue
            hot_c = (subset["priority_tier"] == "HOT").sum()
            warm_c = (subset["priority_tier"] == "WARM").sum()
            avg_prob = subset["interest_probability"].mean()
            avg_spend = subset["predicted_annual_spend"].mean()
            avg_ltv = subset["ltv_3yr_estimate"].mean()

            with st.expander(f"{cname}  —  {len(subset)} customers  |  {hot_c} HOT · {warm_c} WARM",
                             expanded=False):
                cols = st.columns(4)
                cols[0].metric("Avg interest probability", f"{avg_prob:.0%}")
                cols[1].metric("Avg predicted spend (₹/yr)", f"₹{avg_spend:,.0f}")
                cols[2].metric("Avg 3-yr LTV", f"₹{avg_ltv:,.0f}")
                cols[3].metric("HOT leads", f"{hot_c} ({hot_c/max(len(subset),1)*100:.0f}%)")

                st.markdown("---")
                st.markdown(f"**Recommended bundle:** {rules['bundle']}")
                st.markdown(f"**Discount strategy:** {rules['discount']}")
                st.markdown(f"**Acquisition channel:** {rules['channel']}")
                st.markdown(f"**Message angle:** *\"{rules['message']}\"*")

                # Barrier breakdown for this cluster
                barrier_cols = [c for c in df.columns if c.startswith("barrier_")]
                if barrier_cols:
                    bar_means = subset[barrier_cols].mean().sort_values(ascending=False)
                    bar_labels = [c.replace("barrier_", "").replace("_", " ").title()
                                  for c in bar_means.index]
                    fig = px.bar(x=bar_labels, y=bar_means.values,
                                 title="Top barriers in this segment",
                                 labels={"x": "Barrier", "y": "Proportion"},
                                 color=bar_means.values, color_continuous_scale="Reds")
                    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                      margin=dict(t=30, b=10), height=260)
                    st.plotly_chart(fig)

    # ── Tab 2: Priority leaderboard ───────────────────────────────────────────
    with tab2:
        st.subheader("Top prospect leaderboard")
        top_n_leads = st.slider("Show top N leads", 20, 200, 50, 10)

        leaderboard_cols = [
            "respondent_id", "age_group", "city", "occupation",
            "cluster_name", "priority_tier",
            "interest_probability", "predicted_annual_spend", "ltv_3yr_estimate",
        ]
        avail_lb = [c for c in leaderboard_cols if c in df.columns]
        lb = df[avail_lb].sort_values(
            ["priority_tier", "interest_probability"],
            ascending=[True, False]
        ).head(top_n_leads).copy()

        lb["interest_probability"] = (lb["interest_probability"] * 100).round(1).astype(str) + "%"
        lb["predicted_annual_spend"] = lb["predicted_annual_spend"].apply(lambda x: f"₹{x:,.0f}")
        lb["ltv_3yr_estimate"] = lb["ltv_3yr_estimate"].apply(lambda x: f"₹{x:,.0f}")
        lb.columns = [c.replace("_", " ").title() for c in lb.columns]

        st.dataframe(lb,  hide_index=True)

        # Priority tier distribution
        priority_counts = df["priority_tier"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]
        fig = px.pie(priority_counts, names="Priority", values="Count",
                     title="Lead priority distribution",
                     color="Priority",
                     color_discrete_map={"HOT": "#E24B4A", "WARM": "#EF9F27", "COLD": "#B4B2A9"})
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig)

        # Export
        st.subheader("Export scored leads")
        export_cols = [c for c in [
            "respondent_id", "cluster_name", "priority_tier",
            "interest_probability", "predicted_annual_spend", "ltv_3yr_estimate",
            "age_group", "city", "occupation", "language_preference",
        ] if c in df.columns]
        export_df = df[export_cols].copy()
        export_df["interest_probability"] = export_df["interest_probability"].round(3)
        export_df["predicted_annual_spend"] = export_df["predicted_annual_spend"].round(0)
        export_df["ltv_3yr_estimate"] = export_df["ltv_3yr_estimate"].round(0)

        # Add prescriptive columns
        export_df["recommended_bundle"] = export_df["cluster_name"].map(
            {k: v["bundle"] for k, v in PRESCRIPTIVE_RULES.items()}
        )
        export_df["recommended_discount"] = export_df["cluster_name"].map(
            {k: v["discount"] for k, v in PRESCRIPTIVE_RULES.items()}
        )
        export_df["acquisition_channel"] = export_df["cluster_name"].map(
            {k: v["channel"] for k, v in PRESCRIPTIVE_RULES.items()}
        )

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored leads CSV",
            data=csv_bytes, file_name="artpod_scored_leads.csv",
            mime="text/csv"
        )

    # ── Tab 3: Bundle & Discount Matrix ──────────────────────────────────────
    with tab3:
        st.subheader("Barrier → discount recommendation matrix")
        matrix_data = {
            "Barrier": ["Price too high", "Art not for me", "Don't know what to create",
                        "Fear of embarrassment", "No location nearby",
                        "No free time", "Family won't understand", "Already do art at home"],
            "% of respondents": [
                df.get("barrier_price", pd.Series(0)).mean() * 100,
                df.get("barrier_identity", pd.Series(0)).mean() * 100,
                df.get("barrier_skill", pd.Series(0)).mean() * 100,
                df.get("barrier_embarrass", pd.Series(0)).mean() * 100,
                df.get("barrier_location", pd.Series(0)).mean() * 100,
                df.get("barrier_time", pd.Series(0)).mean() * 100,
                df.get("barrier_family", pd.Series(0)).mean() * 100,
                df.get("barrier_diy", pd.Series(0)).mean() * 100,
            ],
            "Recommended offer": [
                "First session free or ₹99 trial",
                '"No skill needed" — walk-in taster + AI guidance',
                "Mood-based prompt generator demo session",
                "Private/solo pod booking option",
                "Partner with malls / colleges in their area",
                "30-min express session (fits in lunch break)",
                "Family pack — bring everyone, normalise it",
                "Print-on-demand upgrade — make something new at home",
            ],
            "Channel": [
                "Instagram/WhatsApp — price anchor comparison",
                "YouTube Reels — 'anyone can do this' testimonials",
                "App demo / in-pod first screen",
                "Booking confirmation email",
                "Google Maps listing / MagicPin",
                "LinkedIn / corporate email",
                "WhatsApp family group content",
                "Instagram — 'level up your art' angle",
            ],
        }
        matrix_df = pd.DataFrame(matrix_data)
        matrix_df["% of respondents"] = matrix_df["% of respondents"].round(1)
        st.dataframe(matrix_df,  hide_index=True)

        # Cluster × channel allocation
        st.subheader("Cluster × acquisition channel allocation")
        channel_map = {k: v["channel"] for k, v in PRESCRIPTIVE_RULES.items()}
        df["channel"] = df["cluster_name"].map(channel_map)
        ch_counts = df["channel"].value_counts().reset_index()
        ch_counts.columns = ["Channel", "Count"]
        fig = px.bar(ch_counts, x="Count", y="Channel",
                     orientation="h", title="Recommended channel reach",
                     labels={"Count": "Customers", "Channel": ""},
                     color="Count", color_continuous_scale="Teal")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig)

    # ── Tab 4: LTV Analysis ───────────────────────────────────────────────────
    with tab4:
        st.subheader("3-year LTV estimate by cluster")
        ltv_by_cluster = df.groupby("cluster_name").agg(
            count=("ltv_3yr_estimate", "count"),
            mean_ltv=("ltv_3yr_estimate", "mean"),
            median_ltv=("ltv_3yr_estimate", "median"),
            total_ltv=("ltv_3yr_estimate", "sum"),
        ).round(0).sort_values("mean_ltv", ascending=False).reset_index()

        ltv_by_cluster["mean_ltv"] = ltv_by_cluster["mean_ltv"].apply(lambda x: f"₹{x:,.0f}")
        ltv_by_cluster["median_ltv"] = ltv_by_cluster["median_ltv"].apply(lambda x: f"₹{x:,.0f}")
        ltv_by_cluster["total_ltv"] = ltv_by_cluster["total_ltv"].apply(lambda x: f"₹{x:,.0f}")
        ltv_by_cluster.columns = ["Cluster", "Count", "Mean 3yr LTV", "Median 3yr LTV", "Total LTV Pool"]
        st.dataframe(ltv_by_cluster,  hide_index=True)

        st.info(
            "LTV is estimated as: Annual Spend × (1 + Retention + Retention²) over 3 years. "
            "Retention rates are cluster-specific (Weekend Creative Parent = 78%, "
            "Disengaged Observer = 30%). These estimates inform how much to invest in acquiring each segment."
        )

        # LTV scatter vs probability
        fig = px.scatter(
            df.sample(min(500, len(df)), random_state=42),
            x="interest_probability", y="ltv_3yr_estimate",
            color="priority_tier",
            color_discrete_map={"HOT": "#E24B4A", "WARM": "#EF9F27", "COLD": "#B4B2A9"},
            size="predicted_annual_spend", size_max=15,
            title="Lead priority map: interest probability vs 3-yr LTV",
            labels={"interest_probability": "P(Interested)",
                    "ltv_3yr_estimate": "3-yr LTV estimate (₹)"},
            opacity=0.7,
        )
        fig.update_layout(margin=dict(t=50, b=10), height=450)
        st.plotly_chart(fig)
