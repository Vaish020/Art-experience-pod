import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run():
    st.header("Descriptive Analytics — Who Are My Customers?")
    if "df_clean" not in st.session_state:
        st.warning("Please upload and load data on the Data Hub page first.")
        return

    df = st.session_state["df_clean"].copy()

    # ── Sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        cities = ["All"] + sorted(df["city"].dropna().unique().tolist()) if "city" in df.columns else ["All"]
        sel_city = st.selectbox("City", cities)

        ages = ["All"] + sorted(df["age_group"].dropna().unique().tolist()) if "age_group" in df.columns else ["All"]
        sel_age = st.selectbox("Age group", ages)

        genders = ["All"] + sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else ["All"]
        sel_gender = st.selectbox("Gender", genders)

        if "visit_intent_5class" in df.columns:
            intents = ["All"] + sorted(df["visit_intent_5class"].dropna().unique().tolist())
            sel_intent = st.selectbox("Visit intent", intents)
        else:
            sel_intent = "All"

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if sel_city != "All" and "city" in df.columns:
        mask &= df["city"] == sel_city
    if sel_age != "All" and "age_group" in df.columns:
        mask &= df["age_group"] == sel_age
    if sel_gender != "All" and "gender" in df.columns:
        mask &= df["gender"] == sel_gender
    if sel_intent != "All" and "visit_intent_5class" in df.columns:
        mask &= df["visit_intent_5class"] == sel_intent
    df = df[mask]

    # ── KPI cards ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total respondents", f"{len(df):,}")

    if "visit_intent_binary" in df.columns:
        pct_int = df["visit_intent_binary"].mean() * 100
        k2.metric("Interested (binary)", f"{pct_int:.1f}%")

    if "creative_identity_score" in df.columns:
        mean_ci = df["creative_identity_score"].mean()
        k3.metric("Avg creative identity", f"{mean_ci:.1f} / 25")

    if "annual_spend_midpoint" in df.columns:
        mean_spend = df["annual_spend_midpoint"].median()
        k4.metric("Median annual spend (₹)", f"₹{mean_spend:,.0f}")

    if "nps_score" in df.columns:
        mean_nps = df["nps_score"].mean()
        k5.metric("Avg NPS score", f"{mean_nps:.1f} / 10")

    st.divider()

    # ── Row 1: Age + Gender + City ───────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        if "age_group" in df.columns:
            age_order = ["Under 18", "18-24", "25-34", "35-44", "45-55", "55+"]
            age_counts = df["age_group"].value_counts().reindex(
                [a for a in age_order if a in df["age_group"].unique()], fill_value=0
            )
            fig = px.bar(x=age_counts.index, y=age_counts.values,
                         labels={"x": "Age group", "y": "Count"},
                         title="Age group distribution",
                         color=age_counts.values, color_continuous_scale="Purples")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c2:
        if "gender" in df.columns:
            g_counts = df["gender"].value_counts()
            fig = px.pie(names=g_counts.index, values=g_counts.values,
                         title="Gender split",
                         color_discrete_sequence=px.colors.sequential.Purp)
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c3:
        if "city" in df.columns:
            city_counts = df["city"].value_counts()
            fig = px.bar(x=city_counts.values, y=city_counts.index,
                         orientation="h", title="City distribution",
                         labels={"x": "Count", "y": "City"},
                         color=city_counts.values, color_continuous_scale="Teal")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    # ── Row 2: Creative identity + Visit intent ──────────────────────────────
    c4, c5 = st.columns(2)

    with c4:
        if "creative_identity_score" in df.columns:
            fig = px.histogram(df, x="creative_identity_score", nbins=20,
                               title="Creative identity score distribution",
                               labels={"creative_identity_score": "Score (5–25)"},
                               color_discrete_sequence=["#7F77DD"])
            fig.add_vline(x=df["creative_identity_score"].mean(),
                          line_dash="dash", line_color="coral",
                          annotation_text=f"Mean: {df['creative_identity_score'].mean():.1f}")
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c5:
        if "visit_intent_5class" in df.columns:
            intent_order = ["Definitely will visit", "Likely to visit", "Undecided",
                            "Unlikely to visit", "Definitely will not visit"]
            intent_counts = df["visit_intent_5class"].value_counts().reindex(
                [i for i in intent_order if i in df["visit_intent_5class"].unique()], fill_value=0
            )
            colors = ["#1D9E75", "#5DCAA5", "#EF9F27", "#E24B4A", "#A32D2D"]
            fig = px.bar(x=intent_counts.index, y=intent_counts.values,
                         title="Visit intent distribution",
                         labels={"x": "Intent", "y": "Count"},
                         color=intent_counts.index,
                         color_discrete_sequence=colors[:len(intent_counts)])
            fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    # ── Row 3: Product interest heatmap ─────────────────────────────────────
    st.subheader("Product interest heatmap by age group")
    prod_cols = [c for c in df.columns if c.startswith("prod_")]
    if prod_cols and "age_group" in df.columns:
        heatmap_data = df.groupby("age_group")[prod_cols].mean().round(2)
        heatmap_data.columns = [c.replace("prod_", "").replace("_", " ").title()
                                 for c in heatmap_data.columns]
        age_order = ["Under 18", "18-24", "25-34", "35-44", "45-55", "55+"]
        heatmap_data = heatmap_data.reindex(
            [a for a in age_order if a in heatmap_data.index]
        )
        fig = px.imshow(heatmap_data.T, aspect="auto",
                        color_continuous_scale="Purples",
                        labels={"color": "Proportion interested"},
                        title="Proportion interested per product by age group")
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig)

    # ── Row 4: Barriers + Annual spend ──────────────────────────────────────
    c6, c7 = st.columns(2)

    with c6:
        barrier_cols = [c for c in df.columns if c.startswith("barrier_")]
        if barrier_cols:
            bar_means = df[barrier_cols].mean().sort_values(ascending=True)
            bar_labels = [c.replace("barrier_", "").replace("_", " ").title()
                          for c in bar_means.index]
            fig = px.bar(x=bar_means.values, y=bar_labels,
                         orientation="h", title="Barrier frequency (% respondents)",
                         labels={"x": "Proportion", "y": "Barrier"},
                         color=bar_means.values, color_continuous_scale="Reds")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c7:
        if "annual_spend_midpoint" in df.columns:
            fig = px.box(df, y="annual_spend_midpoint",
                         x="visit_intent_5class" if "visit_intent_5class" in df.columns else None,
                         title="Annual spend (₹) by visit intent",
                         labels={"annual_spend_midpoint": "Annual spend (₹)",
                                 "visit_intent_5class": "Intent"},
                         color="visit_intent_5class" if "visit_intent_5class" in df.columns else None,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    # ── Row 5: Language + Income ─────────────────────────────────────────────
    c8, c9 = st.columns(2)

    with c8:
        if "language_preference" in df.columns:
            lang_counts = df["language_preference"].value_counts()
            fig = px.pie(names=lang_counts.index, values=lang_counts.values,
                         title="Language preference for AI coach",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c9:
        if "income_band" in df.columns:
            income_order = ["Below 25K", "25K-50K", "50K-1L", "1L-2L", "Above 2L"]
            inc_counts = df["income_band"].value_counts().reindex(
                [i for i in income_order if i in df["income_band"].unique()], fill_value=0
            )
            fig = px.bar(x=inc_counts.index, y=inc_counts.values,
                         title="Household income distribution",
                         labels={"x": "Income band", "y": "Count"},
                         color=inc_counts.values, color_continuous_scale="Oranges")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    # ── Row 6: Gifting intent + Persona breakdown ────────────────────────────
    c10, c11 = st.columns(2)

    with c10:
        gift_cols = [c for c in df.columns if c.startswith("gifting_")]
        if gift_cols:
            gift_means = df[gift_cols].mean().sort_values(ascending=False)
            gift_labels = [c.replace("gifting_", "").replace("_", " ").title()
                           for c in gift_means.index]
            fig = px.bar(x=gift_labels, y=gift_means.values,
                         title="Gifting intent by occasion",
                         labels={"x": "Occasion", "y": "Proportion"},
                         color=gift_means.values, color_continuous_scale="Teal")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    with c11:
        if "persona_type" in df.columns and "visit_intent_binary" in df.columns:
            persona_intent = df.groupby("persona_type")["visit_intent_binary"].mean().sort_values()
            labels = [p.replace("P1_", "").replace("P2_", "").replace("P3_", "")
                       .replace("P4_", "").replace("P5_", "").replace("P6_", "")
                       .replace("P7_", "").replace("_", " ")
                      for p in persona_intent.index]
            fig = px.bar(x=persona_intent.values, y=labels,
                         orientation="h",
                         title="Interest rate by persona",
                         labels={"x": "% Interested", "y": "Persona"},
                         color=persona_intent.values, color_continuous_scale="Greens")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig)

    # ── AI comfort breakdown ──────────────────────────────────────────────────
    if "ai_comfort" in df.columns:
        st.subheader("AI comfort level distribution")
        ai_counts = df["ai_comfort"].value_counts()
        ai_order = ["Daily user-love it", "Occasional-prefer human",
                    "Curious-not tried", "Sceptical", "Not comfortable at all"]
        ai_counts = ai_counts.reindex([a for a in ai_order if a in ai_counts.index])
        fig = px.bar(x=ai_counts.index, y=ai_counts.values,
                     title="AI comfort level",
                     labels={"x": "Comfort level", "y": "Count"},
                     color=ai_counts.values,
                     color_continuous_scale=px.colors.sequential.Blues_r)
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig)

    # ── Occupation ────────────────────────────────────────────────────────────
    if "occupation" in df.columns:
        c12, c13 = st.columns(2)
        with c12:
            occ_counts = df["occupation"].value_counts()
            fig = px.pie(names=occ_counts.index, values=occ_counts.values,
                         title="Occupation breakdown",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)
        with c13:
            if "annual_spend_midpoint" in df.columns:
                occ_spend = df.groupby("occupation")["annual_spend_midpoint"].median().sort_values()
                fig = px.bar(x=occ_spend.values, y=occ_spend.index,
                             orientation="h",
                             title="Median annual spend by occupation (₹)",
                             labels={"x": "Median spend (₹)", "y": "Occupation"},
                             color=occ_spend.values, color_continuous_scale="Purples")
                fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                  margin=dict(t=40, b=10))
                st.plotly_chart(fig)
