import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    engineer_features, get_classification_features, get_regression_features,
    get_cluster_name, assign_priority, compute_ltv_estimate, PRESCRIPTIVE_RULES
)

def run():
    st.header("New Customer Predictor — Upload & Score New Leads")
    st.markdown(
        "Upload any new survey CSV (same column schema) and receive a complete "
        "prediction report: classification, cluster, spend estimate, and marketing playbook."
    )

    if "rf_cls" not in st.session_state or "km_model" not in st.session_state:
        st.warning(
            "Models are not yet trained. Please go to **Data Hub**, upload the base CSV, "
            "and click **Train / retrain all models** first."
        )
        return

    # ── File upload or use preloaded ─────────────────────────────────────────
    if "df_new" in st.session_state:
        df_new = st.session_state["df_new"].copy()
        st.success(f"Using pre-loaded new batch: {len(df_new):,} respondents from Data Hub upload.")
        if st.button("Clear and upload a different file"):
            del st.session_state["df_new"]
            st.rerun()
    else:
        uploaded = st.file_uploader(
            "Upload new respondents CSV (same schema as base survey)",
            type=["csv"], key="new_pred_upload"
        )
        if not uploaded:
            st.info("Upload a CSV to begin scoring.")
            _show_template_info()
            return
        try:
            df_new = pd.read_csv(uploaded)
            df_new = engineer_features(df_new)
            st.success(f"Loaded {len(df_new):,} new respondents")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    # ── Score ────────────────────────────────────────────────────────────────
    with st.spinner("Classifying, clustering and predicting spend…"):

        # --- Classification ---
        rf_cls = st.session_state["rf_cls"]
        X_cls, _, cls_feat_names, cls_imp = get_classification_features(df_new)

        km = st.session_state["km_model"]
        scaler = st.session_state["km_scaler"]
        clust_imp = st.session_state["clust_imp"]
        clust_features = st.session_state["clust_features"]

        avail_clust = [c for c in clust_features if c in df_new.columns]
        X_clust_new = df_new[avail_clust].copy() if avail_clust else pd.DataFrame()

        if len(X_clust_new) > 0 and len(avail_clust) == len(clust_features):
            from sklearn.impute import SimpleImputer
            X_cl_imp = clust_imp.transform(X_clust_new)
            X_cl_sc = scaler.transform(X_cl_imp)
            cluster_labels_new = km.predict(X_cl_sc)
        else:
            cluster_labels_new = np.zeros(len(df_new), dtype=int)

        X_cls["cluster_id"] = cluster_labels_new[:len(X_cls)]
        proba_new = rf_cls.predict_proba(X_cls)[:, 1]

        # --- Regression ---
        gbm_reg = st.session_state["gbm_reg"]
        X_reg, _, reg_feat_names, reg_imp = get_regression_features(df_new)
        X_reg["cluster_id"] = cluster_labels_new[:len(X_reg)]
        spend_new = np.clip(gbm_reg.predict(X_reg), 0, None)

        # --- Assemble results ---
        results = df_new.copy().iloc[:len(proba_new)]
        results["interest_probability"] = proba_new.round(3)
        results["predicted_annual_spend"] = spend_new[:len(results)].round(0)
        results["cluster_id"] = cluster_labels_new[:len(results)]
        results["cluster_name"] = [get_cluster_name(c) for c in results["cluster_id"]]
        results["priority_tier"] = [
            assign_priority(p, s)
            for p, s in zip(proba_new, spend_new[:len(results)])
        ]
        results["ltv_3yr_estimate"] = [
            compute_ltv_estimate(cn, sp)
            for cn, sp in zip(results["cluster_name"], results["predicted_annual_spend"])
        ]
        results["recommended_bundle"] = results["cluster_name"].map(
            {k: v["bundle"] for k, v in PRESCRIPTIVE_RULES.items()}
        )
        results["recommended_discount"] = results["cluster_name"].map(
            {k: v["discount"] for k, v in PRESCRIPTIVE_RULES.items()}
        )
        results["acquisition_channel"] = results["cluster_name"].map(
            {k: v["channel"] for k, v in PRESCRIPTIVE_RULES.items()}
        )
        results["message_angle"] = results["cluster_name"].map(
            {k: v["message"] for k, v in PRESCRIPTIVE_RULES.items()}
        )

    # ── KPI summary ───────────────────────────────────────────────────────────
    st.subheader("Batch summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    hot = (results["priority_tier"] == "HOT").sum()
    warm = (results["priority_tier"] == "WARM").sum()
    k1.metric("Total scored", len(results))
    k2.metric("HOT leads", f"{hot} ({hot/len(results)*100:.0f}%)")
    k3.metric("WARM leads", f"{warm} ({warm/len(results)*100:.0f}%)")
    k4.metric("Avg predicted spend", f"₹{results['predicted_annual_spend'].mean():,.0f}")
    k5.metric("Avg 3-yr LTV", f"₹{results['ltv_3yr_estimate'].mean():,.0f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        tier_counts = results["priority_tier"].value_counts()
        fig = px.pie(names=tier_counts.index, values=tier_counts.values,
                     title="Priority tier split",
                     color=tier_counts.index,
                     color_discrete_map={"HOT": "#E24B4A", "WARM": "#EF9F27", "COLD": "#B4B2A9"})
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig)

    with c2:
        cluster_counts = results["cluster_name"].value_counts()
        fig2 = px.bar(x=cluster_counts.values, y=cluster_counts.index,
                      orientation="h", title="Cluster distribution",
                      color=cluster_counts.values, color_continuous_scale="Purples")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=40, b=10))
        st.plotly_chart(fig2)

    with c3:
        fig3 = px.histogram(results, x="interest_probability", nbins=20,
                            title="Interest probability distribution",
                            color_discrete_sequence=["#1D9E75"])
        fig3.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig3)

    # Scatter: probability vs spend
    fig4 = px.scatter(
        results,
        x="interest_probability", y="predicted_annual_spend",
        color="priority_tier",
        color_discrete_map={"HOT": "#E24B4A", "WARM": "#EF9F27", "COLD": "#B4B2A9"},
        hover_data=["cluster_name"] + (
            ["respondent_id"] if "respondent_id" in results.columns else []
        ),
        title="New leads: interest probability vs predicted spend",
        labels={
            "interest_probability": "P(Interested)",
            "predicted_annual_spend": "Predicted annual spend (₹)"
        },
        opacity=0.7, size_max=12,
    )
    fig4.update_layout(margin=dict(t=50, b=10), height=400)
    st.plotly_chart(fig4)

    # ── Scored results table ──────────────────────────────────────────────────
    st.subheader("Scored respondents — full playbook")
    display_cols = [
        c for c in [
            "respondent_id", "age_group", "city", "occupation",
            "cluster_name", "priority_tier",
            "interest_probability", "predicted_annual_spend", "ltv_3yr_estimate",
            "recommended_bundle", "recommended_discount", "acquisition_channel",
        ] if c in results.columns
    ]
    display_df = results[display_cols].copy()
    display_df["interest_probability"] = (display_df["interest_probability"] * 100).round(1).astype(str) + "%"
    display_df["predicted_annual_spend"] = display_df["predicted_annual_spend"].apply(lambda x: f"₹{x:,.0f}")
    display_df["ltv_3yr_estimate"] = display_df["ltv_3yr_estimate"].apply(lambda x: f"₹{x:,.0f}")
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    # Priority filter
    tier_filter = st.multiselect("Filter by priority tier", ["HOT", "WARM", "COLD"],
                                 default=["HOT", "WARM"])
    mask = results["priority_tier"].isin(tier_filter)
    st.dataframe(display_df[mask.values],  hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("Download scored results")
    export_cols = [
        c for c in [
            "respondent_id", "cluster_name", "priority_tier",
            "interest_probability", "predicted_annual_spend", "ltv_3yr_estimate",
            "recommended_bundle", "recommended_discount", "acquisition_channel", "message_angle",
            "age_group", "city", "occupation", "language_preference",
        ] if c in results.columns
    ]
    export_df = results[export_cols].copy()
    export_df["interest_probability"] = export_df["interest_probability"].round(3)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full scored batch CSV",
        data=csv_bytes,
        file_name="artpod_new_leads_scored.csv",
        mime="text/csv",
        
        type="primary",
    )

    # ── Retrain option ────────────────────────────────────────────────────────
    if len(results) >= 200:
        st.divider()
        st.subheader("Merge and retrain (optional)")
        st.markdown(
            f"This batch has {len(results):,} respondents — enough to meaningfully update the models. "
            "Merging will combine the new data with the base 2,000-row dataset and retrain all models."
        )
        if st.button("Merge new batch with base data and retrain all models", type="primary"):
            if "df_clean" in st.session_state:
                df_combined = pd.concat(
                    [st.session_state["df_clean"], df_new], ignore_index=True
                )
                st.session_state["df_clean"] = df_combined
                st.success(
                    f"Datasets merged: {len(df_combined):,} total rows. "
                    "Go to Data Hub and click 'Train / retrain all models' to update."
                )


def _show_template_info():
    st.markdown("### Expected CSV column schema")
    st.markdown(
        "Your new respondent CSV must have at least these columns for scoring to work. "
        "Missing columns are imputed with the training-set median/mode."
    )
    required = [
        "age_group", "income_band", "creative_identity_score",
        "ai_comfort", "city", "occupation",
        "leisure_spend_band", "current_creative_spend_band",
        "visit_freq_intent", "session_duration_pref",
        "barrier_price", "barrier_identity", "barrier_time",
        "prod_ai_coaching", "prod_art_kit", "prod_kids_session",
        "mot_stress_relief", "mot_child_dev",
    ]
    st.code(", ".join(required))
    st.caption(
        "All binary product/barrier/motivation columns (prod_*, barrier_*, mot_*, act_*, style_*) "
        "should be 0 or 1. Categorical columns should match the original survey option labels exactly."
    )
