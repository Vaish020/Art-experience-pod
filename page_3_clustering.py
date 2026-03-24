import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import CLUSTER_PERSONAS, PRESCRIPTIVE_RULES, get_cluster_name

def run():
    st.header("Clustering — Discover Natural Customer Personas")

    if "df_clean" not in st.session_state:
        st.warning("Please upload data on the Data Hub page first.")
        return
    if "km_model" not in st.session_state:
        st.warning("Please train models on the Data Hub page first.")
        return

    km = st.session_state["km_model"]
    sil_scores = st.session_state["km_sil_scores"]
    best_k = st.session_state["km_best_k"]
    cluster_labels = st.session_state["cluster_labels"]
    X_pca = st.session_state["X_pca"]
    df = st.session_state.get("df_clustered", st.session_state["df_clean"]).copy()
    db_labels = st.session_state.get("db_labels", None)

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal K (clusters)", best_k)
    k2.metric("Best silhouette score", f"{max(sil_scores.values()):.3f}",
              help="Ranges 0–1. Above 0.4 = meaningful structure.")
    k3.metric("Total respondents clustered", len(cluster_labels))
    if db_labels is not None:
        noise = int((db_labels == -1).sum())
        k4.metric("DBSCAN noise points", noise,
                  help="Points that don't fit any cluster (potential true outliers)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Elbow & Silhouette", "Cluster Visualisation", "Persona Cards", "DBSCAN"
    ])

    # ── Tab 1: Elbow + Silhouette ────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            sil_df = pd.DataFrame({
                "K": list(sil_scores.keys()),
                "Silhouette score": list(sil_scores.values())
            })
            fig = px.line(sil_df, x="K", y="Silhouette score",
                          markers=True, title="Silhouette score by K",
                          color_discrete_sequence=["#7F77DD"])
            fig.add_vline(x=best_k, line_dash="dash", line_color="coral",
                          annotation_text=f"Optimal K={best_k}")
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)

        with c2:
            from sklearn.cluster import KMeans
            scaler = st.session_state["km_scaler"]
            clust_imp = st.session_state["clust_imp"]
            clust_features = st.session_state["clust_features"]
            df_feat = df[clust_features].copy()
            X_imp = clust_imp.transform(df_feat)
            X_sc = scaler.transform(X_imp)

            inertias = {}
            for k in range(2, 10):
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_tmp.fit(X_sc)
                inertias[k] = km_tmp.inertia_

            ine_df = pd.DataFrame({"K": list(inertias.keys()), "Inertia": list(inertias.values())})
            fig2 = px.line(ine_df, x="K", y="Inertia",
                           markers=True, title="Elbow curve (inertia by K)",
                           color_discrete_sequence=["#1D9E75"])
            fig2.add_vline(x=best_k, line_dash="dash", line_color="coral",
                           annotation_text=f"Optimal K={best_k}")
            fig2.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig2)

        st.info(
            f"Optimal K = **{best_k}** selected by maximum silhouette score "
            f"({max(sil_scores.values()):.3f}). "
            "A score above 0.35 indicates meaningful, non-arbitrary cluster structure."
        )

    # ── Tab 2: PCA scatter ───────────────────────────────────────────────────
    with tab2:
        cluster_names_map = {i: get_cluster_name(i) for i in range(best_k)}
        cluster_name_labels = [cluster_names_map.get(c, f"Cluster {c}") for c in cluster_labels]

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": cluster_name_labels,
            "Annual spend": df.get("annual_spend_midpoint", pd.Series(0, index=df.index)).fillna(0).values,
        })

        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            size="Annual spend", size_max=18,
            title="Customer clusters in PCA space (bubble = annual spend)",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(t=50, b=10), height=500)
        st.plotly_chart(fig)

        # Cluster size distribution
        cluster_counts = pd.Series(cluster_name_labels).value_counts()
        fig2 = px.pie(names=cluster_counts.index, values=cluster_counts.values,
                      title="Cluster size distribution",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig2)

        # Mean spend per cluster
        if "annual_spend_midpoint" in df.columns:
            spend_df = pca_df.groupby("Cluster")["Annual spend"].median().sort_values()
            fig3 = px.bar(x=spend_df.values, y=spend_df.index,
                          orientation="h",
                          title="Median annual spend by cluster (₹)",
                          labels={"x": "Median spend (₹)", "y": "Cluster"},
                          color=spend_df.values, color_continuous_scale="Purples")
            fig3.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(t=40, b=10))
            st.plotly_chart(fig3)

    # ── Tab 3: Persona cards ─────────────────────────────────────────────────
    with tab3:
        st.markdown("Each cluster has been labelled with a business-meaningful persona name. "
                    "Use these to design targeted marketing campaigns.")

        cluster_names = [get_cluster_name(i) for i in range(best_k)]
        df_c = df.copy()
        df_c["cluster_name"] = [get_cluster_name(c) for c in cluster_labels]

        for cname in cluster_names:
            if cname not in PRESCRIPTIVE_RULES:
                continue
            rules = PRESCRIPTIVE_RULES[cname]
            subset = df_c[df_c["cluster_name"] == cname]
            if len(subset) == 0:
                continue

            with st.expander(f"{cname}  ({len(subset)} respondents)", expanded=False):
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Count", len(subset))
                if "annual_spend_midpoint" in subset.columns:
                    pc2.metric("Median spend (₹)", f"₹{subset['annual_spend_midpoint'].median():,.0f}")
                if "visit_intent_binary" in subset.columns:
                    pc3.metric("Interest rate", f"{subset['visit_intent_binary'].mean()*100:.1f}%")
                if "creative_identity_score" in subset.columns:
                    pc4.metric("Avg creative identity", f"{subset['creative_identity_score'].mean():.1f}")

                st.markdown(f"**Recommended bundle:** {rules['bundle']}")
                st.markdown(f"**Discount strategy:** {rules['discount']}")
                st.markdown(f"**Acquisition channel:** {rules['channel']}")
                st.markdown(f"**Message angle:** *\"{rules['message']}\"*")

                # Top features for this cluster vs rest
                numeric_candidates = [
                    "creative_identity_score", "income_midpoint", "ai_comfort_enc",
                    "leisure_spend_mid", "current_creative_spend_mid",
                    "prod_kids_session", "prod_ai_coaching", "mot_child_dev",
                    "mot_stress_relief", "barrier_price",
                ]
                avail = [c for c in numeric_candidates if c in df_c.columns]
                if avail:
                    cluster_means = subset[avail].mean()
                    overall_means = df_c[avail].mean()
                    diff = ((cluster_means - overall_means) / (overall_means + 1e-9) * 100).round(1)
                    diff_df = diff.sort_values(ascending=False).reset_index()
                    diff_df.columns = ["Feature", "% vs overall mean"]
                    diff_df["Feature"] = diff_df["Feature"].str.replace("_", " ").str.title()
                    colors = ["#1D9E75" if v > 0 else "#E24B4A"
                              for v in diff_df["% vs overall mean"]]
                    fig = go.Figure(go.Bar(
                        x=diff_df["% vs overall mean"],
                        y=diff_df["Feature"],
                        orientation="h",
                        marker_color=colors,
                    ))
                    fig.update_layout(
                        title="Feature deviation from overall mean (%)",
                        xaxis_title="% deviation",
                        margin=dict(t=30, b=10), height=280
                    )
                    st.plotly_chart(fig)

        # Radar chart comparing clusters
        st.subheader("Cluster radar comparison")
        radar_cols = [c for c in [
            "creative_identity_score", "income_midpoint", "leisure_spend_mid",
            "visit_intent_binary", "prod_ai_coaching", "mot_stress_relief",
        ] if c in df_c.columns]
        if radar_cols and len(cluster_names) > 0:
            radar_data = df_c.groupby("cluster_name")[radar_cols].mean()
            # Normalise 0-1
            radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)
            labels = [c.replace("_", " ").title() for c in radar_cols]
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (cname, row) in enumerate(radar_norm.iterrows()):
                vals = row.tolist()
                vals.append(vals[0])
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=labels + [labels[0]],
                    fill="toself", name=cname,
                    line_color=colors[i % len(colors)], opacity=0.7
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Cluster comparison radar (normalised 0–1)",
                margin=dict(t=50, b=10), height=480
            )
            st.plotly_chart(fig)

    # ── Tab 4: DBSCAN ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("DBSCAN — noise and outlier detection")
        st.markdown(
            "DBSCAN identifies respondents who don't belong to any coherent cluster "
            "(label = -1). These are the true mixed-persona or noisy respondents."
        )
        if db_labels is not None:
            db_series = pd.Series(db_labels, name="DBSCAN label")
            db_counts = db_series.value_counts().reset_index()
            db_counts.columns = ["DBSCAN label", "Count"]
            db_counts["Type"] = db_counts["DBSCAN label"].apply(
                lambda x: "Noise / Outlier" if x == -1 else f"Cluster {x}"
            )

            pca_df2 = pd.DataFrame({
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "DBSCAN": db_counts.set_index("DBSCAN label").reindex(db_labels)["Type"].values
                if False else [f"Cluster {l}" if l >= 0 else "Noise" for l in db_labels],
            })
            fig = px.scatter(
                pca_df2, x="PC1", y="PC2", color="DBSCAN",
                title="DBSCAN cluster assignments (grey = noise)",
                opacity=0.6,
                color_discrete_map={"Noise": "#B4B2A9"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(margin=dict(t=50, b=10), height=450)
            st.plotly_chart(fig)

            noise_pct = (db_labels == -1).sum() / len(db_labels) * 100
            st.metric("Noise proportion", f"{noise_pct:.1f}%")
            st.caption("Noise points may represent persona-leakage respondents, "
                       "social desirability bias, or genuinely hard-to-classify individuals.")
