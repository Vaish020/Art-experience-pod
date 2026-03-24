import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import get_arm_basket

def run():
    st.header("Diagnostic Analytics — Why & What Goes Together?")
    if "df_clean" not in st.session_state:
        st.warning("Please upload and load data on the Data Hub page first.")
        return

    df = st.session_state["df_clean"].copy()

    tab1, tab2, tab3 = st.tabs([
        "Association Rule Mining", "Correlation Analysis", "Cross-tab Explorer"
    ])

    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Association Rule Mining")
        st.markdown(
            "Discover which products, activities, art styles and barriers co-occur. "
            "Adjust the sliders to filter rules by statistical strength."
        )

        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            st.error("mlxtend not installed. It will be available after deployment via requirements.txt.")
            return

        col1, col2 = st.columns([1, 2])
        with col1:
            basket_type = st.selectbox("Select basket type", [
                "Product basket",
                "Activity + Product basket",
                "Art style + Product basket",
                "Barrier co-occurrence basket",
            ])
            min_support = st.slider("Min support", 0.02, 0.30, 0.05, 0.01,
                                    help="Minimum fraction of transactions containing the itemset")
            min_confidence = st.slider("Min confidence", 0.10, 0.90, 0.40, 0.05,
                                       help="P(consequent | antecedent)")
            min_lift = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1,
                                 help="How much more likely the rule is vs random chance")
            top_n = st.slider("Show top N rules", 5, 50, 20, 5)

        with col2:
            transactions = get_arm_basket(df, basket_type)

            if not transactions:
                st.warning("No transactions found for this basket type. Check that binary product/activity columns exist.")
            else:
                te = TransactionEncoder()
                te_array = te.fit_transform(transactions)
                te_df = pd.DataFrame(te_array, columns=te.columns_)

                try:
                    freq_items = apriori(te_df, min_support=min_support, use_colnames=True)
                    if len(freq_items) == 0:
                        st.warning(f"No frequent itemsets found at support={min_support:.2f}. Try lowering min support.")
                    else:
                        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
                        rules = rules[rules["confidence"] >= min_confidence]
                        rules = rules.sort_values("lift", ascending=False).head(top_n)

                        if len(rules) == 0:
                            st.warning("No rules meet the current thresholds. Try adjusting the sliders.")
                        else:
                            st.success(f"Found {len(rules)} rules")
                            rules_display = rules[
                                ["antecedents", "consequents", "support", "confidence", "lift"]
                            ].copy()
                            rules_display["antecedents"] = rules_display["antecedents"].apply(
                                lambda x: ", ".join(list(x))
                            )
                            rules_display["consequents"] = rules_display["consequents"].apply(
                                lambda x: ", ".join(list(x))
                            )
                            rules_display["support"] = rules_display["support"].round(3)
                            rules_display["confidence"] = rules_display["confidence"].round(3)
                            rules_display["lift"] = rules_display["lift"].round(2)
                            rules_display.columns = [
                                "Antecedent (IF)", "Consequent (THEN)",
                                "Support", "Confidence", "Lift"
                            ]
                            st.dataframe(rules_display,  hide_index=True)
                except Exception as e:
                    st.error(f"ARM computation error: {e}")
                    return

        # Scatter: confidence vs lift
        st.subheader("Confidence vs Lift scatter (bubble = support)")
        if 'rules' in dir() and len(rules) > 0:
            rules_plot = rules.copy()
            rules_plot["rule_label"] = (
                rules_plot["antecedents"].apply(lambda x: ", ".join(list(x))) + " → " +
                rules_plot["consequents"].apply(lambda x: ", ".join(list(x)))
            )
            fig = px.scatter(
                rules_plot, x="confidence", y="lift",
                size="support", hover_name="rule_label",
                color="lift", color_continuous_scale="Purples",
                labels={"confidence": "Confidence", "lift": "Lift"},
                title="Association rules: confidence vs lift (bubble = support)"
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                          annotation_text="Lift = 1 (random)")
            fig.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig)

        # Network graph of top rules
        st.subheader("Association network (top 15 rules by lift)")
        if 'rules' in dir() and len(rules) > 0:
            try:
                import networkx as nx
                G = nx.DiGraph()
                top_rules = rules.head(15)
                for _, row in top_rules.iterrows():
                    ant = ", ".join(list(row["antecedents"]))
                    con = ", ".join(list(row["consequents"]))
                    G.add_edge(ant, con, weight=row["lift"], conf=row["confidence"])

                pos = nx.spring_layout(G, seed=42, k=2)
                edge_x, edge_y, edge_text = [], [], []
                for u, v, d in G.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                node_x = [pos[n][0] for n in G.nodes()]
                node_y = [pos[n][1] for n in G.nodes()]
                node_text = list(G.nodes())
                degrees = [G.degree(n) for n in G.nodes()]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.8, color="#AFA9EC"),
                    hoverinfo="none", showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    marker=dict(size=[10 + d * 3 for d in degrees],
                                color="#7F77DD", line_width=1, line_color="white"),
                    text=node_text, textposition="top center",
                    textfont=dict(size=9),
                    hoverinfo="text", showlegend=False
                ))
                fig.update_layout(
                    title="Association network",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(t=40, b=10), height=450
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.caption(f"Network graph skipped: {e}")

    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Correlation analysis")

        numeric_cols = [
            "creative_identity_score", "income_midpoint", "ai_comfort_enc",
            "leisure_spend_mid", "current_creative_spend_mid", "kit_wtp_mid",
            "annual_spend_midpoint", "nps_score", "visit_intent_binary",
            "barrier_price", "barrier_identity", "barrier_time",
            "prod_ai_coaching", "prod_art_kit", "prod_kids_session",
            "mot_stress_relief", "mot_child_dev", "mot_skill_learning",
            "conjoint_chose_pod",
        ]
        avail = [c for c in numeric_cols if c in df.columns]

        if avail:
            corr_df = df[avail].dropna()
            corr_matrix = corr_df.corr().round(2)

            fig = px.imshow(
                corr_matrix, text_auto=True,
                color_continuous_scale="RdBu", zmin=-1, zmax=1,
                title="Pearson correlation matrix (key numeric variables)",
                aspect="auto"
            )
            fig.update_layout(margin=dict(t=50, b=10), height=550)
            st.plotly_chart(fig)

        # Feature correlation with target
        st.subheader("Feature correlation with visit intent (binary target)")
        if "visit_intent_binary" in df.columns and avail:
            target_corr = (
                df[avail].dropna()
                .corrwith(df.loc[df[avail].dropna().index, "visit_intent_binary"])
                .drop("visit_intent_binary", errors="ignore")
                .sort_values()
            )
            colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in target_corr.values]
            fig = go.Figure(go.Bar(
                x=target_corr.values,
                y=[c.replace("_", " ").title() for c in target_corr.index],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title="Correlation with visit_intent_binary (green = positive, red = negative)",
                xaxis_title="Pearson r", margin=dict(t=50, b=10), height=450
            )
            st.plotly_chart(fig)

        # Barrier co-occurrence heatmap
        st.subheader("Barrier co-occurrence heatmap")
        barrier_cols = [c for c in df.columns if c.startswith("barrier_")]
        if barrier_cols:
            bar_df = df[barrier_cols].dropna()
            bar_corr = bar_df.corr().round(2)
            bar_corr.index = [c.replace("barrier_", "").replace("_", " ").title()
                               for c in bar_corr.index]
            bar_corr.columns = bar_corr.index
            fig = px.imshow(bar_corr, text_auto=True,
                            color_continuous_scale="Oranges",
                            title="Barrier co-occurrence (correlation)",
                            aspect="auto")
            fig.update_layout(margin=dict(t=50, b=10), height=420)
            st.plotly_chart(fig)

        # Conjoint vs stated WTP comparison
        st.subheader("Conjoint choice vs stated annual spend")
        if "conjoint_choice" in df.columns and "annual_spend_midpoint" in df.columns:
            conj_spend = df.groupby("conjoint_choice")["annual_spend_midpoint"].agg(
                ["mean", "median", "count"]
            ).round(0).reset_index()
            conj_spend.columns = ["Conjoint choice", "Mean spend (₹)", "Median spend (₹)", "Count"]
            st.dataframe(conj_spend,  hide_index=True)

            fig = px.box(df.dropna(subset=["conjoint_choice", "annual_spend_midpoint"]),
                         x="conjoint_choice", y="annual_spend_midpoint",
                         color="conjoint_choice",
                         title="Annual spend distribution by conjoint choice",
                         labels={"annual_spend_midpoint": "Annual spend (₹)",
                                 "conjoint_choice": "Conjoint choice"},
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig)

    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Cross-tab explorer")
        st.markdown("Select any two categorical variables to explore their relationship.")

        cat_cols = [c for c in df.columns
                    if df[c].dtype == object and df[c].nunique() <= 20]
        if len(cat_cols) >= 2:
            cc1, cc2 = st.columns(2)
            var1 = cc1.selectbox("Row variable", cat_cols, index=0)
            var2 = cc2.selectbox("Column variable", cat_cols, index=min(1, len(cat_cols)-1))

            if var1 != var2:
                ct = pd.crosstab(df[var1], df[var2], normalize="index").round(3)
                fig = px.imshow(ct, text_auto=True,
                                color_continuous_scale="Purples",
                                title=f"Cross-tab: {var1} vs {var2} (row-normalised)",
                                aspect="auto")
                fig.update_layout(margin=dict(t=50, b=10), height=400)
                st.plotly_chart(fig)

                # Raw counts
                with st.expander("Raw count cross-tab"):
                    ct_raw = pd.crosstab(df[var1], df[var2])
                    st.dataframe(ct_raw)
