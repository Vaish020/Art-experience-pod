import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def run():
    st.header("Classification — Will This Customer Visit?")
    st.markdown(
        "Two models predict whether each respondent is likely to visit an Art Experience Pod. "
        "Performance is measured by accuracy, precision, recall, F1-score, and ROC-AUC."
    )

    if "df_clean" not in st.session_state:
        st.warning("Please upload data on the Data Hub page first.")
        return
    if "rf_cls" not in st.session_state:
        st.warning("Please train models on the Data Hub page first.")
        return

    rf = st.session_state["rf_cls"]
    xgb = st.session_state["xgb_cls"]
    X_tr = st.session_state["X_cls_train"]
    X_te = st.session_state["X_cls_test"]
    y_tr = st.session_state["y_cls_train"]
    y_te = st.session_state["y_cls_test"]
    feat_names = st.session_state["cls_feat_names"]
    cv_rf = st.session_state.get("cv_rf_auc", np.array([0.0]))
    cv_xgb = st.session_state.get("cv_xgb_auc", np.array([0.0]))

    # ── Controls ─────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        model_choice = st.selectbox("Select model to inspect",
                                    ["Random Forest", "XGBoost", "Both (compare)"])
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                              help="Lower = more recall (catch more potential visitors). "
                                   "Higher = more precision (fewer false positives).")

    # ── Compute predictions ───────────────────────────────────────────────────
    def get_metrics(model, X, y, thr):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= thr).astype(int)
        return {
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1-score": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, proba),
        }, proba, pred

    rf_metrics, rf_proba, rf_pred = get_metrics(rf, X_te, y_te, threshold)
    xgb_metrics, xgb_proba, xgb_pred = get_metrics(xgb, X_te, y_te, threshold)

    # ── Metric comparison table ──────────────────────────────────────────────
    st.subheader("Model performance metrics (test set)")
    metric_df = pd.DataFrame({
        "Metric": list(rf_metrics.keys()),
        "Random Forest": [f"{v:.3f}" for v in rf_metrics.values()],
        "XGBoost": [f"{v:.3f}" for v in xgb_metrics.values()],
    })
    st.dataframe(metric_df,  hide_index=True)

    # Cross-validation AUC
    cv_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "CV ROC-AUC Mean": [cv_rf.mean(), cv_xgb.mean()],
        "CV ROC-AUC Std": [cv_rf.std(), cv_xgb.std()],
    })
    cv_df["CV ROC-AUC Mean"] = cv_df["CV ROC-AUC Mean"].round(3)
    cv_df["CV ROC-AUC Std"] = cv_df["CV ROC-AUC Std"].round(3)
    st.caption("5-fold cross-validation AUC:")
    st.dataframe(cv_df,  hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix", "ROC Curve", "Feature Importance", "Classification Report"
    ])

    # ── Tab 1: Confusion matrix ───────────────────────────────────────────────
    with tab1:
        models_to_show = []
        if model_choice in ["Random Forest", "Both (compare)"]:
            models_to_show.append(("Random Forest", rf_pred, rf_proba))
        if model_choice in ["XGBoost", "Both (compare)"]:
            models_to_show.append(("XGBoost", xgb_pred, xgb_proba))

        cols = st.columns(len(models_to_show))
        for i, (mname, pred, _) in enumerate(models_to_show):
            with cols[i]:
                cm = confusion_matrix(y_te, pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: Not Interested", "Actual: Interested"],
                    columns=["Pred: Not Interested", "Pred: Interested"]
                )
                fig = px.imshow(
                    cm_df, text_auto=True,
                    color_continuous_scale="Purples",
                    title=f"Confusion Matrix — {mname}",
                    labels={"color": "Count"}
                )
                fig.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(fig)

                tn, fp, fn, tp = cm.ravel()
                st.caption(
                    f"TP={tp} | FP={fp} | FN={fn} | TN={tn}  |  "
                    f"Threshold={threshold:.2f}"
                )

    # ── Tab 2: ROC curve ──────────────────────────────────────────────────────
    with tab2:
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray"))

        colors_roc = {"Random Forest": "#7F77DD", "XGBoost": "#1D9E75"}
        for mname, model in [("Random Forest", rf), ("XGBoost", xgb)]:
            proba = model.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = roc_auc_score(y_te, proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{mname} (AUC={auc:.3f})",
                line=dict(color=colors_roc[mname], width=2.5)
            ))

        fig.update_layout(
            title="ROC Curve — Random Forest vs XGBoost",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.05),
            margin=dict(t=50, b=10), height=450
        )
        st.plotly_chart(fig)
        st.caption(
            "ROC-AUC measures ranking quality. AUC=1.0 is perfect; "
            "AUC=0.5 is random. Aim for AUC > 0.75 before using the model for decisions."
        )

        # Probability distribution
        st.subheader("Predicted probability distribution")
        prob_df = pd.DataFrame({
            "Probability (Interested)": rf_proba,
            "Actual label": ["Interested" if y == 1 else "Not Interested"
                              for y in y_te.values]
        })
        fig2 = px.histogram(
            prob_df, x="Probability (Interested)", color="Actual label",
            nbins=40, barmode="overlay", opacity=0.7,
            title="RF: predicted probability by actual class",
            color_discrete_map={"Interested": "#1D9E75", "Not Interested": "#E24B4A"},
        )
        fig2.add_vline(x=threshold, line_dash="dash", line_color="#7F77DD",
                       annotation_text=f"Threshold = {threshold:.2f}")
        fig2.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig2)

    # ── Tab 3: Feature importance ─────────────────────────────────────────────
    with tab3:
        st.subheader("Feature importance")
        top_n_feat = st.slider("Top N features to show", 10, 40, 20, 5)

        fi_cols = st.columns(2)

        # Random Forest importance
        with fi_cols[0]:
            rf_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
            rf_top = rf_imp.head(top_n_feat)
            fig = px.bar(
                x=rf_top.values, y=[f.replace("_", " ").title() for f in rf_top.index],
                orientation="h", title=f"Random Forest — Top {top_n_feat} features",
                labels={"x": "Importance", "y": "Feature"},
                color=rf_top.values, color_continuous_scale="Purples"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=50, b=10), height=max(300, top_n_feat * 22))
            st.plotly_chart(fig)

        # XGBoost importance
        with fi_cols[1]:
            xgb_imp = pd.Series(xgb.feature_importances_, index=feat_names).sort_values(ascending=False)
            xgb_top = xgb_imp.head(top_n_feat)
            fig2 = px.bar(
                x=xgb_top.values, y=[f.replace("_", " ").title() for f in xgb_top.index],
                orientation="h", title=f"XGBoost — Top {top_n_feat} features",
                labels={"x": "Importance", "y": "Feature"},
                color=xgb_top.values, color_continuous_scale="Teal"
            )
            fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(t=50, b=10), height=max(300, top_n_feat * 22))
            st.plotly_chart(fig2)

        # Permutation importance (no external deps needed)
        st.subheader("Permutation feature importance (Random Forest)")
        with st.spinner("Computing permutation importance…"):
            try:
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(
                    rf, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1
                )
                perm_imp = pd.Series(
                    perm.importances_mean, index=feat_names
                ).sort_values(ascending=False).head(top_n_feat)

                fig3 = px.bar(
                    x=perm_imp.values,
                    y=[f.replace("_", " ").title() for f in perm_imp.index],
                    orientation="h",
                    title=f"Permutation importance (RF) — Top {top_n_feat} features",
                    labels={"x": "Mean accuracy decrease", "y": "Feature"},
                    color=perm_imp.values, color_continuous_scale="Oranges"
                )
                fig3.update_layout(showlegend=False, coloraxis_showscale=False,
                                   margin=dict(t=50, b=10),
                                   height=max(300, top_n_feat * 22))
                st.plotly_chart(fig3)
                st.caption(
                    "Permutation importance measures how much model accuracy drops "
                    "when each feature is randomly shuffled. Higher = more important."
                )
            except Exception as e:
                st.caption(f"Permutation importance skipped: {e}")

    # ── Tab 4: Classification report ─────────────────────────────────────────
    with tab4:
        st.subheader("Full classification report")
        for mname, model, pred in [
            ("Random Forest", rf, rf_pred),
            ("XGBoost", xgb, xgb_pred),
        ]:
            st.markdown(f"**{mname}** (threshold = {threshold:.2f})")
            report = classification_report(
                y_te, pred,
                target_names=["Not Interested", "Interested"],
                output_dict=True
            )
            report_df = pd.DataFrame(report).T.round(3)
            report_df = report_df.drop(columns=["support"], errors="ignore")
            st.dataframe(report_df)
            st.markdown("---")

        st.markdown("**Founder's guide to threshold selection:**")
        st.markdown(
            "- **Launch phase**: Lower threshold (0.35–0.45) → higher recall → catch more potential visitors, "
            "accept some false positives. Cost of missing a real customer > cost of a wasted outreach.\n"
            "- **Scale phase**: Raise threshold (0.55–0.65) → higher precision → more efficient marketing spend.\n"
            "- **Current threshold**: A respondent needs a predicted probability ≥ "
            f"{threshold:.0%} to be classified as 'Interested'."
        )
