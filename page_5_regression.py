import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run():
    st.header("Regression — How Much Will Each Customer Spend?")
    st.markdown(
        "Three models predict each respondent's annual spend on Art Experience Pod products. "
        "The predicted spend directly determines marketing investment per lead."
    )

    if "df_clean" not in st.session_state:
        st.warning("Please upload data on the Data Hub page first.")
        return
    if "lr_reg" not in st.session_state:
        st.warning("Please train models on the Data Hub page first.")
        return

    lr = st.session_state["lr_reg"]
    rf_reg = st.session_state["rf_reg"]
    gbm_reg = st.session_state["gbm_reg"]
    X_tr = st.session_state["X_reg_train"]
    X_te = st.session_state["X_reg_test"]
    y_tr = st.session_state["y_reg_train"]
    y_te = st.session_state["y_reg_test"]
    feat_names = st.session_state["reg_feat_names"]

    # ── Controls ──────────────────────────────────────────────────────────────
    log_transform = st.checkbox("Apply log-transform to spend (corrects right skew)", value=False)

    if log_transform:
        y_tr_fit = np.log1p(y_tr)
        y_te_fit = np.log1p(y_te)
    else:
        y_tr_fit = y_tr
        y_te_fit = y_te

    # Refit with transform option
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    lr_fit = LinearRegression().fit(X_tr, y_tr_fit)
    rf_fit = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr_fit)
    gbm_fit = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr_fit)

    def eval_model(model, X, y_true_fit, y_true_orig, log_t):
        pred_fit = model.predict(X)
        pred_orig = np.expm1(pred_fit) if log_t else pred_fit
        pred_orig = np.clip(pred_orig, 0, None)
        return {
            "R²": r2_score(y_true_fit, pred_fit),
            "MAE (₹)": mean_absolute_error(y_true_orig, pred_orig),
            "RMSE (₹)": np.sqrt(mean_squared_error(y_true_orig, pred_orig)),
        }, pred_orig

    lr_metrics, lr_pred = eval_model(lr_fit, X_te, y_te_fit, y_te, log_transform)
    rf_metrics, rf_pred = eval_model(rf_fit, X_te, y_te_fit, y_te, log_transform)
    gbm_metrics, gbm_pred = eval_model(gbm_fit, X_te, y_te_fit, y_te, log_transform)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("Model comparison (test set)")
    metric_df = pd.DataFrame({
        "Metric": list(lr_metrics.keys()),
        "Linear Regression": [f"{v:.3f}" if "R" in k else f"₹{v:,.0f}"
                               for k, v in lr_metrics.items()],
        "Random Forest": [f"{v:.3f}" if "R" in k else f"₹{v:,.0f}"
                          for k, v in rf_metrics.items()],
        "Gradient Boosting": [f"{v:.3f}" if "R" in k else f"₹{v:,.0f}"
                               for k, v in gbm_metrics.items()],
    })
    st.dataframe(metric_df,  hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Actual vs Predicted", "Residual Analysis", "Feature Importance", "Outlier Detection"
    ])

    # ── Tab 1: Actual vs Predicted ────────────────────────────────────────────
    with tab1:
        model_sel = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
        pred_map = {
            "Random Forest": rf_pred,
            "Gradient Boosting": gbm_pred,
            "Linear Regression": lr_pred,
        }
        pred_use = pred_map[model_sel]

        scatter_df = pd.DataFrame({
            "Actual (₹)": y_te.values,
            "Predicted (₹)": pred_use,
        })
        fig = px.scatter(
            scatter_df, x="Actual (₹)", y="Predicted (₹)",
            opacity=0.5,
            title=f"{model_sel} — Actual vs Predicted annual spend (₹)",
            color_discrete_sequence=["#7F77DD"],
        )
        max_val = max(scatter_df["Actual (₹)"].max(), scatter_df["Predicted (₹)"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="coral"),
                      name="Perfect prediction")
        fig.update_layout(margin=dict(t=50, b=10), height=420)
        st.plotly_chart(fig)
        st.caption("Points on the dashed line = perfect prediction. Spread above/below = model error.")

    # ── Tab 2: Residuals ──────────────────────────────────────────────────────
    with tab2:
        residuals = y_te.values - rf_pred
        resid_df = pd.DataFrame({"Predicted (₹)": rf_pred, "Residual (₹)": residuals})

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(resid_df, x="Predicted (₹)", y="Residual (₹)",
                             opacity=0.5, title="Residual plot (RF)",
                             color_discrete_sequence=["#1D9E75"])
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(fig)

        with c2:
            fig2 = px.histogram(resid_df, x="Residual (₹)", nbins=40,
                                title="Residual distribution (RF)",
                                color_discrete_sequence=["#7F77DD"])
            fig2.add_vline(x=0, line_dash="dash", line_color="coral")
            fig2.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(fig2)

        st.caption(
            "Residuals should be centred around zero with no visible funnel pattern. "
            "A funnel (heteroscedasticity) suggests the log-transform checkbox above may help."
        )

    # ── Tab 3: Feature importance ─────────────────────────────────────────────
    with tab3:
        top_n = st.slider("Top N features", 10, 30, 15, 5)

        rc1, rc2 = st.columns(2)
        with rc1:
            rf_fi = pd.Series(rf_fit.feature_importances_, index=feat_names).sort_values(ascending=False).head(top_n)
            fig = px.bar(
                x=rf_fi.values, y=[f.replace("_", " ").title() for f in rf_fi.index],
                orientation="h", title=f"RF Regressor — Top {top_n} features",
                labels={"x": "Importance", "y": "Feature"},
                color=rf_fi.values, color_continuous_scale="Purples"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=50, b=10), height=max(300, top_n * 22))
            st.plotly_chart(fig)

        with rc2:
            lr_coef = pd.Series(lr_fit.coef_, index=feat_names).sort_values(key=abs, ascending=False).head(top_n)
            colors = ["#1D9E75" if v > 0 else "#E24B4A" for v in lr_coef.values]
            fig2 = go.Figure(go.Bar(
                x=lr_coef.values,
                y=[f.replace("_", " ").title() for f in lr_coef.index],
                orientation="h", marker_color=colors
            ))
            fig2.update_layout(
                title=f"Linear Regression coefficients — Top {top_n}",
                xaxis_title="Coefficient (₹)",
                margin=dict(t=50, b=10), height=max(300, top_n * 22)
            )
            st.plotly_chart(fig2)

        # Permutation importance for regression (no external deps)
        st.subheader("Permutation feature importance (RF Regressor)")
        with st.spinner("Computing permutation importance…"):
            try:
                from sklearn.inspection import permutation_importance
                perm_r = permutation_importance(
                    rf_fit, X_te, y_te_fit, n_repeats=10, random_state=42, n_jobs=-1
                )
                perm_r_imp = pd.Series(
                    perm_r.importances_mean, index=feat_names
                ).sort_values(ascending=False).head(top_n)
                fig3 = px.bar(
                    x=perm_r_imp.values,
                    y=[f.replace("_", " ").title() for f in perm_r_imp.index],
                    orientation="h",
                    title=f"Permutation importance (RF Regressor) — Top {top_n}",
                    labels={"x": "Mean R² decrease", "y": "Feature"},
                    color=perm_r_imp.values, color_continuous_scale="Oranges"
                )
                fig3.update_layout(showlegend=False, coloraxis_showscale=False,
                                   margin=dict(t=50, b=10),
                                   height=max(300, top_n * 22))
                st.plotly_chart(fig3)
                st.caption(
                    "Permutation importance: how much R² drops when each feature is shuffled. "
                    "Higher bars = features the model relies on most for spend prediction."
                )
            except Exception as e:
                st.caption(f"Permutation importance skipped: {e}")

    # ── Tab 4: Outlier detection ──────────────────────────────────────────────
    with tab4:
        st.subheader("Cook's distance — influential observations")
        st.markdown(
            "Cook's distance identifies respondents whose responses have outsized influence "
            "on the regression coefficients. Points above the threshold line should be reviewed."
        )
        n, p = X_te.shape
        lr_pred_te = lr_fit.predict(X_te)
        residuals_lr = y_te.values - lr_pred_te
        mse_lr = np.mean(residuals_lr ** 2)

        # Approximate Cook's D
        hat = X_te.values @ np.linalg.pinv(X_te.values.T @ X_te.values) @ X_te.values.T
        leverage = np.diag(hat)
        cooks_d = (residuals_lr ** 2 * leverage) / (p * mse_lr * (1 - leverage + 1e-9) ** 2)
        cooks_threshold = 4 / n

        cook_df = pd.DataFrame({
            "Observation": range(len(cooks_d)),
            "Cook's D": cooks_d,
            "Influential": cooks_d > cooks_threshold,
        })
        fig = px.scatter(
            cook_df, x="Observation", y="Cook's D",
            color="Influential",
            color_discrete_map={True: "#E24B4A", False: "#7F77DD"},
            title="Cook's distance (red = influential observations)",
            opacity=0.7,
        )
        fig.add_hline(y=cooks_threshold, line_dash="dash", line_color="coral",
                      annotation_text=f"Threshold = 4/n = {cooks_threshold:.4f}")
        fig.update_layout(margin=dict(t=50, b=10), height=400)
        st.plotly_chart(fig)

        n_influential = cook_df["Influential"].sum()
        st.metric("Influential observations", f"{n_influential} ({n_influential/n*100:.1f}%)")
        st.caption(
            "These are the outlier respondents injected during synthetic data generation "
            "(ultra-high spenders and implausible low-income high-spenders). "
            "Consider removing or Winsorising them before production use."
        )

        # Spend segment distribution
        st.subheader("Predicted spend segmentation")
        if "df_clean" in st.session_state:
            from utils import get_regression_features
            df_all = st.session_state["df_clean"].copy()
            X_all, y_all, _, imp_all = get_regression_features(df_all)
            if "cluster_id" in df_all.columns:
                X_all["cluster_id"] = df_all["cluster_id"].values[:len(X_all)]
            pred_all = np.clip(np.expm1(gbm_fit.predict(X_all))
                               if log_transform else gbm_fit.predict(X_all), 0, None)
            seg_counts = pd.cut(
                pred_all,
                bins=[0, 4000, 12000, np.inf],
                labels=["Entry (<₹4K)", "Mid-value (₹4K–12K)", "High-value (>₹12K)"]
            ).value_counts()
            fig4 = px.pie(names=seg_counts.index, values=seg_counts.values,
                          title="Customer spend segmentation (GBM predictions)",
                          color_discrete_sequence=["#9FE1CB", "#7F77DD", "#D85A30"])
            fig4.update_layout(margin=dict(t=40, b=10))
            st.plotly_chart(fig4)
