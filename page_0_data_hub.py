import streamlit as st
import pandas as pd
import numpy as np
from utils import engineer_features

def run():
    st.header("Data Hub — Upload & Quality Control")
    st.markdown(
        "Load the base survey dataset and optionally upload new respondent batches. "
        "All models are trained here and cached for use across every page."
    )

    # ── Base dataset ────────────────────────────────────────────────────────
    st.subheader("Base training dataset")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload the 2,000-row survey CSV  (art_pod_survey_india_2000.csv)",
            type=["csv"], key="base_upload"
        )
    with col2:
        st.markdown("**Expected columns include:**")
        st.caption(
            "age_group · gender · city · income_band · creative_identity_score · "
            "ai_comfort · prod_* · barrier_* · visit_intent_binary · annual_spend_midpoint …"
        )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df = engineer_features(df)
            st.session_state["df"] = df
            st.session_state["base_loaded"] = True
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return
    elif "df" not in st.session_state:
        st.info("Please upload the base CSV to begin.")
        _show_sample_schema()
        return

    df = st.session_state["df"]

    # ── Data quality report ──────────────────────────────────────────────────
    st.subheader("Data quality report")
    qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
    total = len(df)
    n_missing_rows = df.isnull().any(axis=1).sum()
    n_straight = df.get("straight_liner_flag", pd.Series(0, index=df.index)).sum()
    n_outlier = df.get("outlier_flag", pd.Series(0, index=df.index)).sum()
    n_incon = df.get("inconsistent_response_flag", pd.Series(0, index=df.index)).sum()

    qcol1.metric("Total respondents", f"{total:,}")
    qcol2.metric("Rows with missing values", f"{n_missing_rows:,}",
                 delta=f"{n_missing_rows/total*100:.1f}%", delta_color="inverse")
    qcol3.metric("Straight-liners (QA)", f"{int(n_straight)}")
    qcol4.metric("Outlier-flagged rows", f"{int(n_outlier)}")
    qcol5.metric("Inconsistent responses", f"{int(n_incon)}")

    # Filter options
    st.subheader("Data cleaning options")
    fcol1, fcol2, fcol3 = st.columns(3)
    rm_straight = fcol1.checkbox("Remove straight-liners", value=True)
    rm_outlier = fcol2.checkbox("Remove spending outliers", value=False)
    rm_incon = fcol3.checkbox("Remove inconsistent responses", value=False)

    df_clean = df.copy()
    if rm_straight and "straight_liner_flag" in df_clean.columns:
        df_clean = df_clean[df_clean["straight_liner_flag"] == 0]
    if rm_outlier and "outlier_flag" in df_clean.columns:
        df_clean = df_clean[df_clean["outlier_flag"] == 0]
    if rm_incon and "inconsistent_response_flag" in df_clean.columns:
        df_clean = df_clean[df_clean["inconsistent_response_flag"] == 0]

    st.session_state["df_clean"] = df_clean
    st.success(f"Clean dataset ready: {len(df_clean):,} rows")

    # Missing value summary
    missing = df_clean.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        with st.expander(f"Missing value detail ({len(missing)} columns with NAs)"):
            miss_df = pd.DataFrame({
                "Column": missing.index,
                "Missing count": missing.values,
                "Missing %": (missing.values / len(df_clean) * 100).round(1)
            })
            st.dataframe(miss_df,  hide_index=True)

    # ── Dataset preview ──────────────────────────────────────────────────────
    st.subheader("Dataset preview")
    preview_cols = [
        "respondent_id", "persona_type", "age_group", "gender", "city",
        "income_band", "creative_identity_score", "ai_comfort",
        "visit_intent_5class", "visit_intent_binary", "annual_spend_midpoint"
    ]
    show_cols = [c for c in preview_cols if c in df_clean.columns]
    st.dataframe(df_clean[show_cols].head(20),  hide_index=True)

    # ── Column completeness ──────────────────────────────────────────────────
    with st.expander("Full column completeness report"):
        completeness = pd.DataFrame({
            "Column": df_clean.columns,
            "Non-null count": df_clean.count().values,
            "Completeness %": (df_clean.count().values / len(df_clean) * 100).round(1),
            "Dtype": df_clean.dtypes.astype(str).values,
        })
        st.dataframe(completeness,  hide_index=True)

    # ── New customer upload ──────────────────────────────────────────────────
    st.divider()
    st.subheader("New customer batch upload (predict & score)")
    st.markdown(
        "Upload any new survey CSV with the same column schema. Each respondent will be "
        "classified, clustered, spend-predicted, and assigned a marketing playbook instantly."
    )

    new_upload = st.file_uploader(
        "Upload new respondents CSV", type=["csv"], key="new_upload"
    )
    if new_upload:
        try:
            df_new = pd.read_csv(new_upload)
            df_new = engineer_features(df_new)
            st.session_state["df_new"] = df_new
            st.success(f"New batch loaded: {len(df_new):,} respondents. "
                       "Go to Page 7 — New Customer Predictor to score them.")
        except Exception as e:
            st.error(f"Could not read new upload: {e}")

    # ── Model train trigger ──────────────────────────────────────────────────
    st.divider()
    if st.button("Train / retrain all models", type="primary"):
        with st.spinner("Training Classification, Clustering, and Regression models…"):
            _train_all_models(df_clean)
        st.success("All models trained and cached. Navigate to any page.")

    if "models_trained" in st.session_state and st.session_state["models_trained"]:
        st.info("Models are trained and ready. Use the sidebar to navigate.")


def _train_all_models(df):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    import xgboost as xgb
    from utils import get_cluster_features, get_classification_features, get_regression_features

    # ── Clustering ──────────────────────────────────────────────────────────
    X_clust, clust_feat_names, clust_imp = get_cluster_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    # Find best K using silhouette
    from sklearn.metrics import silhouette_score
    best_k, best_sil = 5, -1
    sil_scores = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labs)
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil, best_k = sil, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)
    df["cluster_id"] = cluster_labels

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=10)
    db_labels = dbscan.fit_predict(X_scaled)

    st.session_state.update({
        "km_model": km_final, "km_scaler": scaler,
        "km_best_k": best_k, "km_sil_scores": sil_scores,
        "cluster_labels": cluster_labels, "clust_features": clust_feat_names,
        "clust_imp": clust_imp, "X_pca": X_pca, "pca_model": pca,
        "db_labels": db_labels, "df_clustered": df.copy(),
    })

    # Add cluster label back to df_clean
    st.session_state["df_clean"] = df.copy()

    # ── Classification ──────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split, cross_val_score
    X_cls, y_cls, cls_feat_names, cls_imp = get_classification_features(df)
    if y_cls is not None and y_cls.nunique() > 1:
        # Add cluster as feature
        X_cls["cluster_id"] = cluster_labels[:len(X_cls)]
        cls_feat_names = cls_feat_names + ["cluster_id"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
        rf_cls = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_cls.fit(X_tr, y_tr)

        xgb_cls = xgb.XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss",
            verbosity=0
        )
        xgb_cls.fit(X_tr, y_tr)

        cv_rf = cross_val_score(rf_cls, X_cls, y_cls, cv=5, scoring="roc_auc")
        cv_xgb = cross_val_score(xgb_cls, X_cls, y_cls, cv=5, scoring="roc_auc")

        st.session_state.update({
            "rf_cls": rf_cls, "xgb_cls": xgb_cls,
            "X_cls_train": X_tr, "X_cls_test": X_te,
            "y_cls_train": y_tr, "y_cls_test": y_te,
            "cls_feat_names": cls_feat_names, "cls_imp": cls_imp,
            "cv_rf_auc": cv_rf, "cv_xgb_auc": cv_xgb,
        })

    # ── Regression ──────────────────────────────────────────────────────────
    X_reg, y_reg, reg_feat_names, reg_imp = get_regression_features(df)
    if y_reg is not None:
        X_reg["cluster_id"] = cluster_labels[:len(X_reg)]
        reg_feat_names = reg_feat_names + ["cluster_id"]

        X_rtr, X_rte, y_rtr, y_rte = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        lr = LinearRegression()
        lr.fit(X_rtr, y_rtr)
        rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_reg.fit(X_rtr, y_rtr)
        gbm_reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
        gbm_reg.fit(X_rtr, y_rtr)

        st.session_state.update({
            "lr_reg": lr, "rf_reg": rf_reg, "gbm_reg": gbm_reg,
            "X_reg_train": X_rtr, "X_reg_test": X_rte,
            "y_reg_train": y_rtr, "y_reg_test": y_rte,
            "reg_feat_names": reg_feat_names, "reg_imp": reg_imp,
        })

    st.session_state["models_trained"] = True


def _show_sample_schema():
    st.markdown("**Expected column groups in the CSV:**")
    schema = {
        "Demographics": "age_group, gender, city, education, occupation, income_band, income_midpoint",
        "Psychographic": "creative_identity_score, ci_* (5 items), ai_comfort",
        "Behaviour": "act_* (8 cols), exp_* (7 cols), leisure_spend_band, current_creative_spend_band",
        "Products (binary)": "prod_ai_coaching, prod_art_kit, prod_kids_session … (10 cols)",
        "Art styles (binary)": "style_mandala, style_madhubani … (8 cols)",
        "Barriers (binary)": "barrier_price, barrier_identity … (9 cols)",
        "India context": "gifting_*, language_preference, session_time_pref",
        "Targets": "annual_spend_midpoint (regression), visit_intent_binary / visit_intent_5class (classification)",
    }
    for grp, cols in schema.items():
        st.markdown(f"- **{grp}:** `{cols}`")
