import streamlit as st

st.set_page_config(
    page_title="Art Experience Pod — Analytics Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Art Experience Pod")
    st.markdown("### Data-Driven Market Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "0 — Data Hub",
            "1 — Descriptive Analytics",
            "2 — Diagnostic + ARM",
            "3 — Clustering",
            "4 — Classification",
            "5 — Regression",
            "6 — Prescriptive Playbook",
            "7 — New Customer Predictor",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Status indicators
    st.markdown("**Model status**")
    if "models_trained" in st.session_state and st.session_state["models_trained"]:
        st.success("Models trained")
    else:
        st.warning("Models not trained")

    if "df_clean" in st.session_state:
        df = st.session_state["df_clean"]
        st.info(f"Dataset: {len(df):,} rows")
    else:
        st.caption("No data loaded")

    if "df_new" in st.session_state:
        st.info(f"New batch: {len(st.session_state['df_new']):,} rows ready")

    st.markdown("---")
    st.caption(
        "Art Experience Pod · India Market Intelligence\n\n"
        "Analysis layers:\n"
        "Descriptive · Diagnostic · Predictive · Prescriptive\n\n"
        "Algorithms:\n"
        "K-Means · DBSCAN · Random Forest · XGBoost · Linear/RF/GBM Regression · Apriori ARM"
    )

# ── Page routing ─────────────────────────────────────────────────────────────
import page_0_data_hub
import page_1_descriptive
import page_2_diagnostic
import page_3_clustering
import page_4_classification
import page_5_regression
import page_6_prescriptive
import page_7_new_predictor

if "0" in page:
    page_0_data_hub.run()
elif "1" in page:
    page_1_descriptive.run()
elif "2" in page:
    page_2_diagnostic.run()
elif "3" in page:
    page_3_clustering.run()
elif "4" in page:
    page_4_classification.run()
elif "5" in page:
    page_5_regression.run()
elif "6" in page:
    page_6_prescriptive.run()
elif "7" in page:
    page_7_new_predictor.run()
