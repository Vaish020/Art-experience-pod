# Art Experience Pod — Analytics Dashboard

Data-driven market intelligence for India's Art Experience Pod business.
Built with Streamlit · scikit-learn · XGBoost · mlxtend · Plotly · SHAP.

## Dashboard pages

| Page | Layer | Purpose |
|------|-------|---------|
| 0 — Data Hub | Infrastructure | Upload CSV, clean data, train all models |
| 1 — Descriptive Analytics | Descriptive | Who responded, demographics, product interest |
| 2 — Diagnostic + ARM | Diagnostic | Correlations, association rules, cross-tabs |
| 3 — Clustering | Predictive | K-Means personas, silhouette, DBSCAN |
| 4 — Classification | Predictive | RF + XGBoost: will they visit? ROC, SHAP |
| 5 — Regression | Predictive | Spend prediction, Cook's D, feature importance |
| 6 — Prescriptive Playbook | Prescriptive | Segment strategy, bundles, LTV, export |
| 7 — New Customer Predictor | Prescriptive | Upload new leads → instant scoring |

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push all files from this folder to a **public GitHub repository** (no sub-folders — all files at root level).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud reads `requirements.txt` automatically.

## File list (all at root — no sub-folders)

```
app.py                        ← Streamlit entry point
utils.py                      ← Shared helpers, encoders, prescriptive rules
page_0_data_hub.py
page_1_descriptive.py
page_2_diagnostic.py
page_3_clustering.py
page_4_classification.py
page_5_regression.py
page_6_prescriptive.py
page_7_new_predictor.py
requirements.txt              ← All dependencies pinned
.streamlit/config.toml        ← Theme settings
art_pod_survey_india_2000.csv ← Base dataset (upload via dashboard)
README.md
```

## How to use

1. Open the dashboard → go to **Page 0 — Data Hub**.
2. Upload `art_pod_survey_india_2000.csv`.
3. Click **Train / retrain all models** (takes 30–60 seconds).
4. Navigate freely across all pages — models and data are cached in session.
5. For new customer scoring: go to **Page 7**, upload any new survey CSV.

## Algorithms used

- **K-Means + DBSCAN** — customer clustering with PCA visualisation
- **Random Forest + XGBoost** — classification with accuracy, precision, recall, F1, ROC-AUC, SHAP
- **Linear Regression + RF Regressor + GBM** — spend prediction with Cook's distance
- **Apriori (mlxtend)** — association rule mining across 4 baskets (support, confidence, lift)
