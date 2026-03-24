import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Ordinal mappings ────────────────────────────────────────────────────────
AI_COMFORT_MAP = {
    "Daily user-love it": 5,
    "Occasional-prefer human": 4,
    "Curious-not tried": 3,
    "Sceptical": 2,
    "Not comfortable at all": 1,
}
VISIT_FREQ_MAP = {
    "More than once/week": 5,
    "Once a week": 4,
    "2-3 times/month": 3,
    "Once a month": 2,
    "Once/twice a year": 1,
}
INCOME_MID = {
    "Below 25K": 15000, "25K-50K": 37500, "50K-1L": 75000,
    "1L-2L": 150000, "Above 2L": 250000,
}
CREATIVE_SPEND_MID = {
    "Nothing": 0, "Below 500": 250, "500-1500": 1000,
    "1500-3000": 2250, "3000-5000": 4000, "Above 5000": 6000,
}
LEISURE_SPEND_MID = {
    "Below 500": 250, "500-1500": 1000, "1500-3000": 2250,
    "3000-6000": 4500, "Above 6000": 7500,
}
KIT_WTP_MID = {
    "Would not buy": 0, "Up to 299": 150, "300-599": 450,
    "600-999": 800, "1000-1499": 1250, "Above 1500": 2000,
}
ANNUAL_SPEND_MID = {
    "Below 1K": 500, "1K-3K": 2000, "3K-6K": 4500,
    "6K-12K": 9000, "12K-24K": 18000, "Above 24K": 35000,
}
SESSION_DUR_MAP = {"30 min express": 1, "60 min standard": 2, "90 min deep-dive": 3}
VISIT_INTENT_MAP = {
    "Definitely will visit": 5, "Likely to visit": 4,
    "Undecided": 3, "Unlikely to visit": 2, "Definitely will not visit": 1,
}

# ── Cluster persona names (assigned after fitting) ──────────────────────────
CLUSTER_PERSONAS = {
    0: "Weekend Creative Parent",
    1: "Urban Millennial Creator",
    2: "Gen Z Social Explorer",
    3: "Corporate Wellness Seeker",
    4: "Tier-2 Aspirational Learner",
    5: "Mindful Art Therapist",
    6: "Disengaged Observer",
}

# ── Prescriptive rules per cluster ─────────────────────────────────────────
PRESCRIPTIVE_RULES = {
    "Weekend Creative Parent": {
        "bundle": "Kids Art Session + Family Art Kit + Print-on-Demand Keepsake",
        "discount": "Festival/Diwali family discount (20% off)",
        "channel": "WhatsApp groups & school notice boards",
        "message": "Give your child the gift of creativity this weekend",
        "priority_boost": 0.15,
    },
    "Urban Millennial Creator": {
        "bundle": "AI Coaching Session + Digital Tablet + Portfolio App",
        "discount": "First session free (trial hook)",
        "channel": "Instagram Reels & YouTube Shorts",
        "message": "Create something worth sharing",
        "priority_boost": 0.10,
    },
    "Gen Z Social Explorer": {
        "bundle": "Group Social Session + Print Keepsake + Skill Tracker",
        "discount": "Bring-a-friend 2-for-1 offer",
        "channel": "Instagram Stories & college notice boards",
        "message": "Bring your crew — leave with something you made",
        "priority_boost": 0.05,
    },
    "Corporate Wellness Seeker": {
        "bundle": "Corporate Team Session + Branded Keepsake + Group Booking",
        "discount": "Corporate tie-up / employer wellness budget",
        "channel": "LinkedIn & HR email campaigns",
        "message": "Recharge your team's creativity — book a pod session",
        "priority_boost": 0.08,
    },
    "Tier-2 Aspirational Learner": {
        "bundle": "AI Coaching + Skill Tracker + Regional Language Session",
        "discount": "Student discount + first session ₹99 trial",
        "channel": "WhatsApp forwards & YouTube regional creators",
        "message": "Learn art in your language — track your progress every week",
        "priority_boost": 0.03,
    },
    "Mindful Art Therapist": {
        "bundle": "Wellness Art Therapy + Mood Prompt Generator + Mandala Kit",
        "discount": "Monthly wellness membership (bundled sessions)",
        "channel": "Instagram wellness hashtags & yoga studio partnerships",
        "message": "Slow down. Create. Feel better.",
        "priority_boost": 0.06,
    },
    "Disengaged Observer": {
        "bundle": "Express 30-min Taster Session + Take-home Starter Kit",
        "discount": "Aggressive trial: first 30-min session free",
        "channel": "Mall in-person discovery & QR code at entrance",
        "message": "No skill needed — just walk in and try",
        "priority_boost": 0.00,
    },
}


# ── Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Midpoint numeric columns from band labels
    for col, mapping in [
        ("income_band", INCOME_MID),
        ("current_creative_spend_band", CREATIVE_SPEND_MID),
        ("leisure_spend_band", LEISURE_SPEND_MID),
        ("kit_wtp_band", KIT_WTP_MID),
        ("annual_spend_band", ANNUAL_SPEND_MID),
    ]:
        mid_col = col.replace("_band", "_midpoint").replace("_band", "_mid")
        if col in df.columns:
            if mid_col not in df.columns:
                df[mid_col] = df[col].map(mapping)

    # Ensure income_midpoint exists
    if "income_midpoint" not in df.columns and "income_band" in df.columns:
        df["income_midpoint"] = df["income_band"].map(INCOME_MID)

    # Ordinal encodings
    if "ai_comfort" in df.columns:
        df["ai_comfort_enc"] = df["ai_comfort"].map(AI_COMFORT_MAP).fillna(3)
    if "visit_freq_intent" in df.columns:
        df["visit_freq_intent_enc"] = df["visit_freq_intent"].map(VISIT_FREQ_MAP).fillna(2)
    if "session_duration_pref" in df.columns:
        df["session_duration_enc"] = df["session_duration_pref"].map(SESSION_DUR_MAP).fillna(2)
    if "visit_intent_5class" in df.columns:
        df["visit_intent_enc"] = df["visit_intent_5class"].map(VISIT_INTENT_MAP).fillna(3)

    # Binary classification target
    if "visit_intent_binary" not in df.columns and "visit_intent_5class" in df.columns:
        df["visit_intent_binary"] = df["visit_intent_5class"].isin(
            ["Definitely will visit", "Likely to visit"]
        ).astype(int)

    # Conjoint binary
    if "conjoint_choice" in df.columns:
        df["conjoint_chose_pod"] = (df["conjoint_choice"] == "Option A-Art Pod Session").astype(int)

    # City tier numeric
    if "city" in df.columns:
        tier_map = {
            "Metro-Mumbai": 3, "Metro-Delhi NCR": 3, "Metro-Bengaluru": 3,
            "Metro-Hyd/Pune/Chennai": 3, "Tier 2 City": 2, "Tier 3 City/Town": 1,
        }
        df["city_tier"] = df["city"].map(tier_map).fillna(2)

    # Age numeric
    if "age_group" in df.columns:
        age_map = {
            "Under 18": 15, "18-24": 21, "25-34": 29,
            "35-44": 39, "45-55": 50, "55+": 60,
        }
        df["age_mid"] = df["age_group"].map(age_map).fillna(29)

    return df


def get_cluster_features(df: pd.DataFrame) -> tuple:
    """Return feature matrix for clustering + feature names."""
    df = engineer_features(df)
    numeric_candidates = [
        "creative_identity_score", "income_midpoint", "ai_comfort_enc",
        "leisure_spend_mid", "current_creative_spend_mid", "visit_freq_intent_enc",
        "city_tier", "age_mid", "kit_wtp_mid", "session_duration_enc",
        "barrier_price", "barrier_identity", "barrier_time", "barrier_location",
        "prod_kids_session", "prod_wellness_therapy", "prod_ai_coaching",
        "mot_stress_relief", "mot_child_dev", "mot_skill_learning",
        "act_gaming", "act_yoga", "act_painting",
        "conjoint_chose_pod",
    ]
    available = [c for c in numeric_candidates if c in df.columns]
    X = df[available].copy()
    # Impute missing
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    return X_imp, available, imp


def get_classification_features(df: pd.DataFrame) -> tuple:
    """Return X, y for classification."""
    df = engineer_features(df)
    numeric_cols = [
        "creative_identity_score", "income_midpoint", "ai_comfort_enc",
        "leisure_spend_mid", "current_creative_spend_mid", "visit_freq_intent_enc",
        "city_tier", "age_mid", "kit_wtp_mid", "conjoint_chose_pod",
        "barrier_price", "barrier_identity", "barrier_skill", "barrier_embarrass",
        "barrier_location", "barrier_time", "barrier_family", "barrier_none",
        "prod_ai_coaching", "prod_art_kit", "prod_tablet_session",
        "prod_print_on_demand", "prod_skill_tracker", "prod_mood_prompts",
        "prod_kids_session", "prod_corporate", "prod_membership", "prod_wellness_therapy",
        "mot_stress_relief", "mot_skill_learning", "mot_social", "mot_social_media",
        "mot_child_dev", "mot_novelty", "mot_career",
        "act_painting", "act_digital_art", "act_gaming", "act_yoga",
        "act_photography", "act_music", "act_reading", "act_diy",
        "exp_art_workshop", "exp_gaming_zone", "exp_escape_room",
        "style_mandala", "style_madhubani", "style_abstract", "style_anime",
        "gifting_diwali", "gifting_birthday", "gifting_child",
        "nps_score",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    X = df[available].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    X_out = pd.DataFrame(X_imp, columns=available, index=df.index)

    y = None
    if "visit_intent_binary" in df.columns:
        y = df["visit_intent_binary"].fillna(0).astype(int)
    return X_out, y, available, imp


def get_regression_features(df: pd.DataFrame) -> tuple:
    """Return X, y for regression."""
    df = engineer_features(df)
    numeric_cols = [
        "creative_identity_score", "income_midpoint", "ai_comfort_enc",
        "leisure_spend_mid", "current_creative_spend_mid", "visit_freq_intent_enc",
        "city_tier", "age_mid", "kit_wtp_mid", "conjoint_chose_pod",
        "nps_score", "barrier_price", "barrier_none",
        "prod_ai_coaching", "prod_art_kit", "prod_kids_session",
        "prod_membership", "prod_wellness_therapy",
        "mot_stress_relief", "mot_child_dev", "mot_skill_learning",
        "gifting_diwali", "gifting_birthday",
        "session_duration_enc",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    X = df[available].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    X_out = pd.DataFrame(X_imp, columns=available, index=df.index)

    y = None
    if "annual_spend_midpoint" in df.columns:
        y = pd.to_numeric(df["annual_spend_midpoint"], errors="coerce").fillna(
            df["annual_spend_midpoint"].median() if "annual_spend_midpoint" in df.columns else 9000
        )
    return X_out, y, available, imp


def get_arm_basket(df: pd.DataFrame, basket_type: str) -> list:
    """Return list of transactions for ARM."""
    if basket_type == "Product basket":
        prod_cols = [c for c in df.columns if c.startswith("prod_")]
        transactions = []
        for _, row in df[prod_cols].iterrows():
            items = [c.replace("prod_", "").replace("_", " ").title()
                     for c in prod_cols if row[c] == 1]
            if len(items) >= 2:
                transactions.append(items)
        return transactions

    elif basket_type == "Activity + Product basket":
        act_cols = [c for c in df.columns if c.startswith("act_")]
        prod_cols = [c for c in df.columns if c.startswith("prod_")]
        transactions = []
        for _, row in df[act_cols + prod_cols].iterrows():
            items = []
            for c in act_cols:
                if row[c] == 1:
                    items.append("ACT:" + c.replace("act_", "").replace("_", " ").title())
            for c in prod_cols:
                if row[c] == 1:
                    items.append("PROD:" + c.replace("prod_", "").replace("_", " ").title())
            if len(items) >= 2:
                transactions.append(items)
        return transactions

    elif basket_type == "Art style + Product basket":
        style_cols = [c for c in df.columns if c.startswith("style_")]
        prod_cols = [c for c in df.columns if c.startswith("prod_")]
        transactions = []
        for _, row in df[style_cols + prod_cols].iterrows():
            items = []
            for c in style_cols:
                if row[c] == 1:
                    items.append("STYLE:" + c.replace("style_", "").replace("_", " ").title())
            for c in prod_cols:
                if row[c] == 1:
                    items.append("PROD:" + c.replace("prod_", "").replace("_", " ").title())
            if len(items) >= 2:
                transactions.append(items)
        return transactions

    elif basket_type == "Barrier co-occurrence basket":
        barrier_cols = [c for c in df.columns if c.startswith("barrier_")]
        transactions = []
        for _, row in df[barrier_cols].iterrows():
            items = [c.replace("barrier_", "Barrier: ").replace("_", " ").title()
                     for c in barrier_cols if row[c] == 1]
            if len(items) >= 2:
                transactions.append(items)
        return transactions

    return []


def assign_priority(prob: float, spend: float) -> str:
    if prob >= 0.70 and spend >= 9000:
        return "HOT"
    elif prob >= 0.40 or spend >= 4000:
        return "WARM"
    else:
        return "COLD"


def get_cluster_name(cluster_id: int) -> str:
    return CLUSTER_PERSONAS.get(int(cluster_id), f"Cluster {cluster_id}")


def compute_ltv_estimate(cluster_name: str, spend: float) -> float:
    """Simple 3-year LTV estimate with retention factor per cluster."""
    retention = {
        "Weekend Creative Parent": 0.78,
        "Urban Millennial Creator": 0.72,
        "Gen Z Social Explorer": 0.58,
        "Corporate Wellness Seeker": 0.65,
        "Tier-2 Aspirational Learner": 0.50,
        "Mindful Art Therapist": 0.70,
        "Disengaged Observer": 0.30,
    }
    r = retention.get(cluster_name, 0.60)
    return round(spend * (1 + r + r**2), 0)
