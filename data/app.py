import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title="Disease Risk Prediction — Explainable AI",
    page_icon="🩺",
    layout="wide",
)

NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@st.cache_data
def load_dataset():
    """Fetch UCI Cleveland Heart Disease dataset (ID 45), with a local fallback."""
    try:
        hd = fetch_ucirepo(id=45)
        X = hd.data.features.copy()
        y = (hd.data.targets["num"] > 0).astype(int)
    except Exception:
        rng = np.random.default_rng(42)
        n = 303
        X = pd.DataFrame({
            "age": rng.integers(29, 77, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 564, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 202, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": np.round(rng.uniform(0.0, 6.2, n), 1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.choice([0.0, 1.0, 2.0, 3.0, np.nan], n, p=[0.57, 0.21, 0.12, 0.08, 0.02]),
            "thal": rng.choice([3.0, 6.0, 7.0, np.nan], n, p=[0.54, 0.06, 0.38, 0.02]),
        })
        logits = (
            0.03 * (X["age"] - 54)
            + 0.8 * (X["sex"] == 1).astype(float)
            + 0.55 * (X["cp"] == 4).astype(float)
            + 0.02 * (X["oldpeak"] - 1.0)
            + 0.5 * X["ca"].fillna(0)
            - 0.015 * (X["thalach"] - 150)
            - 1.2
        )
        probs = 1 / (1 + np.exp(-logits))
        y = pd.Series(rng.binomial(1, probs), name="target")

    X["ca"] = X["ca"].fillna(X["ca"].mode()[0])
    X["thal"] = X["thal"].fillna(X["thal"].mode()[0])
    return X, y


@st.cache_resource
def train_models(X, y):
    """Train preprocessing + ML models and return fitted artifacts and metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    }

    pipelines = {}
    rows = []

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 3),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 3),
            "F1-Score": round(f1_score(y_test, preds, zero_division=0), 3),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 3),
        })

    metrics_df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return pipelines, metrics_df, X_train, X_test, y_train, y_test


def compute_shap_values(pipe, X_train_df, X_sample_df):
    """Compute SHAP values for a fitted pipeline and sample DataFrame."""
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    X_train_t = prep.transform(X_train_df)
    X_sample_t = prep.transform(X_sample_df)
    feature_names = prep.get_feature_names_out()

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_train_t)
        sv = explainer.shap_values(X_sample_t)
    elif isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
    else:
        bg = shap.sample(X_train_t, min(50, len(X_train_t)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(X_sample_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]

    return np.array(sv), feature_names


# --- Sidebar Controls ---
st.sidebar.title("🩺 Patient & Model Controls")

preset = st.sidebar.selectbox(
    "Load Clinical Preset",
    ["Custom", "Low-Risk Profile", "Borderline Profile", "High-Risk Profile"],
)

defaults = {
    "age": 54,
    "sex": 1,
    "cp": 3,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 152,
    "exang": 0,
    "oldpeak": 0.8,
    "slope": 1,
    "ca": 0.0,
    "thal": 3.0,
}

if preset == "Low-Risk Profile":
    defaults.update({"age": 42, "sex": 0, "cp": 1, "trestbps": 118, "chol": 195, "thalach": 174, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0.0, "thal": 3.0})
elif preset == "Borderline Profile":
    defaults.update({"age": 56, "sex": 1, "cp": 3, "trestbps": 138, "chol": 258, "thalach": 146, "exang": 0, "oldpeak": 1.4, "slope": 2, "ca": 1.0, "thal": 3.0})
elif preset == "High-Risk Profile":
    defaults.update({"age": 64, "sex": 1, "cp": 4, "trestbps": 162, "chol": 304, "thalach": 118, "exang": 1, "oldpeak": 2.8, "slope": 2, "ca": 2.0, "thal": 7.0})

age = st.sidebar.slider("Age (years)", 29, 80, int(defaults["age"]))
sex = st.sidebar.selectbox("Sex", [1, 0], index=0 if defaults["sex"] == 1 else 1, format_func=lambda v: "Male (1)" if v == 1 else "Female (0)")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4], index=[1, 2, 3, 4].index(int(defaults["cp"])), format_func=lambda v: {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Pain", 4: "4: Asymptomatic"}[v])
trestbps = st.sidebar.slider("Resting BP (mm Hg)", 90, 210, int(defaults["trestbps"]))
chol = st.sidebar.slider("Serum Cholesterol (mg/dL)", 120, 570, int(defaults["chol"]))
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], index=int(defaults["fbs"]), format_func=lambda v: "No (0)" if v == 0 else "Yes (1)")
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], index=int(defaults["restecg"]), format_func=lambda v: {0: "0: Normal", 1: "1: ST-T Abnormality", 2: "2: LV Hypertrophy"}[v])
thalach = st.sidebar.slider("Max Heart Rate (thalach)", 70, 210, int(defaults["thalach"]))
exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1], index=int(defaults["exang"]), format_func=lambda v: "No (0)" if v == 0 else "Yes (1)")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.5, float(defaults["oldpeak"]), step=0.1)
slope = st.sidebar.selectbox("Peak Exercise ST Slope", [1, 2, 3], index=[1, 2, 3].index(int(defaults["slope"])), format_func=lambda v: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[v])
ca = st.sidebar.selectbox("Major Vessels Colored (ca)", [0.0, 1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0, 3.0].index(float(defaults["ca"])))
thal = st.sidebar.selectbox("Thalassemia (thal)", [3.0, 6.0, 7.0], index=[3.0, 6.0, 7.0].index(float(defaults["thal"])), format_func=lambda v: {3.0: "3: Normal", 6.0: "6: Fixed Defect", 7.0: "7: Reversible Defect"}[v])

selected_model = st.sidebar.selectbox("Active Classifier", ["Logistic Regression", "Random Forest", "SVM (RBF)"])

# --- Main App ---
st.title("Disease Risk Prediction Using Explainable AI")
st.caption("Clinical decision-support prototype built on the UCI Cleveland Heart Disease dataset with SHAP explanations. For educational/research use only.")

X, y = load_dataset()
pipelines, metrics_df, X_train, X_test, y_train, y_test = train_models(X, y)

patient_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": float(ca),
    "thal": float(thal),
}])

active_pipe = pipelines[selected_model]
risk_prob = float(active_pipe.predict_proba(patient_df)[0, 1])

if risk_prob < 0.35:
    tier, badge_color = "Low Risk", "🟢"
elif risk_prob < 0.65:
    tier, badge_color = "Moderate / Borderline Risk", "🟡"
else:
    tier, badge_color = "High Risk", "🔴"

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Disease Probability", f"{risk_prob * 100:.1f}%")
col2.metric("Risk Stratification", f"{badge_color} {tier}")
col3.metric("Selected Model", selected_model)

tab1, tab2, tab3 = st.tabs(["🔍 Patient SHAP Explanation", "📊 Global Feature Importance", "📈 Model Benchmarks"])

with tab1:
    st.subheader("Local SHAP Attribution for Current Patient")
    sv_patient, feat_names = compute_shap_values(active_pipe, X_train, patient_df)
    contrib_df = pd.DataFrame({
        "Feature": feat_names,
        "SHAP Value": sv_patient[0],
    })
    contrib_df["Abs"] = contrib_df["SHAP Value"].abs()
    top_contrib = contrib_df.sort_values("Abs", ascending=False).head(10).sort_values("SHAP Value")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#dc2626" if v > 0 else "#0284c7" for v in top_contrib["SHAP Value"]]
    ax.barh(top_contrib["Feature"], top_contrib["SHAP Value"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("SHAP Contribution to Log-Odds / Risk Score (+ Risk / - Protective)")
    ax.set_title(f"Top 10 Local Drivers ({selected_model})")
    plt.tight_layout()
    st.pyplot(fig)

    top_risk = contrib_df.sort_values("SHAP Value", ascending=False).iloc[0]
    top_prot = contrib_df.sort_values("SHAP Value", ascending=True).iloc[0]
    st.info(
        f"**Summary:** Strongest risk-increasing factor is **{top_risk['Feature']}** "
        f"(SHAP `{top_risk['SHAP Value']:+.3f}`), while the strongest protective factor is "
        f"**{top_prot['Feature']}** (SHAP `{top_prot['SHAP Value']:+.3f}`)."
    )

with tab2:
    st.subheader("Global Feature Importance Across Test Cohort")
    sv_test, feat_names = compute_shap_values(active_pipe, X_train, X_test)
    global_df = pd.DataFrame({
        "Feature": feat_names,
        "Mean |SHAP|": np.abs(sv_test).mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    top_global = global_df.head(10).iloc[::-1]
    ax2.barh(top_global["Feature"], top_global["Mean |SHAP|"], color="#0d9488")
    ax2.set_xlabel("Mean Absolute SHAP Value")
    ax2.set_title(f"Top 10 Global Clinical Features ({selected_model})")
    plt.tight_layout()
    st.pyplot(fig2)

    st.dataframe(global_df.round(4), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Hold-Out Test Set Performance (80/20 Stratified Split)")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.bar_chart(metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]])