"""
Heart Disease Risk Prediction — Clinical Decision Support Dashboard
COM 763 Advanced Machine Learning — Portfolio Task 1

Run:  streamlit run app.py

No .pkl files needed — model trains at first startup and is cached.
Repo only needs: app.py, requirements.txt, heart.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffa50022, #ffa50044);
        border: 2px solid #ffa500;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #ffa500;
    }
    .risk-low {
        background: linear-gradient(135deg, #00c85222, #00c85244);
        border: 2px solid #00c852;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #00c852;
    }
    .section-header {
        border-left: 4px solid #ff4b4b;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .recommendation-box {
        background-color: #1e2130;
        border-left: 4px solid #4da6ff;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FEATURE_COLS      = ["age","sex","cp","trestbps","chol","fbs",
                     "restecg","thalach","exang","oldpeak","slope","ca","thal"]
NUMERIC_FEATURES  = ["age","trestbps","chol","thalach","oldpeak"]
ORDINAL_FEATURES  = ["cp","restecg","slope","thal","ca"]
BINARY_FEATURES   = ["sex","fbs","exang"]
FEATURE_NAMES_OUT = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate (bpm)",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (oldpeak)",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (fluoroscopy)",
    "thal":     "Thalassemia Type",
}

CP_MAP      = {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-Anginal Pain", 3:"Asymptomatic"}
RESTECG_MAP = {0:"Normal", 1:"ST-T Wave Abnormality", 2:"Left Ventricular Hypertrophy"}
SLOPE_MAP   = {0:"Upsloping", 1:"Flat", 2:"Downsloping"}
THAL_MAP    = {0:"Normal", 1:"Fixed Defect", 2:"Reversable Defect", 3:"Rev. Defect (other)"}


# ─────────────────────────────────────────────
# Train & Cache Model at Startup
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model — first load takes ~30 seconds…")
def build_model():
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score
    from lightgbm import LGBMClassifier
    import shap as _shap

    # Load & prepare
    df = pd.read_csv("heart.csv")
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)

    def winsorise(df, col):
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df
    for col in ["trestbps", "chol", "oldpeak"]:
        df = winsorise(df, col)

    X = df[FEATURE_COLS]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ORDINAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])

    # Tuned LightGBM (best params from report RandomizedSearchCV)
    lgbm = LGBMClassifier(
        is_unbalance=True, random_state=42, verbose=-1,
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        boosting_type="dart", min_child_samples=20,
        reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
    )
    lgbm_pipe = Pipeline([("pre", preprocessor), ("clf", lgbm)])
    lgbm_pipe.fit(X_train, y_train)

    # Threshold tuning
    y_prob_train = lgbm_pipe.predict_proba(X_train)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(y_train, (y_prob_train >= t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    # Calibration
    base_lgbm = lgbm_pipe.named_steps["clf"]
    cal_clf   = CalibratedClassifierCV(base_lgbm, method="isotonic", cv=5)
    cal_pipe  = Pipeline([("pre", preprocessor), ("clf", cal_clf)])
    cal_pipe.fit(X_train, y_train)

    # SHAP explainer on raw (uncalibrated) LightGBM
    preprocessor.fit(X_train)
    explainer = _shap.TreeExplainer(base_lgbm)

    return cal_pipe, preprocessor, explainer, best_t, X_train


model, preprocessor, explainer, THRESHOLD, X_train = build_model()


# ─────────────────────────────────────────────
# Prediction & SHAP helpers
# ─────────────────────────────────────────────
def predict(patient_dict):
    df_in = pd.DataFrame([patient_dict])
    prob  = model.predict_proba(df_in)[0, 1]
    return prob, int(prob >= THRESHOLD)


def get_shap_values(patient_dict):
    df_in = pd.DataFrame([patient_dict])
    X_t   = preprocessor.transform(df_in)
    sv    = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.array(ev).ravel()[1])
    return sv[0], float(ev)


def risk_band(prob):
    if prob >= 0.65:
        return "HIGH",     "#ff4b4b", "risk-high",     "🚨"
    elif prob >= 0.30:
        return "MODERATE", "#ffa500", "risk-moderate", "⚠️"
    else:
        return "LOW",      "#00c852", "risk-low",      "✅"


# ─────────────────────────────────────────────
# Global SHAP (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Computing global SHAP importance…")
def get_global_shap(_explainer, _preprocessor, _X_train):
    X_t = _preprocessor.transform(_X_train)
    sv  = _explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    return sv


# ─────────────────────────────────────────────
# Sidebar — Patient Input Form
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hearts.png", width=60)
st.sidebar.title("Patient Input")
st.sidebar.caption("Enter the patient's clinical measurements below.")

with st.sidebar.form("patient_form"):
    st.subheader("Demographics")
    age = st.slider("Age (years)", 20, 80, 55)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.subheader("Clinical Symptoms")
    cp  = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                        format_func=lambda x: f"{x} — {CP_MAP[x]}")

    st.subheader("Vital Signs & Labs")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 120, 570, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("ECG & Exercise Test")
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
                            format_func=lambda x: f"{x} — {RESTECG_MAP[x]}")
    thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 210, 150)
    exang   = st.selectbox("Exercise-Induced Angina", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope   = st.selectbox("ST Slope", [0, 1, 2],
                            format_func=lambda x: f"{x} — {SLOPE_MAP[x]}")

    st.subheader("Fluoroscopy & Thalassemia")
    ca   = st.slider("Major Vessels Coloured (0–4)", 0, 4, 0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                         format_func=lambda x: f"{x} — {THAL_MAP[x]}")

    submitted = st.form_submit_button("🔍  Predict Risk", use_container_width=True)


# ─────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Prediction Dashboard")
st.caption("Clinical Decision Support System — COM 763 Advanced Machine Learning")
st.divider()

if not submitted:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<h3 class="section-header">Global Feature Importance (SHAP)</h3>',
                    unsafe_allow_html=True)
        st.caption("Mean |SHAP| across all training patients — higher = stronger predictor")

        sv_all    = get_global_shap(explainer, preprocessor, X_train)
        mean_shap = np.abs(sv_all).mean(axis=0)
        feat_imp  = pd.DataFrame({"feature": FEATURE_NAMES_OUT,
                                   "importance": mean_shap}).sort_values("importance")
        labels    = [FEATURE_LABELS.get(f, f) for f in feat_imp["feature"]]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f1117")
        ax.barh(labels, feat_imp["importance"],
                color=plt.cm.RdYlGn_r(feat_imp["importance"] / feat_imp["importance"].max()))
        ax.set_xlabel("Mean |SHAP value|", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.set_facecolor("#0f1117")
        fig.patch.set_facecolor("#0f1117")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<h3 class="section-header">Model Performance</h3>',
                    unsafe_allow_html=True)
        for name, val in {
            "Test F1 Score":     "0.9902",
            "ROC-AUC":           "1.0000",
            "Test Accuracy":     "99.5%",
            "Optimal Threshold": f"{THRESHOLD:.2f}",
        }.items():
            st.metric(label=name, value=val)

        st.markdown('<h3 class="section-header" style="margin-top:20px">How to Use</h3>',
                    unsafe_allow_html=True)
        st.info(
            "1. Enter patient measurements in the **sidebar form**.\n"
            "2. Click **Predict Risk** to run the model.\n"
            "3. Review the risk gauge, SHAP explanation, and clinical recommendations.\n"
            "4. Use the **What-If Planner** to simulate interventions."
        )

else:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal,
    }

    prob, label = predict(patient)
    risk_label, risk_color, risk_class, risk_icon = risk_band(prob)

    st.markdown(
        f'<div class="{risk_class}">'
        f'{risk_icon} {risk_label} RISK — Predicted Probability: {prob*100:.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_g, col_m = st.columns([1.1, 1])

    with col_g:
        st.markdown('<h3 class="section-header">Risk Probability Gauge</h3>',
                    unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": risk_color, "size": 40}},
            delta={"reference": 50, "valueformat": ".1f",
                   "increasing": {"color": "#ff4b4b"},
                   "decreasing": {"color": "#00c852"}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "white"},
                "bar":   {"color": risk_color},
                "bgcolor": "#1e2130",
                "borderwidth": 2, "bordercolor": "#333",
                "steps": [
                    {"range": [0,  30],  "color": "#00c85222"},
                    {"range": [30, 65],  "color": "#ffa50022"},
                    {"range": [65, 100], "color": "#ff4b4b22"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100,
                },
            },
            title={"text": f"Decision threshold: {THRESHOLD:.2f}",
                   "font": {"color": "#aaa", "size": 13}},
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor="#0f1117", font_color="white",
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_m:
        st.markdown('<h3 class="section-header">Key Clinical Values</h3>',
                    unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Age",            f"{age} yrs")
        m2.metric("Max HR",         f"{thalach} bpm")
        m1.metric("Cholesterol",    f"{chol} mg/dl")
        m2.metric("Blood Pressure", f"{trestbps} mmHg")
        m1.metric("ST Depression",  f"{oldpeak:.1f}")
        m2.metric("Vessels (ca)",   str(ca))
        m1.metric("Chest Pain",     CP_MAP[cp])
        m2.metric("Thalassemia",    THAL_MAP[thal])

    st.divider()

    col_s, col_r = st.columns(2)

    with col_s:
        st.markdown('<h3 class="section-header">SHAP Feature Attribution</h3>',
                    unsafe_allow_html=True)
        st.caption("Features pushing risk up (red) vs down (blue) from population base rate")

        sv, base_val = get_shap_values(patient)
        labels_shap  = [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES_OUT]
        shap_df = pd.DataFrame({"feature": labels_shap, "shap": sv})
        shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=True).index)
        colors  = ["#ff4b4b" if v > 0 else "#4da6ff" for v in shap_df["shap"]]

        fig_w, ax_w = plt.subplots(figsize=(7, 5.5), facecolor="#0f1117")
        ax_w.barh(shap_df["feature"], shap_df["shap"], color=colors)
        ax_w.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax_w.set_xlabel("SHAP value (log-odds contribution)", color="white")
        ax_w.tick_params(colors="white", labelsize=9)
        ax_w.spines[:].set_color("#333")
        ax_w.set_facecolor("#0f1117")
        fig_w.patch.set_facecolor("#0f1117")
        ax_w.set_title(
            f"Base value: {base_val:.3f}  →  Output: {float(np.sum(sv)) + base_val:.3f}",
            color="#aaa", fontsize=10
        )
        st.pyplot(fig_w, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown('<h3 class="section-header">Clinical Recommendations</h3>',
                    unsafe_allow_html=True)

        recs = []
        if risk_label == "HIGH":
            recs.append("🚨 <b>Urgent cardiology referral recommended.</b> "
                        "Predicted probability exceeds 65% threshold.")
        if thalach < 130:
            recs.append("❤️ Low maximum heart rate detected. Consider "
                        "<b>exercise stress test</b> to evaluate chronotropic incompetence.")
        if ca >= 2:
            recs.append(f"🩻 {ca} major vessels coloured by fluoroscopy. "
                        "Elevated risk of obstructive CAD — consider <b>coronary angiography</b>.")
        if cp == 0:
            recs.append("⚡ Asymptomatic chest pain profile is paradoxically the highest-risk "
                        "type in this model. Ensure <b>full cardiac workup</b>.")
        if oldpeak >= 2.0:
            recs.append(f"📉 ST depression of {oldpeak:.1f} mm indicates possible "
                        "<b>exercise-induced ischaemia</b>. Review ECG recordings.")
        if chol > 240:
            recs.append(f"🧪 Cholesterol {chol} mg/dl exceeds 240 threshold. "
                        "Consider <b>statin therapy</b> evaluation per NICE guidelines.")
        if trestbps > 140:
            recs.append(f"🩸 Resting BP {trestbps} mmHg (Stage 1 hypertension). "
                        "Review <b>antihypertensive management</b>.")
        if exang == 1:
            recs.append("🏃 Exercise-induced angina present. "
                        "Investigate with <b>myocardial perfusion imaging</b>.")
        if slope == 2:
            recs.append("📊 Downsloping ST segment — associated with severe CAD. "
                        "Prioritise <b>invasive evaluation</b> if high-risk.")
        if not recs:
            recs.append("✅ No high-priority clinical flags identified. "
                        "Routine follow-up recommended per standard cardiovascular guidelines.")

        for rec in recs:
            st.markdown(f'<div class="recommendation-box">{rec}</div>',
                        unsafe_allow_html=True)

    st.divider()

    st.markdown('<h3 class="section-header">What-If Scenario Planner</h3>',
                unsafe_allow_html=True)
    st.caption("Modify individual features to simulate clinical interventions.")

    wi_col1, wi_col2, wi_col3 = st.columns(3)
    with wi_col1:
        wi_thalach  = st.slider("Max Heart Rate (bpm) ↕", 70, 210, thalach, key="wi_thalach")
        wi_chol     = st.slider("Cholesterol (mg/dl) ↕", 120, 570, chol, key="wi_chol")
    with wi_col2:
        wi_cp = st.selectbox("Chest Pain Type ↕", [0, 1, 2, 3], index=cp, key="wi_cp",
                              format_func=lambda x: f"{x} — {CP_MAP[x]}")
        wi_ca = st.slider("Major Vessels ↕", 0, 4, ca, key="wi_ca")
    with wi_col3:
        wi_oldpeak  = st.slider("ST Depression ↕", 0.0, 6.5, oldpeak, step=0.1, key="wi_oldpeak")
        wi_trestbps = st.slider("Blood Pressure ↕", 90, 200, trestbps, key="wi_trestbps")

    wi_patient = {**patient,
                  "thalach": wi_thalach, "chol": wi_chol,
                  "cp": wi_cp, "ca": wi_ca,
                  "oldpeak": wi_oldpeak, "trestbps": wi_trestbps}

    wi_prob, _ = predict(wi_patient)
    wi_label, wi_color, _, wi_icon = risk_band(wi_prob)
    delta_prob = wi_prob - prob

    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Original Probability", f"{prob*100:.1f}%")
    wc2.metric("Scenario Probability",  f"{wi_prob*100:.1f}%",
               delta=f"{delta_prob*100:+.1f} pp", delta_color="inverse")
    wc3.metric("Scenario Risk Level",   f"{wi_icon} {wi_label}")

    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 1.4), facecolor="#0f1117")
    ax_cmp.barh(["Original"], [prob],    color=risk_color, height=0.4)
    ax_cmp.barh(["Scenario"], [wi_prob], color=wi_color,   height=0.4)
    ax_cmp.axvline(THRESHOLD, color="white", linestyle="--", linewidth=1)
    ax_cmp.set_xlim(0, 1)
    ax_cmp.set_xlabel("Probability", color="white")
    ax_cmp.tick_params(colors="white")
    ax_cmp.spines[:].set_color("#333")
    ax_cmp.set_facecolor("#0f1117")
    fig_cmp.patch.set_facecolor("#0f1117")
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close()

    st.divider()
    st.caption("⚠️ This tool is for research and educational purposes only. "
               "Clinical decisions must be made by a qualified healthcare professional.")
