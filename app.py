import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk & Healthcare Cost Predictor",
    page_icon="🩺",
    layout="wide"
)

# ── Load all saved models (cached so they load only once) ────
@st.cache_resource
def load_models():
    clf    = joblib.load('models/rf_classifier.pkl')   # Random Forest
    reg    = joblib.load('models/gb_regressor.pkl')    # Gradient Boosting
    scaler = joblib.load('models/scaler.pkl')          # StandardScaler
    mapper = joblib.load('models/bridge_mapper.pkl')   # Bridge mapper
    return clf, reg, scaler, mapper

clf, reg, scaler, mapper = load_models()

# ── PIMA feature column order (must match training) ──────────
PIMA_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# ── Insurance feature column order (must match training) ─────
# From notebook Cell 3: after get_dummies(region, drop_first=True)
# Region 'northeast' is dropped (it's the reference), so 3 dummies remain:
# region_northwest, region_southeast, region_southwest
INSURANCE_FEATURES = [
    'age', 'sex', 'bmi', 'children', 'smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'diabetes_risk_score'
]

# ── Header ────────────────────────────────────────────────────
st.title("🩺 Explainable AI: Diabetes Risk & Healthcare Cost Predictor")
st.caption("Research by Pari Shah")
st.markdown("---")

# ── Two tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Make a Prediction", "ℹ️ About This Project"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Step 1 — Enter Medical Details (for diabetes risk)")
    st.caption("These fields come from the PIMA Indians Diabetes Dataset")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies",
                                min_value=0, max_value=17, value=3,
                                help="Number of times pregnant")

        glucose = st.slider("Glucose (mg/dL)",
                            min_value=44, max_value=200, value=120,
                            help="Plasma glucose concentration (2-hour oral glucose tolerance test)")

        blood_pressure = st.slider("Blood Pressure (mm Hg)",
                                   min_value=24, max_value=122, value=70,
                                   help="Diastolic blood pressure")

        skin_thickness = st.slider("Skin Thickness (mm)",
                                   min_value=7, max_value=99, value=23,
                                   help="Triceps skin fold thickness")

    with col2:
        insulin = st.slider("Insulin (mu U/ml)",
                            min_value=14, max_value=846, value=80,
                            help="2-Hour serum insulin")

        bmi = st.slider("BMI",
                        min_value=18.0, max_value=67.0, value=32.0, step=0.1,
                        help="Body Mass Index (weight in kg / height in m²)")

        dpf = st.slider("Diabetes Pedigree Function",
                        min_value=0.08, max_value=2.42, value=0.47, step=0.01,
                        help="Scores likelihood of diabetes based on family history")

        age = st.slider("Age (years)",
                        min_value=21, max_value=81, value=33,
                        help="Age in years")

    st.markdown("---")
    st.subheader("Step 2 — Enter Insurance Details (for cost prediction)")
    st.caption("These fields come from the Kaggle Medical Insurance Dataset")

    col3, col4, col5 = st.columns(3)

    with col3:
        smoker_input = st.selectbox("Smoker?", ["No", "Yes"])
        smoker = 1 if smoker_input == "Yes" else 0

    with col4:
        sex_input = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex_input == "Female" else 0

    with col5:
        children = st.slider("Number of Children", min_value=0, max_value=5, value=1)

    region_input = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    region_northwest = 1 if region_input == "northwest" else 0
    region_southeast = 1 if region_input == "southeast" else 0
    region_southwest = 1 if region_input == "southwest" else 0

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────
    predict_btn = st.button("🚀 Predict Now", type="primary", use_container_width=True)

    if predict_btn:
        # ── MODEL A: Diabetes Risk Score ─────────────────────
        pima_input = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        pima_scaled = scaler.transform(pima_input)
        risk_prob   = clf.predict_proba(pima_scaled)[0][1]   # probability of class 1
        risk_pct    = risk_prob * 100

        if risk_prob >= 0.6:
            risk_label = "HIGH RISK"
            risk_color = "🔴"
        elif risk_prob >= 0.3:
            risk_label = "MODERATE RISK"
            risk_color = "🟡"
        else:
            risk_label = "LOW RISK"
            risk_color = "🟢"

        # ── BRIDGE: Generate mapped risk score for insurance ─
        bridge_input = pd.DataFrame({'BMI': [bmi], 'Age': [age]})
        mapped_risk  = float(np.clip(mapper.predict(bridge_input)[0], 0, 1))

        # ── MODEL B: Healthcare Cost Prediction ──────────────
        insurance_row = pd.DataFrame([{
            'age':               age,
            'sex':               sex,
            'bmi':               bmi,
            'children':          children,
            'smoker':            smoker,
            'region_northwest':  region_northwest,
            'region_southeast':  region_southeast,
            'region_southwest':  region_southwest,
            'diabetes_risk_score': mapped_risk
        }])[INSURANCE_FEATURES]   # enforce exact column order

        predicted_cost = reg.predict(insurance_row)[0]

        # ── Show results ──────────────────────────────────────
        st.markdown("## 📊 Results")

        res1, res2, res3 = st.columns(3)

        with res1:
            st.metric(
                label="Diabetes Risk Score",
                value=f"{risk_pct:.1f}%",
                delta=f"{risk_color} {risk_label}"
            )

        with res2:
            st.metric(
                label="Predicted Annual Healthcare Cost",
                value=f"${predicted_cost:,.0f}"
            )

        with res3:
            st.metric(
                label="Bridge Risk Score (mapped)",
                value=f"{mapped_risk:.3f}",
                help="Risk score transferred to the insurance model via BMI + Age bridge"
            )

        # ── Risk gauge bar ────────────────────────────────────
        st.markdown("#### Diabetes Risk Level")
        risk_color_hex = "#e74c3c" if risk_prob >= 0.6 else ("#f39c12" if risk_prob >= 0.3 else "#2ecc71")
        st.markdown(
            f"""
            <div style="background:#eee; border-radius:10px; height:24px; width:100%">
              <div style="background:{risk_color_hex}; width:{risk_pct:.1f}%;
                          height:24px; border-radius:10px;
                          display:flex; align-items:center; padding-left:8px;
                          color:white; font-weight:bold; font-size:13px;">
                {risk_pct:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Threshold: 0–30% = Low  |  30–60% = Moderate  |  60–100% = High")

        st.markdown("---")

        # ── SHAP explanation for cost model ──────────────────
        st.markdown("#### Why this cost? — SHAP Explanation (Model B)")
        st.caption("SHAP values show how much each feature pushed the predicted cost up or down")

        try:
            explainer_b = shap.TreeExplainer(reg)
            shap_vals   = explainer_b.shap_values(insurance_row)

            # Bar chart of feature contributions
            shap_df = pd.DataFrame({
                'Feature':     INSURANCE_FEATURES,
                'SHAP Value':  shap_vals[0]
            }).sort_values('SHAP Value', key=abs, ascending=True)

            colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df['SHAP Value']]

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on predicted cost in $)")
            ax.set_title("Feature contributions to predicted healthcare cost")
            for bar, val in zip(bars, shap_df['SHAP Value']):
                ax.text(
                    val + (200 if val >= 0 else -200),
                    bar.get_y() + bar.get_height() / 2,
                    f"${val:,.0f}", va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=8
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Readable interpretation
            st.markdown("**Top cost drivers for this patient:**")
            top3 = shap_df.sort_values('SHAP Value', key=abs, ascending=False).head(3)
            for _, row in top3.iterrows():
                direction = "increases" if row['SHAP Value'] > 0 else "decreases"
                st.write(f"- **{row['Feature']}** → {direction} cost by **${abs(row['SHAP Value']):,.0f}**")

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

        st.markdown("---")

        # ── SHAP explanation for diabetes model ──────────────
        st.markdown("#### Why this risk score? — SHAP Explanation (Model A)")

        try:
            pima_df     = pd.DataFrame(pima_scaled, columns=PIMA_FEATURES)
            explainer_a = shap.TreeExplainer(clf)
            shap_vals_a = explainer_a.shap_values(pima_df)

            # For RF binary: pick class 1 shap values
            if isinstance(shap_vals_a, list):
                sv = shap_vals_a[1][0]
            elif isinstance(shap_vals_a, np.ndarray) and shap_vals_a.ndim == 3:
                sv = shap_vals_a[0, :, 1]
            else:
                sv = shap_vals_a[0]

            shap_df_a = pd.DataFrame({
                'Feature':    PIMA_FEATURES,
                'SHAP Value': sv
            }).sort_values('SHAP Value', key=abs, ascending=True)

            colors_a = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df_a['SHAP Value']]

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars2 = ax2.barh(shap_df_a['Feature'], shap_df_a['SHAP Value'], color=colors_a)
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_xlabel("SHAP value (impact on diabetes risk probability)")
            ax2.set_title("Feature contributions to diabetes risk score")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP explanation for risk model could not be generated: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("About This Framework")

    st.markdown("""
    ### What this app does
    This is a **dual-model Explainable AI framework** that:
    1. Predicts your **diabetes risk** (0–100%) using a Random Forest classifier trained on the PIMA Indians dataset
    2. Predicts your **annual healthcare cost** (in USD) using a Gradient Boosting regressor trained on the Kaggle insurance dataset
    3. Explains **why** each prediction was made using SHAP (SHapley Additive exPlanations)

    ### The novel contribution — the Risk-Score Bridge
    The two datasets share no patients, but both have **BMI** and **Age**.
    We train a bridge mapper: *BMI + Age → diabetes_risk_score*
    This transfers medical knowledge from PIMA patients into the insurance cost model,
    making `diabetes_risk_score` the **4th most important feature** for cost prediction
    (ahead of children, sex, and all region columns combined).

    ### Model performance
    | Model | Task | Key Metric |
    |---|---|---|
    | Random Forest | Diabetes classification | Recall = 83% |
    | Gradient Boosting | Healthcare cost regression | R² = 0.88, MAE = $2,428 |

    ### Datasets used
    - **PIMA Indians Diabetes** — 768 patients, UCI ML Repository
    - **Kaggle Medical Insurance** — 1,338 patients (mirichoi0218)

    ### Research comparison
    This framework outperforms Patra et al. (2024) on R² (0.88 vs 0.87) and RMSE ($4,306 vs $4,450),
    and matches Shen (ICDSE 2025) while adding the novel diabetes risk bridge that both papers
    identified as a gap.
    """)
