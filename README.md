# 🩺 Explainable AI: Diabetes Risk & Healthcare Cost Predictor

> A dual-model ML framework that predicts diabetes risk AND annual healthcare costs in one unified pipeline — with SHAP explainability across both models.

**Live Demo:** [Click here to try the app](https://parishah2396-diabetes-cost-predictor.streamlit.app/)
*(replace this link after you deploy)*

---

## What this does

Most ML research either predicts disease risk **or** estimates healthcare costs — never both together. This framework bridges that gap with a novel **risk-score bridge** feature.

**Pipeline:**
```
PIMA Dataset (768 pts)
      ↓
Random Forest Classifier → Diabetes Risk Score (0–1)
      ↓
Bridge Mapper (BMI + Age → risk_score)
      ↓
Gradient Boosting Regressor → Annual Healthcare Cost ($)
      ↓
SHAP Explainability (both models)
```

## Results

| Task | Model | Metric |
|---|---|---|
| Diabetes Classification | Random Forest | Recall = 83%, AUC = 0.87 |
| Healthcare Cost Regression | Gradient Boosting | R² = 0.88, MAE = $2,428 |

Outperforms Patra et al. (2024): R² 0.88 vs 0.87, RMSE $4,306 vs $4,450.
Matches Shen (ICDSE 2025) while adding clinical bridge feature both papers identified as a gap.

## Novel Contribution

The `diabetes_risk_score` feature — transferred from the PIMA classifier via a BMI+Age bridge mapper — ranks **4th most important** in the cost model (ahead of children, sex, and all region variables combined).

## Tech stack

- Python · scikit-learn · XGBoost · SHAP · Streamlit
- SMOTE (imbalanced-learn) · pandas · NumPy · matplotlib

## Run locally

```bash
git clone https://github.com/your-username/diabetes-cost-predictor
cd diabetes-cost-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
diabetes-cost-predictor/
├── app.py                    ← Streamlit app
├── requirements.txt
├── README.md
├── models/
│   ├── rf_classifier.pkl     ← Random Forest (diabetes risk)
│   ├── gb_regressor.pkl      ← Gradient Boosting (cost)
│   ├── scaler.pkl            ← StandardScaler
│   └── bridge_mapper.pkl     ← BMI+Age → risk_score mapper
└── notebooks/
    └── PIMA_full_pipeline.ipynb
```

## Datasets

- [PIMA Indians Diabetes — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Medical Insurance Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

*Research project — Pari Shah*  

