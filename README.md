# 🏡 EstateIQ — Real Estate Investment Advisor
> **Transforming raw property listings into actionable investment intelligence using Python, Machine Learning, and Streamlit.**

EstateIQ is an end-to-end data science platform built to evaluate property profitability and project future market valuations across Indian real estate markets. It brings together clean data processing, robust exploratory data analysis, dual machine learning modeling (classification + regression), and MLflow experiment tracking inside an intuitive web application.

---

## 📌 Executive Summary

Investing in real estate requires evaluating dozens of interdependent variables—ranging from price per square foot and property age to neighborhood infrastructure and regional growth rates. Comparing properties manually often leads to inconsistent decisions.

**EstateIQ solves this by delivering:**
1. **Rule-Based & ML Investment Scoring:** Instantly flags whether a listing qualifies as a "Good Investment" based on a multi-factor evaluation framework.
2. **Multi-Year Valuation Forecasting:** Projects property values over a custom 1-to-10-year horizon using learned market features and compound appreciation models.
3. **Data-Driven Transparency:** Offers permutation feature importance, statistical hypothesis testing, and model performance auditing to explain *why* predictions are made.

---

## ✨ Key Features & Application Modules

The Streamlit app (`main.py`) is structured into six dedicated modules:

| Module | Core Functionality & Business Value |
| :--- | :--- |
| **📊 Overview** | High-level market metrics, dataset health report, model status cards, and an **interactive 1–10 year price growth calculator**. |
| **🔍 EDA (20 Questions)** | Deep dive into 20 market queries using interactive Plotly charts, descriptive tables, **ANOVA**, **Pearson/Spearman correlations**, and **Welch t-tests**. |
| **🤖 Investment Advisor** | Real-time property evaluation interface. Input property specs to get classification probability, rule criteria checklist, 5-year estimate, and prediction confidence bounds. |
| **🔎 Property Finder** | Market-wide search engine with custom multi-filters (budget, BHK, size, location). Ranks properties by model-scored investment likelihood with CSV export. |
| **📈 Market Insights** | Locality price heatmaps, property age vs. price trends, and infrastructure impact analysis (schools, hospitals, public transit). |
| **🧪 Model Lab** | Interactive model training trigger, champion vs. baseline performance comparisons, ROC curves, confusion matrices, and **MLflow tracking integration**. |

---
## 🔄 System Architecture & Workflow

The following flowchart details how raw real estate data moves through data cleaning, feature engineering, dual-model training, experiment tracking, and user interaction within the Streamlit interface:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        1. DATA INGESTION & CLEANING                     │
│  [india_housing_prices.csv] ──► Deduplication ──► Coerce Numerics      │
│                               ──► Price/SqFt Recomputation ──► Imputation │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       2. FEATURE ENGINEERING                           │
│  • Infrastructure Score (Schools, Hospitals, Public Transport)         │
│  • Amenity Counts, Parking & Security Flags                            │
│  • City-Relative Price Ratios (Price vs City Median PPSF)               │
│  • Target 1: Good Investment Label (Score ≥ 3 of 5 Rules)              │
│  • Target 2: 5-Year Scenario Growth Target                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       3. MODEL TRAINING & EVALUATION                   │
│                                                                        │
│   CLASSIFICATION (Good Investment)       REGRESSION (5Y Valuation)     │
│   ├─ Baseline: Logistic Regression       ├─ Baseline: Ridge Regression │
│   └─ Champion: XGBoost / HistGrad        └─ Champion: XGBoost / HistGrad│
│                                                                        │
│   Evaluated on Held-Out Test Set (F1, ROC-AUC, RMSE, MAE, R²)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   4. EXPERIMENT TRACKING & REGISTRY                    │
│  • MLflow Logs Parameters, Hyperparameters, Metrics & Feature Importance│
│  • SQLite Backend (`mlflow.db`) + Artifact Storage                      │
│  • Champions Saved to Bundle (`models/model_bundle.joblib`)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         5. STREAMLIT APPLICATION                       │
│  • Overview: High-level KPIs & Interactive 1–10 Year Price Forecast    │
│  • EDA: 20 Question Analytics (ANOVA, Correlations, Welch t-tests)     │
│  • Advisor: Single Property Evaluation & Gauge Charts                  │
│  • Finder: Multi-Filter Search Engine & Ranked Rankings Export         │
│  • Model Lab: Interactive Retraining Trigger & Confusion Matrices      │
└────────────────────────────────────────────────────────────────────────┘
```
---

## ⚙️ Machine Learning Pipeline & Methodology

EstateIQ uses two distinct machine learning tasks trained on engineered property features:

```
Raw CSV → Data Cleaning & Imputation → Feature Engineering → Pipeline Processing → Model Evaluation → MLflow Logging
```

### 1. Classification: *"Is this a Good Investment?"*
- **Target Logic:** A multi-factor rule engine evaluates 5 key metrics (Price/SqFt ≤ City Median, Total Price ≤ City Median, 3+ BHK, Ready-to-Move status, and Infrastructure Score ≥ Median). Meeting $\ge 3$ rules flags the property as a **Good Investment**.
- **Models Compared:** Logistic Regression *(Baseline)* vs. XGBoost / HistGradientBoosting *(Challenger)*.
- **Metrics Tracked:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### 2. Regression: *"5-Year Estimated Value"*
- **Target Logic:** Simulates compounding growth scenarios combining baseline annual growth ($8\%$) adjusted for local city demand, property type, infrastructure density, transit access, and property age.
- **Models Compared:** Ridge Regression *(Baseline)* vs. XGBoost / HistGradientBoosting *(Challenger)*.
- **Metrics Tracked:** RMSE, MAE, $R^2$, MAPE, and 90th percentile relative error bounds.

---

## 📂 Project Directory Structure

```text
Real Estate Investment Advisor/
│
├── main.py                   # Main Streamlit dashboard application
├── housing_pipeline.py       # Core data cleaning, feature engineering & model logic
├── train.py                  # CLI training script for offline background execution
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   ├── india_housing_prices.csv   # Raw housing dataset
│   └── processed_housing.csv     # Preprocessed dataset (generated on export)
│
├── models/
│   └── model_bundle.joblib        # Serialized pipeline, reference metrics & trained models
│
├── mlflow.db                 # SQLite database for MLflow experiment tracking
└── mlartifacts/              # Logged model artifacts and feature importance exports
```

---

## 🚀 Getting Started Locally

### Prerequisites
Make sure you have Python 3.9 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Real-Estate-Investment-Advisor.git
cd Real-Estate-Investment-Advisor
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
Place `india_housing_prices.csv` into the project root directory or inside the `data/` folder.

### 5. Train the Machine Learning Models (Optional)
You can train models directly from the Streamlit UI or run the CLI script:
```bash
python train.py --rows 60000
```

### 6. Launch the Application
```bash
streamlit run main.py
```
Open `http://localhost:8501` in your browser to start exploring.

---

## 📊 Tracking Experiments with MLflow

EstateIQ records training parameters, evaluation metrics, and feature importances to a local MLflow tracking server. To launch the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` to inspect experiment runs, compare parameters, and view registered champion models.

---

## ⚠️ Advisory Disclaimer

*EstateIQ is designed for educational, research, and decision-support purposes only. Machine learning outputs are statistical estimations based on historical data patterns and scenario assumptions. They do not constitute formal financial advice, legal guarantees, or appraisal values. Users should perform independent due diligence, physical inspections, and legal verification before entering into real estate transactions.*

---

## 👨‍💻 Author

**Ankan**  
*Data Science & Machine Learning Enthusiast*  
Interested in predictive analytics, real-world machine learning systems, and interactive business intelligence.

---
⭐ *If you find this project helpful, feel free to give it a star on GitHub!*
