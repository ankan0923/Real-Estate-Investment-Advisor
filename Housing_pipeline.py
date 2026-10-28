"""
EstateIQ - shared data preparation, feature engineering and modelling logic.

Used by both `main.py` (Streamlit app) and `train.py` (command-line training).

Project tasks (see the project brief):
  * Classification: is a property a "Good Investment"?
  * Regression:     estimated property price after 5 years.

Models compared (no Random Forest):
  * Classification: Logistic Regression (baseline)  vs  XGBoost
  * Regression:     Ridge (linear baseline)         vs  XGBoost
  XGBoost falls back to scikit-learn's HistGradientBoosting if it is not installed.

Experiment tracking: every model is logged to MLflow (local SQLite store), the
best model of each task is registered in the MLflow Model Registry.

Important methodology notes
  * The dataset has no observed prices from five years later. The "future price"
    target is therefore a documented growth *scenario* (see `growth_rates`).
  * The "Good Investment" label is a rule-based multi-factor score (see
    `investment_criteria`). Because the label is derived from features, high
    accuracy shows the model recovered the rule, not independent validation.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:  # XGBoost is optional at import time; availability is re-checked by a tiny fit.
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_IMPORTED = True
except Exception:  # pragma: no cover - depends on the user's environment
    _XGB_IMPORTED = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"
PROCESSED_PATH = BASE_DIR / "data" / "processed_housing.csv"
MLFLOW_URI = "sqlite:///" + (BASE_DIR / "mlflow.db").as_posix()
MLFLOW_ARTIFACTS = BASE_DIR / "mlartifacts"
MLFLOW_EXPERIMENT = "EstateIQ-Real-Estate-Advisor"
MLFLOW_CLF_NAME = "EstateIQ-Good-Investment-Classifier"
MLFLOW_REG_NAME = "EstateIQ-Future-Price-Regressor"

BUNDLE_VERSION = 2
RANDOM_STATE = 42
HORIZON_YEARS = 5
BASE_GROWTH = 0.08            # fixed-rate example from the project brief (r = 8%)
GOOD_SCORE_THRESHOLD = 3      # properties meeting >= 3 of 5 criteria are "Good"
DEFAULT_TRAIN_ROWS = 60000    # random sample used for model training (speed)
CURRENT_YEAR = datetime.now().year

NUMERIC_RAW = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
]
TEXT_RAW = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
]
REQUIRED = ["City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs"]
KEY_TEXT = ["State", "City", "Locality", "Property_Type"]   # missing -> "Unknown"
NO_TOKENS = {"", "no", "none", "nan", "null", "n/a", "na", "0", "false", "not available"}

# Model inputs (engineered features included)
NUM_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "Amenity_Count", "Parking_Flag", "Security_Flag", "Transport_Score",
    "Ready_To_Move", "School_Density_Score", "Infrastructure_Score",
    "Price_vs_City_Median", "Price_Ratio_vs_City",
]
CAT_FEATURES = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
                "Facing", "Owner_Type"]

CRITERIA_LABELS = {
    "Value_PPSF": "Price per sq ft is at or below the city median",
    "Value_Price": "Total price is at or below the city median",
    "Family_Size": "Family-sized home (3+ BHK)",
    "Ready_To_Move": "Ready to move (not under construction)",
    "Infrastructure": "Infrastructure score at or above the dataset median",
}


# ----------------------------------------------------------------------------
# Data loading and cleaning (Step 1 of the project brief)
# ----------------------------------------------------------------------------
def find_dataset() -> Path | None:
    names = ["india_housing_prices.csv", "India_Housing_Prices.csv"]
    folders = [BASE_DIR / "data", BASE_DIR, Path.cwd() / "data", Path.cwd()]
    for folder in folders:
        for name in names:
            p = folder / name
            if p.exists():
                return p
    return None


def _clean_text(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.mask(t.str.lower().isin(["", "nan", "none", "null", "na", "n/a"]))
    return t.astype(object).where(t.notna(), np.nan)


def _present_flag(s: pd.Series) -> np.ndarray:
    """1 if a parking/security feature is present (numeric > 0 or a 'yes'-like text)."""
    num = pd.to_numeric(s, errors="coerce")
    txt = s.astype("string").str.strip().str.lower().fillna("")
    text_yes = ~txt.isin(NO_TOKENS)
    return np.where(num.notna(), num > 0, text_yes).astype(int)


def _transport_score(s: pd.Series) -> pd.Series:
    """Low=1, Medium=2, High=3 (unknown -> 2)."""
    t = s.astype("string").str.strip().str.lower().fillna("")
    out = np.full(len(t), 2.0)
    out[t.str.contains("low", regex=False).to_numpy(dtype=bool)] = 1.0
    out[t.str.contains("med", regex=False).to_numpy(dtype=bool)] = 2.0
    out[t.str.contains("high", regex=False).to_numpy(dtype=bool)] = 3.0
    return pd.Series(out, index=s.index)


def _amenity_count(s: pd.Series) -> pd.Series:
    t = s.astype("string").fillna("").str.strip()
    cnt = t.str.count(",") + 1
    return cnt.where(t != "", 0).astype(float)


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip text, coerce numbers, drop duplicates/invalid rows, impute key text columns."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")

    report: dict = {"raw_rows": int(len(df))}
    report["absent_columns"] = [c for c in NUMERIC_RAW + TEXT_RAW if c not in df.columns]
    for c in report["absent_columns"]:
        df[c] = np.nan

    for c in TEXT_RAW:
        df[c] = _clean_text(df[c])
    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    report["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    # Duplicates (ignoring the ID column)
    subset = [c for c in df.columns if c != "ID"]
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    report["duplicates_removed"] = int(before - len(df))

    # Rows without a usable price or size cannot be analysed
    before = len(df)
    df = df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt"])
    df = df[(df["Price_in_Lakhs"] > 0) & (df["Size_in_SqFt"] > 0)]
    report["invalid_removed"] = int(before - len(df))

    # Derived / repaired numeric columns
    if df["Age_of_Property"].isna().any() and df["Year_Built"].notna().any():
        df["Age_of_Property"] = df["Age_of_Property"].fillna(CURRENT_YEAR - df["Year_Built"]).clip(lower=0)
    if df["Year_Built"].isna().any() and df["Age_of_Property"].notna().any():
        df["Year_Built"] = df["Year_Built"].fillna(CURRENT_YEAR - df["Age_of_Property"])

    # Price per sq ft is recomputed (Rs per sq ft) so its unit is always consistent
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    # Key grouping columns: missing -> "Unknown" (other columns are imputed in the model pipeline)
    for c in KEY_TEXT:
        df[c] = df[c].fillna("Unknown")

    report["clean_rows"] = int(len(df))
    return df.reset_index(drop=True), report


def _demand_index(medians: pd.Series) -> dict:
    """Standardised market index in [-1, 1]: >0 means pricier than the average market."""
    sd = medians.std(ddof=0)
    sd = sd if sd and np.isfinite(sd) else 1.0
    return ((medians - medians.mean()) / sd).clip(-1.5, 1.5).div(1.5).to_dict()


def _infra_raw(schools, hospitals, transport, ref) -> pd.Series:
    sch = (schools / ref["school_hi"]).clip(0, 1)
    hos = (hospitals / ref["hosp_hi"]).clip(0, 1)
    tr = (transport - 1) / 2
    return 10 * (0.35 * sch + 0.25 * hos + 0.40 * tr)


def build_reference(df: pd.DataFrame) -> dict:
    """Lookup tables learned from the full dataset (used for features, labels, new listings)."""
    ref: dict = {
        "base_growth": BASE_GROWTH, "horizon": HORIZON_YEARS, "threshold": GOOD_SCORE_THRESHOLD,
        "global_ppsf": float(df["Price_per_SqFt"].median()),
        "global_price": float(df["Price_in_Lakhs"].median()),
    }
    sch_med = df["Nearby_Schools"].median()
    ref["global_schools"] = float(sch_med) if pd.notna(sch_med) and sch_med > 0 else 1.0
    ref["city_ppsf"] = df.groupby("City")["Price_per_SqFt"].median().to_dict()
    ref["city_price"] = df.groupby("City")["Price_in_Lakhs"].median().to_dict()
    ref["city_schools"] = df.groupby("City")["Nearby_Schools"].median().to_dict()
    ref["city_demand"] = _demand_index(df.groupby("City")["Price_per_SqFt"].median())
    ref["type_demand"] = _demand_index(df.groupby("Property_Type")["Price_per_SqFt"].median())

    q = df["Nearby_Schools"].quantile(0.99)
    ref["school_hi"] = float(q) if pd.notna(q) and q > 0 else 10.0
    q = df["Nearby_Hospitals"].quantile(0.99)
    ref["hosp_hi"] = float(q) if pd.notna(q) and q > 0 else 10.0

    fences = {}
    for col in ("Price_per_SqFt", "Size_in_SqFt"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        fences[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    ref["fences"] = fences

    infra = _infra_raw(df["Nearby_Schools"], df["Nearby_Hospitals"],
                       _transport_score(df["Public_Transport_Accessibility"]), ref)
    ref["infra_median"] = float(np.nanmedian(infra)) if infra.notna().any() else 5.0
    return ref


def add_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Feature engineering - works for the whole dataset and for a single new listing."""
    d = df.copy()
    for c in KEY_TEXT:
        d[c] = d[c].fillna("Unknown")
    d["Price_per_SqFt"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    d["Amenity_Count"] = _amenity_count(d["Amenities"])
    d["Parking_Flag"] = _present_flag(d["Parking_Space"])
    d["Security_Flag"] = _present_flag(d["Security"])
    d["Transport_Score"] = _transport_score(d["Public_Transport_Accessibility"])
    ready = d["Availability_Status"].astype("string").str.lower().str.contains("ready", regex=False)
    d["Ready_To_Move"] = ready.fillna(False).astype(int)
    d["Floor_Ratio"] = (d["Floor_No"] / d["Total_Floors"].where(d["Total_Floors"] > 0)).clip(0, 1)

    city = d["City"]
    d["City_Median_PPSF"] = city.map(ref["city_ppsf"]).fillna(ref["global_ppsf"]).astype(float)
    d["City_Median_Price"] = city.map(ref["city_price"]).fillna(ref["global_price"]).astype(float)
    d["Price_vs_City_Median"] = d["Price_per_SqFt"] / d["City_Median_PPSF"]
    d["Price_Ratio_vs_City"] = d["Price_in_Lakhs"] / d["City_Median_Price"]

    city_schools = city.map(ref["city_schools"]).fillna(ref["global_schools"]).astype(float)
    d["School_Density_Score"] = (d["Nearby_Schools"] / city_schools.replace(0, np.nan)).clip(0, 5)
    d["Infrastructure_Score"] = _infra_raw(
        d["Nearby_Schools"], d["Nearby_Hospitals"], d["Transport_Score"], ref
    ).fillna(ref["infra_median"])

    lo, hi = ref["fences"]["Price_per_SqFt"]
    d["Outlier_PPSF"] = (d["Price_per_SqFt"] < lo) | (d["Price_per_SqFt"] > hi)
    lo, hi = ref["fences"]["Size_in_SqFt"]
    d["Outlier_Size"] = (d["Size_in_SqFt"] < lo) | (d["Size_in_SqFt"] > hi)
    return d


# ----------------------------------------------------------------------------
# Targets: "Good Investment" label and 5-year growth scenarios
# ----------------------------------------------------------------------------
def investment_criteria(d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """The five yes/no tests behind the multi-factor investment score."""
    return pd.DataFrame({
        "Value_PPSF": d["Price_per_SqFt"] <= d["City_Median_PPSF"],
        "Value_Price": d["Price_in_Lakhs"] <= d["City_Median_Price"],
        "Family_Size": d["BHK"] >= 3,
        "Ready_To_Move": d["Ready_To_Move"] == 1,
        "Infrastructure": d["Infrastructure_Score"] >= ref["infra_median"],
    }, index=d.index)


def add_label(d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    crit = investment_criteria(d, ref)
    d["Investment_Score"] = crit.sum(axis=1).astype(int)
    d["Good_Investment"] = (d["Investment_Score"] >= ref["threshold"]).astype(int)
    return d


def growth_rates(d: pd.DataFrame, ref: dict, method: str = "feature", base: float | None = None) -> pd.Series:
    """Annual growth-rate assumption per property.

    fixed    : the same rate for every property (project brief: r = 8%).
    location : base rate +/-2 pp by city market level and +/-1 pp by property type
               (markets with a higher median Rs/sq ft are assumed to grow a little faster).
    feature  : the location rate adjusted by property features - infrastructure
               (+/-1 pp), high public transport (+0.5 pp), age (up to -1 pp) and
               ready-to-move (+0.4 pp). This is the regression training target.
    """
    base = ref["base_growth"] if base is None else base
    if method == "fixed":
        return pd.Series(base, index=d.index, dtype=float)
    city_idx = d["City"].map(ref["city_demand"]).fillna(0.0).astype(float)
    type_idx = d["Property_Type"].map(ref["type_demand"]).fillna(0.0).astype(float)
    r = base + 0.02 * city_idx + 0.01 * type_idx
    if method == "location":
        return r.clip(-0.10, 0.30)
    infra = ((d["Infrastructure_Score"] - 5) / 5).clip(-1, 1)
    age_pen = (d["Age_of_Property"].fillna(10) / 30).clip(0, 1)
    r = (r + 0.010 * infra + 0.005 * (d["Transport_Score"] == 3).astype(float)
         - 0.010 * age_pen + 0.004 * d["Ready_To_Move"])
    return r.clip(-0.10, 0.30)


def future_value(price, rate, years: int = HORIZON_YEARS):
    return price * (1 + rate) ** years


def prepare_dataset(path) -> tuple[pd.DataFrame, dict, dict]:
    """CSV -> cleaned dataframe with engineered features, label and scenario target."""
    raw = pd.read_csv(path, low_memory=False)
    df, report = clean_dataset(raw)
    ref = build_reference(df)
    df = add_features(df, ref)
    df = add_label(df, ref)
    df["Growth_Rate_Feature"] = growth_rates(df, ref, "feature")
    df["Future_Price_5Y"] = future_value(df["Price_in_Lakhs"], df["Growth_Rate_Feature"], HORIZON_YEARS)
    report["good_share"] = float(df["Good_Investment"].mean() * 100)
    report["outliers_ppsf"] = int(df["Outlier_PPSF"].sum())
    report["outliers_size"] = int(df["Outlier_Size"].sum())
    return df, ref, report


# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------
def prepare_X(d: pd.DataFrame) -> pd.DataFrame:
    """Model input frame: float numerics + object categoricals (NaN for missing)."""
    X = pd.DataFrame(index=d.index)
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in CAT_FEATURES:
        s = d[c].astype(object)
        X[c] = s.where(pd.notna(s), np.nan)
    return X


@lru_cache(maxsize=1)
def xgboost_available() -> bool:
    if not _XGB_IMPORTED:
        return False
    try:
        XGBClassifier(n_estimators=2, n_jobs=1).fit(
            np.random.RandomState(0).rand(20, 3), np.array([0, 1] * 10))
        return True
    except Exception:
        return False


def boosted_name() -> str:
    return "XGBoost" if xgboost_available() else "HistGradientBoosting"


def _preprocessor(linear: bool) -> ColumnTransformer:
    """Impute, then scale+one-hot (linear models) or ordinal-encode (tree models)."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if linear:
        num_steps.append(("scaler", StandardScaler()))
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
    else:
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", cat_enc)]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_FEATURES),
        ("cat", Pipeline(cat_steps), CAT_FEATURES),
    ])


def make_classifiers() -> dict:
    models = {"Logistic Regression": Pipeline([
        ("prep", _preprocessor(True)),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])}
    if xgboost_available():
        boosted = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE)
    else:
        boosted = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6, random_state=RANDOM_STATE)
    models[boosted_name()] = Pipeline([("prep", _preprocessor(False)), ("model", boosted)])
    return models


def make_regressors() -> dict:
    models = {"Ridge Regression": Pipeline([
        ("prep", _preprocessor(True)),
        ("model", Ridge(alpha=1.0)),
    ])}
    if xgboost_available():
        boosted = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    else:
        boosted = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, max_depth=6, random_state=RANDOM_STATE)
    models[boosted_name()] = Pipeline([("prep", _preprocessor(False)), ("model", boosted)])
    return models


def _importance(pipe, X_te, y_te, scoring, n=3000) -> pd.DataFrame:
    """Permutation importance on raw input columns (model-agnostic, test data)."""
    rs = np.random.RandomState(RANDOM_STATE)
    take = rs.choice(len(X_te), size=min(n, len(X_te)), replace=False)
    r = permutation_importance(pipe, X_te.iloc[take], y_te[take], scoring=scoring,
                               n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    out = pd.DataFrame({"Feature": X_te.columns, "Importance": r.importances_mean,
                        "Std": r.importances_std})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def train_all(df: pd.DataFrame, ref: dict, max_rows: int = DEFAULT_TRAIN_ROWS,
              log_mlflow: bool = True, progress=None) -> dict:
    """Train and evaluate all candidate models, optionally log them to MLflow."""
    def say(msg: str, frac: float):
        if progress:
            progress(msg, frac)

    if len(df) < 200:
        raise ValueError("At least 200 rows are needed to train the models.")
    sample = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=RANDOM_STATE)
    say("Preparing training data…", 0.02)
    X = prepare_X(sample)
    y_clf = sample["Good_Investment"].to_numpy()
    y_reg = sample["Future_Price_5Y"].to_numpy()
    idx = np.arange(len(sample))
    stratify = y_clf if len(np.unique(y_clf)) > 1 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    plot_idx = np.random.RandomState(RANDOM_STATE).choice(len(te), size=min(2000, len(te)), replace=False)

    bundle: dict = {
        "version": BUNDLE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "ref": ref,
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "sample_rows": int(len(sample)), "total_rows": int(len(df)),
        "boosted_name": boosted_name(),
        "clf": None, "reg": None, "mlflow": {"enabled": False, "error": None},
    }

    # ---------------- Classification ----------------
    if len(np.unique(y_clf[tr])) > 1:
        models, metrics, imps = {}, {}, {}
        candidates = make_classifiers()
        for i, (name, pipe) in enumerate(candidates.items()):
            say(f"Training classifier: {name}", 0.08 + 0.22 * i)
            t0 = time.time()
            pipe.fit(X_tr, y_clf[tr])
            fit_s = time.time() - t0
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_clf[te], proba)
            keep = np.unique(np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int))
            metrics[name] = {
                "accuracy": float(accuracy_score(y_clf[te], pred)),
                "precision": float(precision_score(y_clf[te], pred, zero_division=0)),
                "recall": float(recall_score(y_clf[te], pred, zero_division=0)),
                "f1": float(f1_score(y_clf[te], pred, zero_division=0)),
                "auc": float(roc_auc_score(y_clf[te], proba)),
                "cm": confusion_matrix(y_clf[te], pred),
                "fpr": fpr[keep], "tpr": tpr[keep], "fit_seconds": float(fit_s),
            }
            say(f"Feature importance: {name}", 0.14 + 0.22 * i)
            imps[name] = _importance(pipe, X_te, y_clf[te], "roc_auc")
            models[name] = pipe
        champion = max(metrics, key=lambda n: (metrics[n]["f1"], metrics[n]["auc"]))
        bundle["clf"] = {"models": models, "metrics": metrics, "importance": imps, "champion": champion}

    # ---------------- Regression ----------------
    models, metrics, imps = {}, {}, {}
    for i, (name, pipe) in enumerate(make_regressors().items()):
        say(f"Training regressor: {name}", 0.52 + 0.17 * i)
        t0 = time.time()
        pipe.fit(X_tr, y_reg[tr])
        fit_s = time.time() - t0
        pred = pipe.predict(X_te)
        y_true = y_reg[te]
        rel = np.abs(pred - y_true) / np.maximum(y_true, 1e-9)
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)),
            "mape": float(rel.mean() * 100),
            "rel_err_q90": float(np.quantile(rel, 0.9)),
            "y_test": y_true[plot_idx], "y_pred": pred[plot_idx],
            "fit_seconds": float(fit_s),
        }
        say(f"Feature importance: {name}", 0.58 + 0.17 * i)
        imps[name] = _importance(pipe, X_te, y_true, "r2")
        models[name] = pipe
    champion = min(metrics, key=lambda n: metrics[n]["rmse"])
    bundle["reg"] = {"models": models, "metrics": metrics, "importance": imps, "champion": champion}

    # ---------------- MLflow ----------------
    if log_mlflow:
        say("Logging experiments to MLflow…", 0.9)
        bundle["mlflow"] = _log_to_mlflow(bundle)
    say("Done", 1.0)
    return bundle


# ----------------------------------------------------------------------------
# MLflow tracking
# ----------------------------------------------------------------------------
def _log_to_mlflow(bundle: dict) -> dict:
    status = {"enabled": False, "uri": MLFLOW_URI, "experiment": MLFLOW_EXPERIMENT,
              "runs": 0, "error": None}
    try:
        import mlflow
        import mlflow.sklearn
        import sklearn

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        if client.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
            client.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        reqs = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}"]
        if xgboost_available():
            import xgboost
            reqs.append(f"xgboost=={xgboost.__version__}")

        def log_model(pipe, registered_name):
            try:
                mlflow.sklearn.log_model(pipe, name="model", pip_requirements=reqs,
                                         registered_model_name=registered_name)
            except TypeError:  # MLflow 2.x uses artifact_path instead of name
                mlflow.sklearn.log_model(pipe, artifact_path="model", pip_requirements=reqs,
                                         registered_model_name=registered_name)

        for task, key, reg_name in (("classification", "clf", MLFLOW_CLF_NAME),
                                    ("regression", "reg", MLFLOW_REG_NAME)):
            block = bundle.get(key)
            if not block:
                continue
            for name, pipe in block["models"].items():
                is_champ = name == block["champion"]
                with mlflow.start_run(run_name=f"{task}-{name}"):
                    mlflow.set_tags({"task": task, "model": name, "champion": str(is_champ)})
                    mlflow.log_params({
                        "train_rows": bundle["train_rows"], "test_rows": bundle["test_rows"],
                        "n_numeric_features": len(NUM_FEATURES),
                        "n_categorical_features": len(CAT_FEATURES),
                        "horizon_years": HORIZON_YEARS, "base_growth": BASE_GROWTH,
                        "good_score_threshold": GOOD_SCORE_THRESHOLD,
                        "estimator": type(pipe.named_steps["model"]).__name__,
                    })
                    hyper = pipe.named_steps["model"].get_params()
                    mlflow.log_params({f"hp_{k}": str(v) for k, v in hyper.items()
                                       if k in ("C", "alpha", "n_estimators", "max_iter", "max_depth",
                                                "learning_rate", "subsample", "colsample_bytree",
                                                "class_weight")})
                    m = block["metrics"][name]
                    scalar = {k: v for k, v in m.items() if isinstance(v, float)}
                    mlflow.log_metrics(scalar)
                    mlflow.log_text(block["importance"][name].to_csv(index=False),
                                    "feature_importance.csv")
                    try:
                        log_model(pipe, reg_name if is_champ else None)
                    except Exception as exc:  # keep the run even if model logging fails
                        mlflow.set_tag("model_logging_error", str(exc)[:200])
                    status["runs"] += 1
        status["enabled"] = True
    except Exception as exc:  # MLflow missing or misconfigured: training still succeeds
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def mlflow_runs(limit: int = 30) -> tuple[pd.DataFrame | None, str | None]:
    """Tidy table of the latest MLflow runs for the Model Lab page."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            return None, "No MLflow runs yet. Train the models to create the first runs."
        runs = mlflow.search_runs([exp.experiment_id], max_results=limit, order_by=["start_time DESC"])
        if runs.empty:
            return None, "No MLflow runs yet."
        cols = {"tags.task": "Task", "tags.model": "Model", "tags.champion": "Champion",
                "metrics.accuracy": "Accuracy", "metrics.f1": "F1", "metrics.auc": "ROC-AUC",
                "metrics.rmse": "RMSE", "metrics.mae": "MAE", "metrics.r2": "R²",
                "metrics.fit_seconds": "Fit s", "start_time": "Started"}
        out = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
        return out, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------
# Saving / loading / prediction helpers
# ----------------------------------------------------------------------------
def save_bundle(bundle: dict) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    return BUNDLE_PATH


def load_bundle() -> dict | None:
    if not BUNDLE_PATH.exists():
        return None
    try:
        bundle = joblib.load(BUNDLE_PATH)
    except Exception:
        return None
    return bundle if isinstance(bundle, dict) and bundle.get("version") == BUNDLE_VERSION else None


def predict_good_probability(bundle: dict, name: str, d: pd.DataFrame) -> np.ndarray:
    return bundle["clf"]["models"][name].predict_proba(prepare_X(d))[:, 1]


def predict_future_price(bundle: dict, name: str, d: pd.DataFrame) -> np.ndarray:
    return bundle["reg"]["models"][name].predict(prepare_X(d))
