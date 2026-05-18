"""Shared data prep for model notebooks (same split and preprocessing everywhere)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CLEANED_PATH = ROOT / "data" / "cleaned_developer_survey.csv"

FEATURES = [
    "WorkExp",
    "EdLevel",
    "LearnCodeChoose",
    "LearnCode",
    "HasCertificationLearning",
]
TARGET = "ConvertedCompYearly"
NUMERIC_FEATURES = ["WorkExp"]
CATEGORICAL_FEATURES = [
    "EdLevel",
    "LearnCodeChoose",
    "LearnCode",
    "HasCertificationLearning",
]


def load_ml_dataframe() -> pd.DataFrame:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLEANED_PATH}. Run cleaning cells in analysis.ipynb first."
        )

    ml_df = pd.read_csv(CLEANED_PATH)

    if "HasCertificationLearning" not in ml_df.columns and "LearnCode" in ml_df.columns:
        ml_df["HasCertificationLearning"] = ml_df["LearnCode"].str.contains(
            "Online Courses or Certification",
            case=False,
            na=False,
        )

    ml_df = ml_df[FEATURES + [TARGET]].copy()

    ml_df["WorkExp"] = pd.to_numeric(ml_df["WorkExp"], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col == "HasCertificationLearning":
            continue
        mode_vals = ml_df[col].mode(dropna=True)
        fill_value = mode_vals.iloc[0] if len(mode_vals) else ""
        ml_df[col] = ml_df[col].fillna(fill_value).astype(str)

    ml_df = ml_df.dropna(subset=[TARGET, "WorkExp"])
    return ml_df


def split_train_test(ml_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    x = ml_df[FEATURES]
    y = ml_df[TARGET]
    y_strata = pd.qcut(y, q=4, labels=False, duplicates="drop")

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_strata,
    )


def build_preprocess() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocess()),
            ("model", model),
        ]
    )


def evaluate_regression(y_test, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }


def get_train_test_split():
    ml_df = load_ml_dataframe()
    return split_train_test(ml_df)


def make_high_salary_label(ml_df: pd.DataFrame) -> pd.Series:
    """Binary target for classification: 1 = at/above median salary, 0 = below."""
    median_salary = ml_df[TARGET].median()
    return (ml_df[TARGET] >= median_salary).astype(int)


def split_train_test_classification(
    ml_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    x = ml_df[FEATURES]
    y = make_high_salary_label(ml_df)

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_classification_split():
    ml_df = load_ml_dataframe()
    return split_train_test_classification(ml_df)


def evaluate_classification(y_test, y_pred, model_name: str = "classifier") -> dict:
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_score_0": report["0"]["f1-score"],
        "support_0": int(report["0"]["support"]),
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_score_1": report["1"]["f1-score"],
        "support_1": int(report["1"]["support"]),
    }
    return metrics, report
