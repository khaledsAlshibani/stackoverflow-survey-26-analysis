"""Regenerate README and report figures from cleaned survey data."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from utils import (  # noqa: E402
    build_pipeline,
    get_classification_split,
    load_ml_dataframe,
)

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")


def save_summary_tables(df: pd.DataFrame) -> None:
    exp = (
        df.groupby("WorkExp")["ConvertedCompYearly"]
        .agg(["count", "mean", "median"])
        .round(2)
        .reset_index()
    )
    ed = (
        df.groupby("EdLevel")["ConvertedCompYearly"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("median", ascending=False)
        .reset_index()
    )
    learn = (
        df.groupby("LearnCodeChoose")["ConvertedCompYearly"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("median", ascending=False)
        .reset_index()
    )
    exp.to_csv(os.path.join(DATA, "results_experience_salary.csv"), index=False)
    ed.to_csv(os.path.join(DATA, "results_education_salary.csv"), index=False)
    learn.to_csv(os.path.join(DATA, "results_learningchoice_salary.csv"), index=False)


def plot_salary_vs_experience(exp: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(exp["WorkExp"], exp["mean"], marker="o", label="mean")
    plt.plot(exp["WorkExp"], exp["median"], marker="o", label="median")
    plt.xlabel("work experience")
    plt.ylabel("yearly salary")
    plt.title("salary vs work experience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_vs_experience.png"), dpi=120)
    plt.close()


def plot_salary_by_education(ed: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.barh(ed["EdLevel"], ed["median"])
    plt.xlabel("median yearly salary")
    plt.ylabel("education")
    plt.title("median salary by education")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_by_education.png"), dpi=120)
    plt.close()


def plot_salary_by_learning(learn: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.barh(learn["LearnCodeChoose"], learn["median"])
    plt.xlabel("median yearly salary")
    plt.ylabel("LearnCodeChoose")
    plt.title("median salary by learning choice")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_by_learning_choice.png"), dpi=120)
    plt.close()


def plot_cert_learning_lines(df: pd.DataFrame) -> None:
    work = df.copy()
    if "HasCertificationLearning" not in work.columns:
        work["HasCertificationLearning"] = work["LearnCode"].str.contains(
            "Online Courses or Certification",
            case=False,
            na=False,
        )
    line_data = (
        work.groupby(["WorkExp", "HasCertificationLearning"])["ConvertedCompYearly"]
        .median()
        .reset_index()
    )
    yes_data = line_data[line_data["HasCertificationLearning"]]
    no_data = line_data[~line_data["HasCertificationLearning"]]

    plt.figure(figsize=(12, 5))
    plt.plot(
        no_data["WorkExp"],
        no_data["ConvertedCompYearly"],
        marker="o",
        label="no cert learning",
    )
    plt.plot(
        yes_data["WorkExp"],
        yes_data["ConvertedCompYearly"],
        marker="o",
        label="has cert learning",
    )
    plt.xlabel("work experience")
    plt.ylabel("median salary")
    plt.title("salary by experience and cert learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_cert_learning_vs_experience.png"), dpi=120)
    plt.close()


def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(df["WorkExp"], vert=False)
    axes[0].set_title("work experience")
    axes[1].boxplot(df["ConvertedCompYearly"], vert=False)
    axes[1].set_title("yearly salary")
    axes[1].set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "outlier_boxplots.png"), dpi=120)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["low", "high"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"pred {labels[j]}" for j in range(2)])
    ax.set_yticklabels([f"actual {labels[i]}" for i in range(2)])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=120)
    plt.close()


def plot_classifier_f1(metrics_rows: list[dict]) -> None:
    names = [row["label"] for row in metrics_rows]
    scores = [row["f1_score"] for row in metrics_rows]
    plt.figure(figsize=(8, 4))
    plt.bar(names, scores, color=["#4c78a8", "#72b7b2"])
    plt.ylabel("f1 score")
    plt.title("classifier f1 on test set")
    plt.ylim(0, 1)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "classifier_f1_comparison.png"), dpi=120)
    plt.close()


def train_and_plot_classifiers() -> list[dict]:
    X_train, X_test, y_train, y_test = get_classification_split()
    models = [
        (
            "logistic regression",
            "confusion_matrix_logistic.png",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        (
            "random forest",
            "confusion_matrix_random_forest.png",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=10,
                class_weight="balanced",
            ),
        ),
    ]
    metrics_rows = []
    for label, cm_file, estimator in models:
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, f"{label} confusion matrix", cm_file)
        metrics_rows.append(
            {
                "label": label,
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            }
        )
    return metrics_rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cleaned_path = os.path.join(DATA, "cleaned_developer_survey.csv")
    if not os.path.isfile(cleaned_path):
        raise FileNotFoundError(
            f"Missing {cleaned_path}. Run cleaning in analysis.ipynb first."
        )

    df = pd.read_csv(cleaned_path)
    save_summary_tables(df)

    exp = pd.read_csv(os.path.join(DATA, "results_experience_salary.csv"))
    ed = pd.read_csv(os.path.join(DATA, "results_education_salary.csv"))
    learn = pd.read_csv(os.path.join(DATA, "results_learningchoice_salary.csv"))

    plot_salary_vs_experience(exp)
    plot_salary_by_education(ed)
    plot_salary_by_learning(learn)
    plot_cert_learning_lines(df)
    plot_outlier_boxplots(df)

    load_ml_dataframe()
    metrics_rows = train_and_plot_classifiers()
    plot_classifier_f1(metrics_rows)

    written = sorted(f for f in os.listdir(OUT) if f.endswith(".png"))
    print("wrote", len(written), "png files to", OUT)
    for name in written:
        print(" ", name)


if __name__ == "__main__":
    main()
