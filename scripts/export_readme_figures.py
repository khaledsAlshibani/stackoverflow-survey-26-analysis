"""Regenerate main analysis plots as PNG files for the README."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    exp = pd.read_csv(os.path.join(DATA, "results_experience_salary.csv"))
    ed = pd.read_csv(os.path.join(DATA, "results_education_salary.csv"))
    learn = pd.read_csv(os.path.join(DATA, "results_learningchoice_salary.csv"))
    df = pd.read_csv(os.path.join(DATA, "cleaned_developer_survey.csv"))

    plt.figure(figsize=(12, 5))
    plt.plot(exp["WorkExp"], exp["mean"], marker="o", label="Mean salary")
    plt.plot(exp["WorkExp"], exp["median"], marker="o", label="Median salary")
    plt.xlabel("Work experience (years)")
    plt.ylabel("Yearly salary (USD)")
    plt.title("Salary vs work experience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_vs_experience.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.barh(ed["EdLevel"], ed["median"])
    plt.xlabel("Median yearly salary (USD)")
    plt.ylabel("Education level")
    plt.title("Median salary by education level")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_by_education.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.barh(learn["LearnCodeChoose"], learn["median"])
    plt.xlabel("Median yearly salary (USD)")
    plt.ylabel("LearnCodeChoose")
    plt.title("Median salary by learning choice")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_by_learning_choice.png"), dpi=120)
    plt.close()

    df = df.copy()
    df["HasCertificationLearning"] = df["LearnCode"].str.contains(
        "Online Courses or Certification",
        case=False,
        na=False,
    )
    line_data = (
        df.groupby(["WorkExp", "HasCertificationLearning"])["ConvertedCompYearly"]
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
        label="No certification-style learning",
    )
    plt.plot(
        yes_data["WorkExp"],
        yes_data["ConvertedCompYearly"],
        marker="o",
        label="Has certification-style learning",
    )
    plt.xlabel("Work experience (years)")
    plt.ylabel("Median salary (USD)")
    plt.title("Median salary by experience and certification-related learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "salary_cert_learning_vs_experience.png"), dpi=120)
    plt.close()

    print("Wrote PNG files to", OUT)


if __name__ == "__main__":
    main()
