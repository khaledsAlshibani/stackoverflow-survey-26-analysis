# Stack Overflow Developer Survey

First Python and data practice. Not meant to be anything serious, just learning.

## What the notebook looks at

`analysis.ipynb` asks one thing: what lines up most with salary in this survey, years of experience, school level, or how people answered the learning and certification style questions.

## The raw table

First step was just to see what is in the file: row count, column count, what the important names mean, what helps the salary question, what can be ignored.

The public CSV had 49123 rows and 170 columns. `survey_results_schema.csv` in `data/` explains the columns. Everything else was dropped except pay and a few background columns.

## Columns that stayed

Five columns stayed in the notebook.

`WorkExp` is years on the job. `EdLevel` is highest school level. `LearnCodeChoose` is whether someone is new to coding or still adding languages or skills. `LearnCode` is how they said they learn, including online courses and certs mixed with other options. `ConvertedCompYearly` is yearly pay in USD.

## Cleaning

Rows with no salary were removed. Rows missing `WorkExp`, `EdLevel`, or `LearnCodeChoose` were removed too. Blank `LearnCode` became the text `Unknown`. Salary outliers were cut with the usual IQR fences on `ConvertedCompYearly` (1.5 times the IQR past Q1 and Q3). That went from 23422 rows to 22122. The cleaned table is `data/cleaned_developer_survey.csv`.

Big raw files from Kaggle are listed in `.gitignore` so they do not land in git. `data/readme.md` links the Kaggle page.

## What showed up in the numbers

All of this is from the summary CSVs after cleaning down to 22122 rows. The survey is a single point in time, so it does not show that one factor causes another.

Pay goes up a lot over the first ten years of `WorkExp`. Two example medians from the table: about 20342 USD after one year, about 79261 USD after ten years. After that the rise is smaller and the very high year values rest on fewer answers each.

<img width="1440" height="600" alt="salary_vs_experience" src="https://github.com/user-attachments/assets/2deb11d1-a41b-465b-9f45-a39a5e18c687" />

Median pay is highest for people who picked a professional degree (about 83531 USD in the export). It is lower for master’s, then bachelor’s, associate, some college, and lower again for secondary and primary answers.

<img width="1440" height="720" alt="salary_by_education" src="https://github.com/user-attachments/assets/845dffe7-0c20-44d3-aaae-4cf3fb1cee28" />

For `LearnCodeChoose`, the group that said they are not new to coding and are not learning new languages or techniques right now has the highest median here (about 81870 USD). The group that is experienced but still learning sits around 69609 USD median. The small group that said they are new or still a student is much lower (about 10000 USD median).

<img width="1440" height="720" alt="salary_by_learning_choice" src="https://github.com/user-attachments/assets/019c1390-d015-4322-b769-160e2795975b" />

Median pay by years of experience, split by whether `LearnCode` mentions online courses or certification (same check as in the notebook):

<img width="1440" height="600" alt="salary_cert_learning_vs_experience" src="https://github.com/user-attachments/assets/f8a46206-980b-45e1-af2a-96826fddb51d" />

Experience and education show the steadiest step pattern in these tables. The learning field is a rough self label, not the same thing as counting certificates.

## How this was started

Python from: [Python 3.12.13 downloads](https://www.python.org/downloads/release/python-31213/).

The dataset is the Stack Overflow annual developer survey on Kaggle: [dataset on Kaggle](https://www.kaggle.com/datasets/edoardogalli/stack-overflow-annual-developer-survey-2025?resource=download). After downloading, the files were placed under `data/`, including `survey_results_public.csv` and anything else that came with the archive.

From this folder, an environment was created:

```bash
python3 -m venv ds-env
```

Dependencies were installed with the environment activated:

```bash
source ds-env/bin/activate
pip install jupyter pandas matplotlib
jupyter notebook
```

> Jupyter opened in the browser on: [http://localhost:8888/tree](http://localhost:8888/tree).

## Jupyter kernel

**In the Jupyter UI:**
new, then a Python notebook. The kernel was pointed at the `ds-env` environment when the default interpreter was not the venv (shown as "Python 3 (ds-env)" after `ipykernel` is available). The notebook file was renamed.

## Verify the setup

This was run in a cell to confirm pandas could read the public results file:

```python
import pandas as pd

df = pd.read_csv("data/survey_results_public.csv")
df.head()
```

<img width="1800" height="1042" alt="2026-04-18_10-56-34 (1)" src="https://github.com/user-attachments/assets/55a26567-0378-4c24-ada7-cdbdd0a8d6b7" />
