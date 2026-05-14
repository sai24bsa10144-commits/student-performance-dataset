# Exploratory Data Analysis — Student Performance Dataset

**Author:** Sai Sovan Pattanayak  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Overview

An end-to-end EDA on a 10,000-row student performance dataset to identify the key drivers of academic outcomes.

**Key Findings:**
- Students with 90%+ attendance score **91.8 on average**, vs 82.1 for students with <60% attendance — a **9.7-point gap**
- Students studying 8+ hours/week score **95.7 on average**, vs 79.3 for <2 hours/week
- Attendance + study hours together explain **35% of score variance** (R² = 0.35)
- Completing a test prep course adds ~5 points on average
- Students with Master's-educated parents outperform no-diploma households by ~4.5 points

---

## Dataset

| Feature | Type | Description |
|---|---|---|
| student_id | string | Unique identifier |
| gender | categorical | Male / Female |
| age | int | Student age (15–19) |
| parental_education | categorical | Highest education level |
| lunch_type | categorical | Standard / Free/Reduced |
| test_prep_course | categorical | None / Completed |
| attendance_rate | float | % attendance (40–100) |
| weekly_study_hours | float | Hours studied per week |
| math_score | float | Math score (0–100) |
| reading_score | float | Reading score (0–100) |
| writing_score | float | Writing score (0–100) |

- **10,000 rows** | **11 columns** | ~4% missing values (imputed with median)

---

## Analysis Sections

1. **Data Loading & Overview** — shape, dtypes, null counts
2. **Missing Value Analysis** — visualised and imputed with median
3. **Descriptive Statistics** — mean, std, quartiles for all numeric features
4. **Score Distributions** — histograms with mean lines for all 3 subjects
5. **Correlation Analysis** — heatmap of all numeric features
6. **Attendance vs Score** — scatter with trend line + boxplot by attendance band
7. **Study Hours vs Score** — scatter with trend line + bar chart by study band
8. **Categorical Factor Impact** — gender, parental education, test prep, lunch type
9. **Combined Heatmap** — attendance band × study hours band → mean score
10. **Regression** — Linear regression: R² and feature coefficients
11. **Key Insights Summary** — comparative bar chart of top/bottom factors

---

## Charts Generated

| File | Description |
|---|---|
| charts/01_missing_values.png | Missing value % by column |
| charts/02_score_distributions.png | Score histograms with mean lines |
| charts/03_correlation_heatmap.png | Feature correlation matrix |
| charts/04_attendance_vs_score.png | Scatter + boxplot by attendance band |
| charts/05_study_hours_vs_score.png | Scatter + bar by study hours band |
| charts/06_categorical_factors.png | 4-panel categorical factor comparison |
| charts/07_attendance_study_heatmap.png | Combined attendance × study heatmap |
| charts/08_key_insights_summary.png | Summary insight comparison bar chart |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Generate the dataset
python3 generate_dataset.py

# 3. Run the full EDA
python3 eda_analysis.py
```

Output: 8 charts saved to `charts/`

---

## Tech Stack

| Library | Version | Use |
|---|---|---|
| Pandas | 2.x | Data loading, cleaning, groupby |
| NumPy | 1.x | Array operations, polyfit |
| Matplotlib | 3.x | Custom visualisations |
| Seaborn | 0.x | Heatmaps, styled plots |
| Scikit-learn | 1.x | Linear regression, R² |
