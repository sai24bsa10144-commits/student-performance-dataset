import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

gender = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
age = np.random.randint(15, 20, n)
parental_education = np.random.choice(
    ['No Diploma', 'High School', 'Some College', "Bachelor's", "Master's"],
    n, p=[0.10, 0.25, 0.25, 0.28, 0.12]
)
lunch = np.random.choice(['Standard', 'Free/Reduced'], n, p=[0.65, 0.35])
test_prep = np.random.choice(['None', 'Completed'], n, p=[0.64, 0.36])

attendance = np.clip(np.random.normal(80, 12, n), 40, 100).round(1)
study_hours = np.clip(np.random.normal(5, 2.5, n), 0, 14).round(1)

base_score = (
    45
    + attendance * 0.30
    + study_hours * 2.5
    + (parental_education == "Master's") * 6
    + (parental_education == "Bachelor's") * 4
    + (parental_education == "Some College") * 2
    + (lunch == 'Standard') * 4
    + (test_prep == 'Completed') * 5
    + np.random.normal(0, 8, n)
)

math_score    = np.clip(base_score + np.random.normal(0, 5, n), 0, 100).round(1)
reading_score = np.clip(base_score + np.random.normal(2, 4, n), 0, 100).round(1)
writing_score = np.clip(base_score + np.random.normal(1, 4, n), 0, 100).round(1)

# Inject ~4% missing values
for arr in [math_score, reading_score, writing_score, attendance]:
    idx = np.random.choice(n, int(n * 0.04), replace=False)
    arr[idx] = np.nan

df = pd.DataFrame({
    'student_id':          [f'STU{str(i).zfill(5)}' for i in range(1, n+1)],
    'gender':              gender,
    'age':                 age,
    'parental_education':  parental_education,
    'lunch_type':          lunch,
    'test_prep_course':    test_prep,
    'attendance_rate':     attendance,
    'weekly_study_hours':  study_hours,
    'math_score':          math_score,
    'reading_score':       reading_score,
    'writing_score':       writing_score,
})

df.to_csv('student_performance.csv', index=False)
print(f"Dataset saved: {len(df)} rows, {df.isnull().sum().sum()} missing values")
print(df.head(3).to_string())
