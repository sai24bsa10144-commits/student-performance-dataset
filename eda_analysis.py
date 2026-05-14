"""
Exploratory Data Analysis — Student Performance Dataset
Author: Sai Sovan Pattanayak
Tools : Python, Pandas, NumPy, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)

# ── colour palette ──────────────────────────────────────────
BLUE   = '#1F497D'
TEAL   = '#2E86AB'
CORAL  = '#E84855'
AMBER  = '#F4A261'
GREEN  = '#2D936C'
LIGHT  = '#F0F4F8'
PALETTE = [BLUE, TEAL, GREEN, AMBER, CORAL]

sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': LIGHT,
                     'font.size': 11})

print("=" * 60)
print("  STUDENT PERFORMANCE — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD & OVERVIEW
# ══════════════════════════════════════════════════════════════
print("\n[1] Loading data…")
df = pd.read_csv('student_performance.csv')
print(f"    Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")
print("\n── First 5 rows ──")
print(df.head().to_string())
print("\n── Data Types & Non-Null Counts ──")
print(df.info())

# ══════════════════════════════════════════════════════════════
# SECTION 2 — MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n[2] Missing Value Analysis")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
miss_df = miss_df[miss_df['Missing Count'] > 0]
print(miss_df.to_string())

# Visualise missing values
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(miss_df.index, miss_df['Missing %'], color=CORAL, edgecolor='white', height=0.5)
for bar, val in zip(bars, miss_df['Missing %']):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', fontsize=10, color='#333')
ax.set_xlabel('Missing Values (%)')
ax.set_title('Missing Value Distribution by Column', fontsize=13, fontweight='bold', color=BLUE)
ax.set_xlim(0, 6)
plt.tight_layout()
plt.savefig('charts/01_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/01_missing_values.png")

# Fill missing with median (numerical) — safe for skewed data
for col in ['math_score', 'reading_score', 'writing_score', 'attendance_rate']:
    df[col].fillna(df[col].median(), inplace=True)
print(f"    Missing after imputation: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n[3] Descriptive Statistics (Numeric Columns)")
num_cols = ['attendance_rate', 'weekly_study_hours', 'math_score', 'reading_score', 'writing_score']
stats = df[num_cols].describe().round(2)
print(stats.to_string())

df['avg_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1).round(2)

# ══════════════════════════════════════════════════════════════
# SECTION 4 — SCORE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════
print("\n[4] Score Distributions")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
score_cols = ['math_score', 'reading_score', 'writing_score']
colors = [BLUE, TEAL, GREEN]
titles = ['Math Score', 'Reading Score', 'Writing Score']

for ax, col, color, title in zip(axes, score_cols, colors, titles):
    ax.hist(df[col].dropna(), bins=30, color=color, edgecolor='white', alpha=0.85)
    mean_val = df[col].mean()
    ax.axvline(mean_val, color=CORAL, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.set_title(title, fontsize=12, fontweight='bold', color=BLUE)
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)

fig.suptitle('Distribution of Scores Across Subjects', fontsize=14, fontweight='bold', color=BLUE, y=1.02)
plt.tight_layout()
plt.savefig('charts/02_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/02_score_distributions.png")
for col in score_cols:
    print(f"    {col:<20} mean={df[col].mean():.2f}  median={df[col].median():.2f}  std={df[col].std():.2f}")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
print("\n[5] Correlation Analysis")
corr = df[num_cols + ['avg_score']].corr().round(2)
print(corr.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
            ax=ax, linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 10})
ax.set_title('Correlation Matrix — Student Performance Features',
             fontsize=13, fontweight='bold', color=BLUE, pad=15)
plt.tight_layout()
plt.savefig('charts/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/03_correlation_heatmap.png")

# Key correlations with avg_score
for col in ['attendance_rate', 'weekly_study_hours']:
    r = df[[col, 'avg_score']].corr().iloc[0, 1]
    print(f"    Correlation  {col} ↔ avg_score : {r:.3f}")

# ══════════════════════════════════════════════════════════════
# SECTION 6 — ATTENDANCE vs SCORE
# ══════════════════════════════════════════════════════════════
print("\n[6] Attendance Rate vs Average Score")
df['attendance_band'] = pd.cut(df['attendance_rate'],
                                bins=[0, 60, 70, 80, 90, 100],
                                labels=['<60%', '60–70%', '70–80%', '80–90%', '90–100%'])
band_stats = df.groupby('attendance_band', observed=True)['avg_score'].agg(['mean','std','count']).round(2)
print(band_stats.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter with trend
att_clean = df[['attendance_rate', 'avg_score']].dropna()
sample = att_clean.sample(1500, random_state=42)
ax1.scatter(sample['attendance_rate'], sample['avg_score'],
            alpha=0.25, color=TEAL, s=20)
m, b = np.polyfit(att_clean['attendance_rate'], att_clean['avg_score'], 1)
x_line = np.linspace(40, 100, 100)
ax1.plot(x_line, m * x_line + b, color=CORAL, linewidth=2.5, label=f'Trend  y={m:.2f}x+{b:.1f}')
ax1.set_xlabel('Attendance Rate (%)')
ax1.set_ylabel('Average Score')
ax1.set_title('Attendance Rate vs Average Score', fontweight='bold', color=BLUE)
ax1.legend()

# Box plot by band
colors_box = [CORAL, AMBER, TEAL, GREEN, BLUE]
groups = [df[df['attendance_band'] == b]['avg_score'].dropna() for b in df['attendance_band'].cat.categories]
bp = ax2.boxplot(groups, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax2.set_xticklabels(df['attendance_band'].cat.categories, rotation=15)
ax2.set_xlabel('Attendance Band')
ax2.set_ylabel('Average Score')
ax2.set_title('Score Distribution by Attendance Band', fontweight='bold', color=BLUE)

plt.tight_layout()
plt.savefig('charts/04_attendance_vs_score.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/04_attendance_vs_score.png")

# ══════════════════════════════════════════════════════════════
# SECTION 7 — STUDY HOURS vs SCORE
# ══════════════════════════════════════════════════════════════
print("\n[7] Weekly Study Hours vs Average Score")
df['study_band'] = pd.cut(df['weekly_study_hours'],
                           bins=[0, 2, 4, 6, 8, 14],
                           labels=['0–2 hrs', '2–4 hrs', '4–6 hrs', '6–8 hrs', '8+ hrs'])

study_stats = df.groupby('study_band', observed=True)['avg_score'].mean().round(2)
s2 = df[['weekly_study_hours','avg_score']].dropna().sample(1500, random_state=1)
ax1.scatter(s2['weekly_study_hours'], s2['avg_score'], alpha=0.25, color=GREEN, s=20)
m2, b2 = np.polyfit(df['weekly_study_hours'], df['avg_score'], 1)
x2 = np.linspace(0, 14, 100)
ax1.plot(x2, m2*x2+b2, color=CORAL, linewidth=2.5, label=f'Trend  y={m2:.2f}x+{b2:.1f}')
ax1.set_xlabel('Weekly Study Hours')
ax1.set_ylabel('Average Score')
ax1.set_title('Study Hours vs Average Score', fontweight='bold', color=BLUE)
ax1.legend()

bars = ax2.bar(study_stats.index, study_stats.values,
               color=[CORAL, AMBER, TEAL, GREEN, BLUE], edgecolor='white', width=0.6)
for bar, val in zip(bars, study_stats.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel('Study Hours Band')
ax2.set_ylabel('Mean Average Score')
ax2.set_title('Mean Score by Study Hours Band', fontweight='bold', color=BLUE)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('charts/05_study_hours_vs_score.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/05_study_hours_vs_score.png")

# ══════════════════════════════════════════════════════════════
# SECTION 8 — CATEGORICAL FACTORS
# ══════════════════════════════════════════════════════════════
print("\n[8] Impact of Categorical Factors on Average Score")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a — Gender
gender_means = df.groupby('gender')['avg_score'].mean().round(2)
print(f"    Gender means:\n{gender_means.to_string()}")
axes[0,0].bar(gender_means.index, gender_means.values, color=[TEAL, CORAL],
              edgecolor='white', width=0.5)
for i, (idx, val) in enumerate(gender_means.items()):
    axes[0,0].text(i, val+0.5, f'{val:.1f}', ha='center', fontweight='bold')
axes[0,0].set_title('Mean Score by Gender', fontweight='bold', color=BLUE)
axes[0,0].set_ylabel('Mean Average Score')
axes[0,0].set_ylim(0, 100)

# 8b — Parental Education
edu_order = ['No Diploma', 'High School', 'Some College', "Bachelor's", "Master's"]
edu_means = df.groupby('parental_education')['avg_score'].mean().reindex(edu_order).round(2)
print(f"\n    Parental Education means:\n{edu_means.to_string()}")
axes[0,1].bar(range(len(edu_means)), edu_means.values,
              color=PALETTE, edgecolor='white', width=0.6)
axes[0,1].set_xticks(range(len(edu_means)))
axes[0,1].set_xticklabels(edu_order, rotation=20, ha='right', fontsize=9)
for i, val in enumerate(edu_means.values):
    axes[0,1].text(i, val+0.5, f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
axes[0,1].set_title('Mean Score by Parental Education', fontweight='bold', color=BLUE)
axes[0,1].set_ylabel('Mean Average Score')
axes[0,1].set_ylim(0, 100)

# 8c — Test Prep
prep_means = df.groupby('test_prep_course')['avg_score'].mean().round(2)
prep_counts = df.groupby('test_prep_course')['avg_score'].count()
print(f"\n    Test Prep means:\n{prep_means.to_string()}")
axes[1,0].bar(prep_means.index, prep_means.values, color=[TEAL, GREEN],
              edgecolor='white', width=0.5)
for i, (idx, val) in enumerate(prep_means.items()):
    axes[1,0].text(i, val+0.5, f'{val:.1f}', ha='center', fontweight='bold')
axes[1,0].set_title('Mean Score by Test Prep Course', fontweight='bold', color=BLUE)
axes[1,0].set_ylabel('Mean Average Score')
axes[1,0].set_ylim(0, 100)

# 8d — Lunch Type
lunch_means = df.groupby('lunch_type')['avg_score'].mean().round(2)
print(f"\n    Lunch Type means:\n{lunch_means.to_string()}")
axes[1,1].bar(lunch_means.index, lunch_means.values, color=[BLUE, AMBER],
              edgecolor='white', width=0.5)
for i, (idx, val) in enumerate(lunch_means.items()):
    axes[1,1].text(i, val+0.5, f'{val:.1f}', ha='center', fontweight='bold')
axes[1,1].set_title('Mean Score by Lunch Type', fontweight='bold', color=BLUE)
axes[1,1].set_ylabel('Mean Average Score')
axes[1,1].set_ylim(0, 100)

fig.suptitle('Impact of Categorical Factors on Student Performance',
             fontsize=14, fontweight='bold', color=BLUE)
plt.tight_layout()
plt.savefig('charts/06_categorical_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/06_categorical_factors.png")

# ══════════════════════════════════════════════════════════════
# SECTION 9 — COMBINED EFFECT (ATTENDANCE + STUDY HOURS)
# ══════════════════════════════════════════════════════════════
print("\n[9] Combined Effect — Attendance × Study Hours on Score")

pivot = df.pivot_table(
    values='avg_score',
    index='attendance_band',
    columns='study_band',
    aggfunc='mean',
    observed=True
).round(1)
print(pivot.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues',
            ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'fontweight': 'bold'})
ax.set_title('Average Score by Attendance Band × Study Hours Band',
             fontsize=13, fontweight='bold', color=BLUE, pad=15)
ax.set_xlabel('Weekly Study Hours Band', fontsize=11)
ax.set_ylabel('Attendance Rate Band', fontsize=11)
plt.tight_layout()
plt.savefig('charts/07_attendance_study_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Chart saved: charts/07_attendance_study_heatmap.png")

# ══════════════════════════════════════════════════════════════
# SECTION 10 — VARIANCE EXPLAINED
# ══════════════════════════════════════════════════════════════
print("\n[10] Variance Explained by Attendance + Study Hours")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

clean_lr = df[['attendance_rate','weekly_study_hours','avg_score']].dropna()
features = clean_lr[['attendance_rate', 'weekly_study_hours']]
target = clean_lr['avg_score']

model = LinearRegression()
model.fit(features, target)
r2 = model.score(features, target)
print(f"    R² (attendance + study hours) = {r2:.4f}  ({r2*100:.1f}% of variance explained)")
print(f"    Coefficients: attendance={model.coef_[0]:.3f}, study_hours={model.coef_[1]:.3f}")
print(f"    Intercept: {model.intercept_:.2f}")

# ══════════════════════════════════════════════════════════════
# SECTION 11 — KEY INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)

high_att = df[df['attendance_rate'] >= 90]['avg_score'].mean()
low_att  = df[df['attendance_rate'] < 60]['avg_score'].mean()
high_study = df[df['weekly_study_hours'] >= 8]['avg_score'].mean()
low_study  = df[df['weekly_study_hours'] < 2]['avg_score'].mean()
prep_clean = df[df['test_prep_course'].notna()]
prep_means2 = prep_clean.groupby('test_prep_course')['avg_score'].mean().round(2)
prep_diff = prep_means2.get('Completed', 0) - prep_means2.get('None', 0)

print(f"  1. Attendance impact   : Students with 90%+ attendance score {high_att:.1f} on avg,")
print(f"                           vs {low_att:.1f} for <60% attendance — a {high_att-low_att:.1f}-point gap.")
print(f"  2. Study hours impact  : 8+ hrs/week → avg {high_study:.1f}, vs {low_study:.1f} for <2 hrs/week.")
print(f"  3. Combined variance   : Attendance + study hours explain {r2*100:.1f}% of score variance.")
print(f"  4. Test prep benefit   : Completing test prep course adds {prep_diff:.1f} avg points.")
masters_avg = edu_means["Master's"]
no_diploma_avg = edu_means['No Diploma']
print(f"  5. Parental education  : Master's-educated parents -> avg {masters_avg:.1f},")
print(f"                           vs {no_diploma_avg:.1f} for no diploma.")
print(f"  6. Lunch type gap      : Standard lunch students score {lunch_means['Standard'] - lunch_means['Free/Reduced']:.1f} pts higher on avg.")

# Final combined chart — insight summary visual
fig, ax = plt.subplots(figsize=(10, 5))
factors = ['90%+ Attendance', '<60% Attendance', '8+ Study Hrs', '<2 Study Hrs',
           'Test Prep Done', 'No Test Prep', "Master's Parent", 'No Diploma Parent']
scores  = [high_att, low_att, high_study, low_study,
           prep_means2.get('Completed',0), prep_means2.get('None',0),
           edu_means["Master's"], edu_means['No Diploma']]
bar_colors = [GREEN, CORAL, GREEN, CORAL, GREEN, CORAL, GREEN, CORAL]

bars = ax.barh(factors[::-1], scores[::-1], color=bar_colors[::-1],
               edgecolor='white', height=0.6)
for bar, val in zip(bars, scores[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Mean Average Score')
ax.set_title('Key Factors — Mean Average Score Comparison',
             fontsize=13, fontweight='bold', color=BLUE)
ax.set_xlim(0, 105)
ax.axvline(df['avg_score'].mean(), color=AMBER, linestyle='--', linewidth=2,
           label=f'Overall mean: {df["avg_score"].mean():.1f}')
ax.legend()
plt.tight_layout()
plt.savefig('charts/08_key_insights_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    → Chart saved: charts/08_key_insights_summary.png")
print("\n  All 8 charts saved to charts/")
print("  EDA complete.")
