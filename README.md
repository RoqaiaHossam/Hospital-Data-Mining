# 🏥 Hospital Healthcare Data Mining & Analytics System

> An end-to-end data mining pipeline applied to 55,500 hospital patient records — covering exploratory analysis, clustering, fuzzy logic, genetic optimization, and an integrated patient assessment system.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tasks & Methodology](#tasks--methodology)
  - [Task 1 — Exploratory Data Analysis](#task-1--exploratory-data-analysis)
  - [Task 2 — Data Preprocessing](#task-2--data-preprocessing)
  - [Task 3 — K-Medoids Clustering](#task-3--k-medoids-clustering)
  - [Task 4 — Hierarchical Clustering](#task-4--hierarchical-clustering)
  - [Task 5 — Fuzzy Logic System](#task-5--fuzzy-logic-system)
  - [Task 6 — Genetic Algorithm Optimization](#task-6--genetic-algorithm-optimization)
  - [Task 7 — Integrated System Implementation](#task-7--integrated-system-implementation)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results Summary](#results-summary)

---

## Project Overview

This project applies a comprehensive data mining pipeline to a real-world hospital dataset to extract clinically and financially meaningful insights. The pipeline progresses from raw data exploration through unsupervised clustering, intelligent rule-based reasoning, and evolutionary optimization — culminating in an integrated system that produces a full patient risk and financial support report from a single patient record.

**Core goals:**
- Identify natural patient groupings based on medical and administrative features
- Build an automated financial support recommendation engine
- Optimize feature selection using a genetic algorithm
- Deliver a unified, production-style analytics system

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) |
| Records | 55,500 patient admissions |
| Features | 15 (demographic, medical, administrative, financial) |
| Target domains | EDA, clustering, billing prediction, resource utilization |

### Features

| Feature | Type | Description |
|---|---|---|
| Name | Categorical | Patient full name |
| Age | Numerical | Patient age in years |
| Gender | Categorical | Male / Female |
| Blood Type | Categorical | Patient blood group |
| Medical Condition | Categorical | Primary diagnosis |
| Date of Admission | Date | Hospital admission date |
| Discharge Date | Date | Hospital discharge date |
| Doctor | Categorical | Attending physician |
| Hospital | Categorical | Treating hospital |
| Insurance Provider | Categorical | Covering insurance company |
| Billing Amount | Numerical | Total medical bill ($) |
| Room Number | Numerical | Assigned room |
| Admission Type | Categorical | Emergency / Urgent / Elective |
| Medication | Categorical | Prescribed medication |
| Test Results | Categorical | Medical test outcome |

---

## Project Structure

```
Hospital-Data-Mining/
│
├── Hospital-Data-Mining_prjct_Final.ipynb   # Main notebook (all 7 tasks)
├── healthcare_dataset.csv                   # Raw dataset
├── pipeline_flowchart.png                   # System pipeline diagram
└── README.md
```

---

## Tasks & Methodology

### Task 1 — Exploratory Data Analysis

A thorough visual and statistical exploration of the dataset to understand distributions, relationships, and potential issues before modeling.

**Visualizations produced:**
- Age distribution (histogram + KDE)
- Gender distribution (pie chart)
- Medical condition frequency (bar plot)
- Admission type distribution
- Billing amount distribution (histogram + KDE)
- Billing amount by admission type (boxplot)
- Insurance provider frequency
- Length of stay distribution
- Monthly admissions trend

**Key observations:**
- The dataset is uniformly distributed across medical conditions, admission types, and insurance providers — no single category dominates
- Age spans the full adult range with no dominant group
- Billing amounts follow an approximately normal distribution
- **Length of Stay** (engineered from admission/discharge dates) emerged as the most clinically significant feature

---

### Task 2 — Data Preprocessing

A complete preprocessing pipeline to prepare the data for machine learning.

| Step | Method | Justification |
|---|---|---|
| Missing values | `SimpleImputer` (median for numeric, most-frequent for categorical) | Robust to outliers; no data loss |
| Negative billing | `.clip(lower=0)` | Billing cannot be negative |
| Duplicates | `drop_duplicates()` | Ensure data integrity |
| Column names | Lowercase, underscores, strip special chars | Consistency for downstream processing |
| Outlier detection | IQR method | Standard, interpretable outlier detection |
| Categorical encoding | One-Hot Encoding (drop first) | Avoid artificial ordinality in nominal data |
| Feature scaling | `StandardScaler` | Required for distance-based algorithms (KNN, K-Medoids) |

---

### Task 3 — K-Medoids Clustering

K-Medoids was chosen over K-Means for its robustness to outliers — critical in healthcare data where extreme billing amounts could distort centroids.

**Optimal K selection:**

| Method | Result |
|---|---|
| Elbow Method | K = 3 (clearest inertia inflection) |
| Silhouette Score | 0.56 at K=2, **0.46 at K=3** |

K=3 was selected: the silhouette difference is marginal, and K=3 provides richer clinical interpretation.

**Cluster interpretation:**

| Cluster | Severity | Avg. Stay | Description |
|---|---|---|---|
| Cluster 1 | **Mild** | ~6 days | Short stay, routine recovery |
| Cluster 0 | **Moderate** | ~16 days | Standard treatment and monitoring |
| Cluster 2 | **Severe** | ~26 days | Extended care, serious condition |

Clusters were visualized using both **PCA** and **t-SNE** dimensionality reduction.

---

### Task 4 — Hierarchical Clustering

Three linkage methods were evaluated — Ward, Complete, and Average — with Single linkage excluded due to the chaining effect producing one dominant cluster.

**Final model:** Complete linkage, K=3

| Linkage | Silhouette Score (K=3) |
|---|---|
| Complete | **0.4489** |
| Ward | 0.4286 |
| Average | 0.4023 |

**Cluster interpretation:**

| Cluster | Severity | Avg. Stay |
|---|---|---|
| HC_Cluster 2 | **Mild** | ~4 days |
| HC_Cluster 0 | **Moderate** | ~14 days |
| HC_Cluster 1 | **Severe** | ~25 days |

The Hierarchical and K-Medoids results independently converged on the same three severity groups — validating that the patient stratification pattern reflects a genuine signal in the data, not an artifact of any single algorithm.

---

### Task 5 — Fuzzy Logic System

A **Fuzzy Inference System** was designed to recommend financial support percentages based on patient severity and billing amount — replacing blunt binary decisions with nuanced, continuous outputs.

**Inputs:**
- `severity` — patient cluster (Mild / Moderate / Severe)
- `billing` — total bill amount ($0–$55,000)

**Output:**
- `support` — financial aid percentage (0–100%)

**Membership functions:** Triangular (trimf) for all variables

**Rule base (9 rules):**

| Severity \ Billing | Low | Medium | High |
|---|---|---|---|
| Mild | None | None | Partial |
| Moderate | None | Partial | Partial |
| Severe | Partial | Full | Full |

**Defuzzification:** Centroid method

**Validation results:**

| Case | Cluster | Billing | Expected | Actual |
|---|---|---|---|---|
| Severe + High bill | Severe | ~$44,919 | Full support | ~88–90% |
| Severe + Medium bill | Severe | ~$28,814 | Full support | ~88–90% |
| Moderate + Medium bill | Moderate | ~$32,353 | Partial | ~12% |
| Moderate + Low bill | Moderate | ~$10,199 | Minimal | ~11% |

---

### Task 6 — Genetic Algorithm Optimization

A Genetic Algorithm was applied to the **feature selection problem** — finding the optimal subset of features that maximizes the K-Medoids Silhouette Score, reducing noise from redundant one-hot encoded columns.

**GA Configuration:**

| Parameter | Value |
|---|---|
| Population size | 20 chromosomes |
| Generations | 15 |
| Crossover rate | 0.80 (single-point) |
| Mutation rate | 0.05 (bit-flip) |
| Selection | Tournament |
| Fitness function | Silhouette Score (K-Medoids, k=3) |
| Elitism | Top chromosome preserved each generation |

**Chromosome encoding:** Binary vector — `1` = feature included, `0` = excluded

**Results:**

| Method | Silhouette Score |
|---|---|
| Baseline (all features) | 0.46 |
| Initial population best | varies |
| GA optimized | improved score with fewer features |

The GA confirmed that length-of-stay-related features carry the majority of discriminative signal, and many one-hot encoded columns are redundant for clustering.

---

### Task 7 — Integrated System Implementation

A production-style `healthcare_analytics_system()` pipeline that accepts a raw patient record dictionary and returns a complete clinical and financial report.

**Pipeline steps:**

```
patient_record (dict)
        │
        ▼
[Step 1] Processing Patient Data
  Clean, encode, scale → prepared_patient + computed_stay
        │
        ▼
[Step 2] Patient Classification
  K-Medoids + Hierarchical → ensemble_cluster (0 / 1 / 2)
        │
        ▼
[Step 3] Financial Support Calculation
  fuzzy.calculate_support() → support_percent + support_level
        │
        ▼
[Step 4] Feature Optimisation Results
  genetic_results → optimal_features + quality_boost
        │
        ▼
[Step 5] Overall Risk Assessment
  severity + billing + computed_stay → risk_total / 100
        │
        ▼
[Step 6] Final Report
  Assemble all outputs → return final_results
        │
        ▼
final_results (dict)
```

**Risk score components (equal thirds):**
- Severity component: up to 33.33 points
- Billing component: up to 33.33 points
- Length of stay component: up to 33.33 points

**Risk categories:**

| Score | Category | Action |
|---|---|---|
| < 33 | Low Risk | Standard discharge, routine follow-up |
| 33–67 | Moderate Risk | Close monitoring, consider social work consultation |
| > 67 | High Risk | Immediate care coordination, financial counselling |

---

## Key Findings

1. **Length of stay is the primary differentiating signal.** All other features (age, billing, medical condition, admission type, test results) were uniformly distributed across clusters. This is clinically meaningful — a patient staying 26 days represents a fundamentally different case than one staying 6 days.

2. **Both clustering algorithms independently confirmed the same three severity groups** (mild / moderate / severe), validating the robustness of the patient stratification.

3. **The fuzzy system successfully handles boundary cases**, providing proportional support percentages rather than abrupt binary decisions — particularly valuable for patients near billing category thresholds.

4. **The GA identified a reduced, higher-quality feature subset**, demonstrating that many one-hot encoded columns are redundant for clustering and that dimensionality can be reduced without sacrificing cluster quality.

5. **The uniform dataset distribution is itself a finding** — it suggests the data reflects real-world diversity and that resource demand across conditions and admission types is broadly balanced.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, clustering (Agglomerative), PCA, t-SNE, Silhouette Score |
| `scikit-learn-extra` | K-Medoids clustering |
| `scikit-fuzzy` | Fuzzy Logic inference system |
| `scipy` | Hierarchical linkage / dendrogram |

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/RoqaiaHossam/Price-Prediction-for-players-cars.git
cd Price-Prediction-for-players-cars
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-learn-extra scikit-fuzzy scipy
```

**3. Add the dataset**

Download `healthcare_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) and place it in the project root (or update the path in the notebook).

**4. Run the notebook**
```bash
jupyter notebook Hospital-Data-Mining_prjct_Final.ipynb
```

> The notebook is designed to run top-to-bottom. Each task builds on the output of the previous one.

---

## Usage

To run the integrated system on a new patient record:

```python
sample_patient = {
    "name": "Sarah Johnson",
    "age": 72,
    "gender": "Female",
    "medical_condition": "Diabetes Type 2",
    "date_of_admission": "2024-01-10",
    "discharge_date": "2024-02-05",
    "billing_amount": 38500,
    "admission_type": "Emergency",
    "medication": "Metformin",
    "test_results": "Borderline",
}

result = healthcare_analytics_system(
    kmedoids_model=model,
    hierarchical_model=hc_model,
    fuzzy_system=FuzzyWrapper(),
    genetic_results=genetic_results,
    preprocessing_pipeline=preprocessing_pipeline,
    patient_record=sample_patient,
)
```

The function returns a dictionary with clinical severity, financial support percentage, risk score, and recommended care plan.

---

## Results Summary

| Component | Outcome |
|---|---|
| Optimal clusters | K = 3 (Mild / Moderate / Severe) |
| K-Medoids Silhouette | 0.46 |
| HC Silhouette (Complete) | 0.449 |
| Fuzzy support range | 11% (Minimal) → 90% (Full) |
| GA feature reduction | Reduced redundant features while maintaining/improving cluster quality |
| Primary differentiator | Length of hospital stay |
