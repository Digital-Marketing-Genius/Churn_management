# Customer Subscription Churn Prediction
### DATA 5000 — Introduction to Data Science
**Sprott School of Business, Carleton University**

> Predicting customer churn using interpretable machine learning: a systematic comparison of sampling strategies and classification models.

---

## Team

| Name | Role |
|---|---|
| Oluwatayo Alofun | Project Lead & Report Lead |
| Adeyeye Adedayo | Data Analyst & Visualization Lead |
| Kyle Bruinsma | Data Engineer & Modeling Support |
| Kumuditha Udugama | ML Engineer & Preprocessing Lead |

---

## Project Overview

Customer churn — the voluntary discontinuation of a subscription service — is one of the most costly challenges in subscription-based businesses. This project applies **logistic regression** and **random forest** classifiers to the IBM Telco Customer Churn dataset to predict whether a customer will churn, using only non-invasive account-level and behavioral variables.

A central methodological contribution is the **systematic comparison of four class-balancing strategies** to address the natural class imbalance in the dataset (26.54% churn rate):

- **Config 1:** No sampling (natural baseline)
- **Config 2:** SMOTE 50/50 oversampling
- **Config 3:** SMOTE 30/70 oversampling
- **Config 4:** Manual stratified undersampling *(selected final approach)*

Eight model-configuration combinations were evaluated. **Logistic regression with manual stratified undersampling** was selected as the final model based on the highest F1 score (61.13%) and recall (67.91%) across all combinations.

---

## Dataset

| Property | Detail |
|---|---|
| Source | IBM Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| File name | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Records | 7,043 customers |
| Variables | 21 (20 predictors + 1 target) |
| Target variable | `Churn` (Yes / No) |
| Natural churn rate | 26.54% |
| Missing values | 11 in TotalCharges (imputed with 0) |

**Key variables include:** tenure, contract type, monthly charges, internet service type, payment method, tech support, online security, and demographic information.

---

## Repository Structure

```
DATA5000-Churn-Prediction/
│
├── DATA5000_Churn_Prediction_Main.ipynb   # Main notebook (all code)
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (download from Kaggle)
│
├── outputs/
│   ├── charts/
│   │   ├── chart1_churn_distribution.png
│   │   ├── chart2_churn_by_contract.png
│   │   ├── chart3_churn_by_tenure.png
│   │   ├── chart4_monthly_charges.png
│   │   ├── chart5_churn_by_internet.png
│   │   ├── lr_confusion_matrices.png
│   │   ├── lr_roc_curves.png
│   │   ├── rf_confusion_matrices.png
│   │   ├── rf_roc_curves.png
│   │   ├── combined_roc_curves.png
│   │   ├── feature_importance.png
│   │   └── final_model_confusion_matrix.png
│   │
│   └── results/
│       └── performance_metrics.csv
│
├── report/
│   └── DATA5000_Churn_Report.docx
│
├── presentation/
│   └── DATA5000_Churn_Presentation.pptx
│
└── README.md
```

---

## Requirements

### Python Version
Python 3.11 or higher

### Required Libraries

Install all dependencies in one command:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn statsmodels
```

| Library | Version | Purpose |
|---|---|---|
| `pandas` | >= 2.0 | Data manipulation and encoding |
| `numpy` | >= 1.24 | Numerical operations and log transforms |
| `matplotlib` | >= 3.7 | Chart generation |
| `seaborn` | >= 0.12 | Statistical visualizations |
| `scikit-learn` | >= 1.3 | Modeling, metrics, and train/test splitting |
| `imbalanced-learn` | >= 0.11 | SMOTE oversampling |
| `statsmodels` | >= 0.14 | VIF multicollinearity analysis |

### Development Environment
This project was built and tested in **Google Colab** (free tier). No GPU is required. The dataset is small enough to run on any standard laptop or cloud CPU.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `DATA5000_Churn_Prediction_Main.ipynb`
3. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to your Google Drive
4. Update the `PROJECT_PATH` variable in Cell 4 to match your Drive folder path
5. Run all cells in order from top to bottom using **Runtime > Run All**

### Option 2: Local Machine

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn statsmodels
   ```
3. Place the dataset CSV in the `data/` folder
4. Open the notebook in Jupyter:
   ```bash
   jupyter notebook DATA5000_Churn_Prediction_Main.ipynb
   ```
5. Update `PROJECT_PATH` in Cell 4 to point to your local `data/` folder
6. Run all cells in order

---

## Notebook Structure

The notebook is organized into six clearly labelled sections:

| Section | Tasks Covered | Description |
|---|---|---|
| **1. Library Installation & Imports** | Setup | Install and import all required libraries; confirm versions |
| **2. Data Loading & Initial Review** | Task 3 | Load dataset from Google Drive; confirm shape and churn rate |
| **3. Exploratory Data Analysis** | Tasks 6-10 | Descriptive stats, visualizations, VIF analysis, EDA summary |
| **4. Data Preprocessing & Sampling** | Tasks 11-17 | Encoding, log transforms, train/test split, 4 sampling configs, prevalence inflation table |
| **5. Machine Learning Modeling** | Tasks 18-25 | Logistic regression and random forest across all configs; confusion matrices; ROC curves; feature importance |
| **6. Model Evaluation & Results** | Tasks 22-24 | Combined ROC curve; final model summary; business interpretation |

---

## Key Results

### Final Model: Logistic Regression — Config 4 (Manual Stratified Undersampling)

| Metric | Score |
|---|---|
| F1 Score | **61.13%** |
| Recall | **67.91%** |
| AUC-ROC | **83.73%** |
| Accuracy | 77.08% |
| Precision | 55.58% |

### Confusion Matrix (Test Set: 1,409 records)

|  | Predicted: No Churn | Predicted: Churned |
|---|---|---|
| **Actual: No Churn** | 832 (True Neg.) | 203 (False Pos.) |
| **Actual: Churned** | 120 (False Neg.) | 254 (True Pos.) |

Out of 374 actual churners in the test set, the model correctly identified **254 (67.91%)**.

### Sampling Strategy Comparison

| Configuration | Churn Rate | Inflation Factor | LR F1 | LR AUC |
|---|---|---|---|---|
| Config 1: No Sampling | 26.53% | 1.00x | 57.14% | 84.00% |
| Config 2: SMOTE 50/50 | 50.00% | 1.88x | 58.86% | 82.15% |
| Config 3: SMOTE 30/70 | 30.00% | 1.13x | 59.94% | 83.82% |
| **Config 4: Manual Undersampling** | **37.58%** | **1.42x** | **61.13%** | **83.73%** |

### Top Churn Predictors (Logistic Regression Coefficients)

**Increase churn risk:** Month-to-month contract, fiber optic internet, high monthly charges, electronic check payment, senior citizen status

**Reduce churn risk:** Annual / two-year contract, longer tenure, tech support subscription, online security add-on, automatic payment method

---

## Methodology Summary

```
Raw Dataset (7,043 records, 21 variables)
        │
        ▼
Exploratory Data Analysis
  ├── Descriptive statistics
  ├── 5 visualization charts
  ├── Missing value imputation (TotalCharges)
  └── VIF analysis → TotalCharges removed (VIF = 8.08)
        │
        ▼
Data Preprocessing
  ├── Churn encoded: Yes=1, No=0
  ├── Dropped: customerID, tenure_group
  ├── Log transformed: tenure, MonthlyCharges
  ├── One-hot encoded: all categorical variables
  └── Stratified split: 60% train / 20% val / 20% test
        │
        ▼
4 Sampling Configurations
  ├── Config 1: No sampling (baseline)
  ├── Config 2: SMOTE 50/50
  ├── Config 3: SMOTE 30/70
  └── Config 4: Manual stratified undersampling ← selected
        │
        ▼
Model Training (LR + RF × 4 configs = 8 combinations)
        │
        ▼
Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC
        │
        ▼
Final Model: LR + Config 4
  ├── Feature importance analysis
  └── Business recommendation mapping
```

---

## Reproducibility

All random operations in this project use a fixed seed of `random_state=42` to ensure fully reproducible results. This applies to:

- Train/validation/test splitting (`train_test_split`)
- SMOTE oversampling (`SMOTE`)
- Manual stratified undersampling (`DataFrame.sample`)
- Random forest training (`RandomForestClassifier`)
- Logistic regression training (`LogisticRegression`)

Running the notebook from top to bottom on the same dataset will produce identical results every time.

---

## References

1. Burez, J., & Van den Poel, D. (2009). Handling class imbalance in customer churn prediction. *Expert Systems with Applications, 36*(3), 4626-4636. https://doi.org/10.1016/j.eswa.2008.05.027

2. Huang, B., Buckley, B., & Kechadi, T. M. (2012). Multi-objective feature selection by using NSGA-II for customer churn prediction in telecommunications. *Expert Systems with Applications, 37*(5), 3638-3646. https://doi.org/10.1016/j.eswa.2009.10.027

3. IBM. (2023). *Telco customer churn* [Dataset]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

4. Vafeiadis, T., Diamantaras, K. I., Sarigiannidis, G., & Chatzisavvas, K. C. (2015). A comparison of machine learning techniques for customer churn prediction. *Simulation Modelling Practice and Theory, 55*, 1-9. https://doi.org/10.1016/j.simpat.2015.03.003

5. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research, 218*(1), 211-229. https://doi.org/10.1016/j.ejor.2011.09.031

---

## License

This project was completed for academic purposes as part of DATA 5000 at Carleton University. The IBM Telco Customer Churn dataset is publicly available under the terms of its Kaggle listing. All code in this repository is original work by the project team.
