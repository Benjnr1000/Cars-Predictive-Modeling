# 🚗 Employee Commute Mode Prediction

A classification pipeline in R to predict whether an employee uses a **Car** vs **Not Car** for commuting, based on their personal and professional attributes.

---

## 📋 Project Overview

- **Objective**: Build and compare models to predict an employee’s likelihood of commuting by **car**.  
- **Dataset**: `Cars-dataset.csv` (418 records, 9 features + target):
  - Numerical: Age, Work.Exp (years), Salary (₹ thousands), Distance (km)  
  - Categorical (factors): Gender, Engineer, MBA, license  
  - Target: `Transport` (2Wheeler, Car, Public Transport) → recoded to `Carusage` (Car / Not.Car)  

---

## 🗂️ Repository Structure
```bash
├── data/
│ └── Cars-dataset.csv # Raw input
├── scripts/
│ ├── 01_eda.R # EDA and visualization
│ ├── 02_data_prep.R # Cleaning, feature engineering, SMOTE
│ ├── 03_models.R # Model training and evaluation
│ └── 04_compare_models.R # Resampling comparison and summary
├── outputs/
│ ├── plots/ # EDA & model diagnostic plots
│ └── metrics.csv # Test-set metrics per model
└── README.md # This file
```
---

## 🛠️ Setup & Dependencies

1. **Install R (≥ 4.0)** and **RStudio**.  
2. **Install required packages**:
   ```r
   install.packages(c(
     "readr", "dplyr", "ggplot2", "gridExtra", "corrplot",
     "DataExplorer", "caret", "DMwR", "randomForest", "gbm",
     "xgboost", "e1071"
   ))
3. Place `Cars-dataset.csv` in the `data/` folder.

📈 Results Summary

| Model                   | Accuracy  | Sensitivity | Specificity |
| ----------------------- | --------- | ----------- | ----------- |
| K-Nearest Neighbors     | 0.9758    | 0.80        | 0.9912      |
| Naive Bayes             | 0.9758    | 0.90        | 0.9825      |
| Logistic Regression     | 0.9839    | 0.90        | 0.9912      |
| **Random Forest**       | **1.000** | **1.00**    | **1.00**    |
| Gradient Boosting (GBM) | 1.000     | 1.00        | 1.00        |
| XGBoost                 | 0.9919    | 0.90        | 1.00        |

> Best Performer: Random Forest (100% on all metrics)


🔍 Key Takeaways
- **Salary**, **Age**, **Work Experience**, **Distance**, and **License** status are the top predictors (via variable importance).
- SMOTE effectively balanced the minority class before modeling.
- Tree-based methods (Random Forest, GBM) outperformed linear models on this task.


👤 Author

Benedict Egwuchukwu

Data Science & Analytics Practitioner

📧 egwuchukwubenedict@yahoo.com




