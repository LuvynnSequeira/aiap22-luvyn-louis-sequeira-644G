# Phishing Website Detection - AIAP Batch 22

**Name:** Luvyn Louis Sequeira  
**Email:** [your-aiap-email@domain.com]  
**GitHub:** LuvynnSequeira

---

## Overview

This project builds a machine learning pipeline to detect phishing websites. It includes exploratory data analysis and an end-to-end ML pipeline that trains multiple classifiers to identify phishing sites based on website features.

## Project Structure

```
├── data/
│   └── phishing.db          # Download separately (not in repo)
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── run_pipeline.py
├── eda.ipynb                # Exploratory data analysis
├── requirements.txt
├── run.sh
└── README.md
```

## Dataset

**Source:** 10,500 website samples from phishing.db  
**Features:** 16 total (12 numerical, 2 categorical, 1 target)

### Key Features:
- LineOfCode, LargestLineLength
- NoOfURLRedirect, NoOfSelfRedirect, NoOfPopup, NoOfiFrame
- NoOfImage (number of images found in website)
- NoOfSelfRef, NoOfExternalRef
- Robots (has robots.txt), IsResponsive
- DomainAgeMonths
- Industry, HostingProvider (categorical)
- **label**: 0 = Legitimate, 1 = Phishing

**Class Distribution:** ~30% legitimate, ~70% phishing (imbalanced)

**Data Issues:**
- ~2,355 missing values in LineOfCode
- High cardinality in Industry and HostingProvider
- Outliers in several numerical features

---

## Setup

### Requirements
- Python 3.8+
- SQLite database file (download from assessment)

### Installation

1. Clone the repo
2. Download `phishing.db` and place in `data/` folder
3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run Complete Pipeline
```bash
bash run.sh
```

This will:
1. Load and preprocess data
2. Train 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
3. Perform hyperparameter tuning
4. Evaluate and save best model
5. Generate visualizations and reports

### Options
```bash
# Faster execution (skip tuning)
python src/run_pipeline.py --no-tune

# Don't save models
python src/run_pipeline.py --no-save
```

### View EDA
```bash
jupyter notebook eda.ipynb
```

---

## Approach

### 1. Data Preprocessing
- **Missing values:** Median imputation
- **Categorical encoding:** Frequency + label encoding (handles high cardinality)
- **Scaling:** StandardScaler
- **Split:** 80-20 stratified (maintains class balance)

### 2. Models Trained
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting

All models use `class_weight='balanced'` to handle imbalanced data.

### 3. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation on:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Gradient Boosting: n_estimators, learning_rate, max_depth

### 4. Evaluation Metrics
- **Primary:** F1-score (balances precision/recall for imbalanced data)
- **Secondary:** Accuracy, Precision, Recall, ROC-AUC
- **Visualizations:** Confusion matrix, ROC curve, feature importance

---

## Key Findings from EDA

### What Distinguishes Phishing Sites:
1. **Younger domains** (lower DomainAgeMonths)
2. **More URL redirects**
3. **Less likely to have robots.txt**
4. **Less responsive design**
5. **More pop-ups**

### Preprocessing Decisions:
- **Median imputation** over mean (robust to outliers)
- **Frequency encoding** for high-cardinality categoricals (59 industries, 247 hosting providers)
- **Stratified split** to preserve class distribution
- **StandardScaler** for algorithms sensitive to feature scales

### Model Choice:
Tree-based models (Random Forest, Gradient Boosting) work best because they:
- Handle non-linear relationships
- Are robust to outliers
- Don't require feature scaling
- Handle class imbalance well with class weights

---

## Results

Expected performance (from testing):
- **F1-Score:** ~84-92%
- **ROC-AUC:** ~88-94%
- **Accuracy:** ~82-88%

The final model balances catching phishing sites (recall) while minimizing false alarms (precision).

---

## Files Generated

When you run the pipeline:
```
models/
├── phishing_detector.pkl    # Trained model
└── preprocessor.pkl          # Scaler, encoders, imputer

results/
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── feature_importance.csv
└── evaluation_metrics.txt
```

---

## Design Decisions

### Why F1-Score as Primary Metric?
- Balances precision and recall
- More meaningful than accuracy for imbalanced data
- Business goal: catch phishing sites without too many false alarms

### Why Frequency Encoding?
- One-hot encoding would create 300+ columns
- Target encoding risks overfitting
- Frequency encoding captures patterns while keeping dimensionality low

### Why Tree-Based Models?
- No need for feature scaling
- Handles outliers naturally
- Captures non-linear patterns
- Provides feature importance for interpretability

---

## Future Improvements

- Try XGBoost/LightGBM for better performance
- Experiment with SMOTE for handling class imbalance
- Add feature interactions
- Implement SHAP for better explainability
- Create REST API for deployment

---

## Notes

- Database file (`phishing.db`) is NOT included per assessment instructions
- Generated models and results are NOT included (will be created when pipeline runs)
- Pipeline takes ~3-5 minutes without tuning, ~10-15 minutes with tuning

---

**Last Updated:** November 2025
