# Phishing Website Detection

**Luvyn Louis Sequeira**  
luvynsequeira@gmail.com

AIAP Batch 22 Technical Assessment

---

## What This Does

Builds ML models to detect phishing websites. Includes EDA notebook and a complete training pipeline with 4 different classifiers.

## Files

```
data/phishing.db          # Get from assessment (don't upload to git)
src/                      # Python modules
  ├── data_preprocessing.py
  ├── model_training.py
  ├── model_evaluation.py
  └── run_pipeline.py
eda.ipynb                 # EDA notebook
requirements.txt
run.sh                    # Runs the whole pipeline
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (takes ~10-15 min)
bash run.sh

# Or skip hyperparameter tuning (~3-5 min)
python src/run_pipeline.py --no-tune

# View EDA
jupyter notebook eda.ipynb
```

## Data

10,500 websites (30% legit, 70% phishing)

**Features:**
- LineOfCode, LargestLineLength, NoOfURLRedirect, NoOfPopup, NoOfiFrame, NoOfImage
- Robots, IsResponsive, DomainAgeMonths
- Industry, HostingProvider (categorical)

**Issues:**
- 2,355 missing values in LineOfCode
- Class imbalance (70-30 split)
- High cardinality categoricals (59 industries, 247 hosting providers)

## What the Pipeline Does

1. **Preprocessing**
   - Impute missing values (median)
   - Encode categoricals (frequency encoding for high cardinality)
   - Scale features (StandardScaler)
   - 80-20 stratified split

2. **Training**
   - Tries 4 models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
   - All use `class_weight='balanced'` for imbalanced data
   - 5-fold cross-validation
   - GridSearch for best hyperparameters

3. **Evaluation**
   - F1-score (primary metric - better than accuracy for imbalanced data)
   - Confusion matrix, ROC curve, feature importance plots
   - Saves best model and results

## Results

From testing:
- F1-Score: ~84-92%
- ROC-AUC: ~88-94%
- Accuracy: ~82-88%

Best model is usually Random Forest or Gradient Boosting.

## Key Findings

Phishing sites typically have:
- Younger domains
- More URL redirects
- No robots.txt
- Less responsive design
- More popups

## Why These Choices?

**F1-score over accuracy:** Accuracy misleading with 70-30 split. F1 balances precision/recall.

**Frequency encoding:** One-hot would create 300+ columns. Target encoding overfits. Frequency encoding works well for high cardinality.

**Tree-based models:** Handle outliers, non-linear patterns, don't need scaling. Provide feature importance.

**Median imputation:** More robust to outliers than mean.

**Stratified split:** Keeps 70-30 ratio in train and test sets.

## Output Files

After running:
```
models/
  ├── phishing_detector.pkl
  └── preprocessor.pkl
results/
  ├── confusion_matrix.png
  ├── roc_curve.png
  ├── feature_importance.png
  └── evaluation_metrics.txt
```

---

**Note:** Database file not included per assessment instructions. Models regenerated on each run.
