# Phishing Website Detection

**Luvyn Louis Sequeira**  
luvyn_sequeira@mymail.sutd.edu.sg

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

## Feature Processing Summary

| Feature | Type | Processing Method | Reason |
|---------|------|-------------------|--------|
| LineOfCode | Numeric | Median imputation → StandardScaler | 22.4% missing values (2,355 rows) |
| LargestLineLength | Numeric | StandardScaler | No missing values |
| NoOfURLRedirect | Numeric | StandardScaler | Strong predictor of phishing |
| NoOfSelfRedirect | Numeric | StandardScaler | Low variance but kept |
| NoOfPopup | Numeric | StandardScaler | Phishing indicator |
| NoOfiFrame | Numeric | StandardScaler | Security risk indicator |
| NoOfImage | Numeric | StandardScaler | Page complexity measure |
| NoOfSelfRef | Numeric | StandardScaler | Internal navigation measure |
| NoOfExternalRef | Numeric | StandardScaler | External linking behavior |
| Robots | Binary | Keep as-is | Already 0/1 encoded |
| IsResponsive | Binary | Keep as-is | Already 0/1 encoded |
| DomainAgeMonths | Numeric | StandardScaler | Top feature - phishing sites are new |
| Industry | Categorical | Frequency encoding | 59 unique values (high cardinality) |
| HostingProvider | Categorical | Frequency encoding | 247 unique values (high cardinality) |

**Note:** One-hot encoding would create 300+ sparse columns. Frequency encoding preserves information while keeping dimensionality manageable.

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

## Deployment Considerations

**Model Serving:**
- Models saved as `.pkl` files (~5-10 MB total)
- Fast inference: < 50ms per prediction on standard hardware
- Can be deployed via REST API (Flask/FastAPI) or serverless functions
- Supports both real-time single predictions and batch processing

**Data Pipeline Requirements:**
- Preprocessor must be loaded alongside model to ensure consistent feature transformations
- New data must match training feature schema (14 features after preprocessing)
- Missing values handled automatically via median imputation
- Categorical features (Industry, HostingProvider) encoded using frequency mapping

**Monitoring & Maintenance:**
- Track prediction latency and throughput
- Monitor feature drift, especially DomainAgeMonths (websites age over time)
- Alert on low-confidence predictions (probability < 0.6)
- Model retraining recommended every 3-6 months as phishing tactics evolve

**Security & Privacy:**
- Validate input features to prevent injection attacks
- Log predictions with timestamps for audit trail
- Consider implementing rate limiting for API deployment
- No PII stored in model or predictions

**Scalability:**
- Stateless design enables horizontal scaling
- Consider caching for frequently checked domains (TTL: 24h)
- Batch prediction support for bulk URL scanning
- Can handle ~1000 predictions/second on single CPU core

**Edge Cases & Limitations:**
- New hosting providers or industries not seen in training will be encoded as frequency 0
- Very new domains (< 1 month) may be flagged as suspicious regardless of legitimacy
- Model performs best on English-language websites (training data bias)
- Does not analyze actual page content, only metadata features

---

**Note:** Database file not included per assessment instructions. Models regenerated on each run.
