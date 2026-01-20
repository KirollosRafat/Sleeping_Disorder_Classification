# Sleep Disorder Classification

A machine learning project that predicts sleep disorders (None, Sleep Apnea, or Insomnia) based on health and lifestyle factors using XGBoost classification.

## 📊 Project Overview

This project analyzes sleep health and lifestyle data to classify individuals into three categories:
- **None**: No sleep disorder
- **Sleep Apnea**: Sleep breathing disorder
- **Insomnia**: Sleep onset/maintenance disorder

## 🎯 Model Performance

The hyperparameter-tuned XGBoost classifier achieved:
- **Overall Accuracy**: 97%
- **Macro F1-Score**: 0.96

### Detailed Metrics by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| None (0) | 0.93 | 0.93 | 0.93 | 15 |
| Insomnia (1) | 1.00 | 1.00 | 1.00 | 44 |
| Sleep Apnea (2) | 0.94 | 0.94 | 0.94 | 16 |

## 📁 Project Structure

```
├── Sleep_health_and_lifestyle_dataset.csv  # Original dataset
├── Sleep_Disorder_Preprocessed.csv         # Processed dataset
├── PreProcess.ipynb                        # Data preprocessing notebook
├── Classifier.ipynb                        # Model training notebook
├── functions.py                            # Helper functions
└── README.md                               # This file
```

## 📋 Features

### Input Features
- **Gender**: Male/Female
- **Age Group**: Young, Adult, Middle-Aged, Senior
- **Occupation**: 11 different professions
- **Sleep Duration**: Hours of sleep per day
- **Quality of Sleep**: Scale 1-10
- **Physical Activity Level**: Minutes per day
- **Stress Level**: Scale 1-10
- **BMI Category**: Normal, Overweight, Obese
- **Heart Rate**: Beats per minute
- **Daily Steps**: Number of steps
- **BP Category**: Blood pressure classification (Normal, Elevated, Hypertension Stage 1/2)

### Target Variable
- **Sleep Disorder**: None, Sleep Apnea, or Insomnia

## 🔄 Preprocessing Pipeline

1. **Data Cleaning**
   - Removed Person ID column
   - Handled missing values in Sleep Disorder (NaN → "NONE")

2. **Feature Engineering**
   - Created `Age_Group` from continuous Age feature
   - Extracted `BP_Category` from Blood Pressure readings
   - Dropped redundant features (Age, Blood Pressure, Systolic, Diastolic)

3. **Encoding & Scaling**
   - Label Encoding for categorical features
   - MinMaxScaler for numerical features

## 🤖 Model Training

### Model Selection
Compared baseline models:
- **Random Forest Classifier**: F1-Score = 0.83
- **XGBoost Classifier**: F1-Score = 0.85 (Selected)

### Hyperparameter Tuning
Used GridSearchCV with StratifiedKFold (5 splits):

```python
param_grid = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 150, 200, 250]
}
```

**Best Parameters**:
- `max_depth`: 2
- `n_estimators`: 50
- Cross-validation F1-Score: 0.892

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sleep-disorder-classification.git
cd sleep-disorder-classification

# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Usage

1. **Preprocess the data**:
   ```bash
   jupyter notebook PreProcess.ipynb
   ```

2. **Train the model**:
   ```bash
   jupyter notebook Classifier.ipynb
   ```

3. **Use helper functions**:
   ```python
   from functions import add_bp_category, assign_age_group
   
   # Add blood pressure category
   df = add_bp_category(df, bp_col='Blood Pressure')
   
   # Assign age group
   df['Age_Group'] = df['Age'].apply(assign_age_group)
   ```

## 📈 Results

### Confusion Matrix
The model shows excellent classification performance across all three classes with minimal misclassifications.

### Key Insights
- No class imbalance detected (None: 58.6%, Sleep Apnea: 20.9%, Insomnia: 20.6%)
- Hyperparameter tuning significantly improved model performance
- Stratified K-Fold validation ensures reliable performance estimates

---

**Note**: This project uses a synthetic/research dataset for demonstration purposes. Always consult healthcare professionals for medical advice regarding sleep disorders.