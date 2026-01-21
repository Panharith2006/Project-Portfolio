# 🏥 Diabetes Prediction System

A machine learning project for predicting diabetes diagnosis using patient health metrics. Built with XGBoost, featuring comprehensive EDA, feature engineering, and a Streamlit web application for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)

---

## 📊 Project Overview

This project develops a binary classification model to predict diabetes diagnosis (diabetic/non-diabetic) using the **TAIPEI diabetes dataset** containing health metrics from 15,000 women patients.

### Key Features:
- ✅ **Comprehensive EDA**: Univariate, bivariate, and correlation analysis
- ✅ **Advanced Feature Engineering**: Binning, interaction terms, and derived features
- ✅ **Multiple ML Models**: XGBoost, Gradient Boosted Trees, Neural Networks, Logistic Regression
- ✅ **Hyperparameter Tuning**: RandomizedSearchCV optimization
- ✅ **Model Evaluation**: Cross-validation, ROC curves, confusion matrices, calibration plots
- ✅ **Web Application**: Streamlit interface for real-time predictions
- ✅ **Production Ready**: Model deployment with serialized artifacts

---

## 📁 Project Structure

```
Project Portfolio/
│
├── Data_Preparation_n_model (2).ipynb  # Main analysis notebook
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── TAIPEI_diabetes.csv                 # Dataset (15,000 samples)
├── best_diabetes_model.pkl             # Trained XGBoost model
├── scaler.pkl                          # StandardScaler for preprocessing
├── feature_names.pkl                   # Feature list for consistency
│
└── README.md                           # This file
```

---

## 📋 Dataset Description

**TAIPEI_diabetes.csv** contains 8 health features and 1 target variable:

| Feature | Description | Type |
|---------|-------------|------|
| `Pregnancies` | Number of times pregnant | Numeric |
| `PlasmaGlucose` | Plasma glucose concentration (2-hour OGTT) | Numeric |
| `DiastolicBloodPressure` | Diastolic blood pressure (mm Hg) | Numeric |
| `TricepsThickness` | Triceps skin fold thickness (mm) | Numeric |
| `SerumInsulin` | 2-hour serum insulin (mu U/ml) | Numeric |
| `BMI` | Body mass index (kg/m²) | Numeric |
| `DiabetesPedigree` | Family history score | Numeric |
| `Age` | Age in years | Numeric |
| **`Diabetic`** | **Target: 1 = diabetic, 0 = not diabetic** | **Binary** |

**Class Distribution**: ~66% non-diabetic, ~34% diabetic (imbalanced)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "d:\Year 3\Data Mining\Project Portfolio"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔬 Running the Analysis

### 1. Jupyter Notebook (Complete Analysis)

Open and run the main notebook:

```bash
jupyter notebook "Data_Preparation_n_model (2).ipynb"
```

**Notebook Sections**:
- 📊 Exploratory Data Analysis (EDA)
- 🔧 Data Preparation & Feature Engineering
- 🤖 Model Training (4 algorithms)
- ✅ Cross-Validation & Hyperparameter Tuning
- 📈 Model Evaluation & Comparison
- 🎯 Feature Importance Analysis
- 💾 Model Deployment Export

### 2. Streamlit Web Application

Launch the interactive prediction interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features**:
- 📝 Manual input form with validation
- 📤 Batch CSV upload
- 🎯 Real-time probability predictions
- 📊 Interactive visualizations
- 💡 Risk interpretation

---

## 🧠 Machine Learning Pipeline

### 1. Data Preparation
- **Outlier Handling**: IQR-based capping for `TricepsThickness` and `SerumInsulin`
- **Feature Engineering**:
  - Age groups (binning): Young, MiddleAge, Senior, Elderly
  - BMI categories: Underweight, Normal, Overweight, Obese
  - Interaction terms: `Glucose_BMI_interaction`, `Age_BMI_interaction`
  - Risk score: `Pregnancy_Age_risk`
- **Scaling**: StandardScaler (fit on training set only)
- **Train/Test Split**: 80/20 stratified split

### 2. Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 0.9543 | 0.9235 | 0.9410 | 0.9321 | 0.9913 |
| Gradient Boosted Tree | 0.9550 | 0.9382 | 0.9260 | 0.9321 | 0.9910 |
| Neural Network | 0.9023 | 0.8560 | 0.8560 | 0.8530 | 0.9594 |
| Logistic Regression | 0.8404 | 0.6732 | 0.7910 | 0.7274 | 0.8861 |

**Selected Model**: **XGBoost** (optimal balance of performance, interpretability, and robustness)

### 3. Hyperparameter Tuning

- **Method**: RandomizedSearchCV (20 iterations, 3-fold CV)
- **Optimized Parameters**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`
- **Scoring Metric**: ROC-AUC

### 4. Key Features (by Importance)

1. 🥇 **PlasmaGlucose** - Strongest predictor
2. 🥈 **BMI** - Major risk factor
3. 🥉 **Age** - Significant contributor
4. **Glucose_BMI_interaction** - Powerful interaction term
5. **DiabetesPedigree** - Family history impact
6. **SerumInsulin** - Secondary biomarker

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold Stratified)
- **Accuracy**: 0.9540 ± 0.0045
- **ROC-AUC**: 0.9910 ± 0.0020

### Test Set Performance
- **True Negatives**: High (minimal false alarms)
- **True Positives**: High (catches most diabetic cases)
- **False Negatives**: Low (critical for medical applications)
- **Calibration**: Well-calibrated probabilities

---

## 🖥️ Web Application Usage

### Single Prediction

1. Navigate to the **"Single Patient Prediction"** tab
2. Enter patient health metrics using the input form
3. Click **"Predict Diabetes Risk"**
4. View:
   - 🎯 Prediction result (Diabetic/Not Diabetic)
   - 📊 Probability score with confidence meter
   - 💡 Risk interpretation

### Batch Prediction

1. Navigate to the **"Batch Prediction"** tab
2. Upload a CSV file with the following columns:
   ```
   Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
   SerumInsulin, BMI, DiabetesPedigree, Age
   ```
3. Download predictions as CSV with:
   - Original features
   - Prediction labels
   - Probability scores

---

## 📊 Evaluation Metrics Explained

| Metric | Medical Interpretation |
|--------|------------------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Avoids unnecessary treatments (reduces false positives) |
| **Recall** | **Critical** - Catches all at-risk patients (minimizes false negatives) |
| **F1-Score** | Balanced performance on imbalanced data |
| **ROC-AUC** | Model's discrimination ability (higher = better) |

**For diabetes prediction, Recall is prioritized** to minimize missing diabetic patients.

---

## 🔧 Dependencies

Core libraries (see `requirements.txt`):

```
streamlit          # Web application framework
pandas             # Data manipulation
numpy              # Numerical operations
scikit-learn       # ML algorithms & preprocessing
xgboost            # Gradient boosting
joblib             # Model serialization
seaborn            # Statistical visualization
matplotlib         # Plotting
imbalanced-learn   # SMOTE (optional)
tensorflow         # Neural network
```

---

## 📝 Model Deployment Files

Generated after running the notebook:

- **`best_diabetes_model.pkl`** - Trained XGBoost classifier
- **`scaler.pkl`** - StandardScaler for feature preprocessing
- **`feature_names.pkl`** - Feature list for consistency checks

---

## 🎯 Clinical Insights

### Top Risk Factors Identified:

1. **High Plasma Glucose** (>140 mg/dL) - Strong diabetes indicator
2. **Elevated BMI** (>30 kg/m²) - Obesity correlation
3. **Age >40 years** - Risk increases with age
4. **Family History** - Genetic predisposition matters
5. **High Serum Insulin** - Insulin resistance signal

### Model Interpretation:

- **Medical-Grade Performance**: ROC-AUC >0.99
- **High Recall**: Minimizes missing diabetic cases (critical for screening)
- **Calibrated Probabilities**: Confidence scores are reliable
- **Explainable Features**: Aligns with established medical knowledge

---

## 🚨 Important Notes

### Limitations:
- ⚠️ **Not a substitute for professional medical diagnosis**
- ⚠️ Dataset is specific to women; generalization may vary
- ⚠️ Model trained on TAIPEI dataset; regional differences may apply

### Recommendations:
- ✅ Use as a screening tool or risk assessment aid
- ✅ Always consult healthcare professionals for diagnosis
- ✅ Combine with additional clinical tests (HbA1c, fasting glucose)
- ✅ Regular model retraining with new data

---

## 🤝 Contributing

This is an academic project for Data Mining coursework (Year 3). Feedback and suggestions are welcome!

---

## 📚 References

- **Dataset**: TAIPEI diabetes dataset (modified from Pima Indians Diabetes Database)
- **Algorithms**: XGBoost, Gradient Boosting, Neural Networks, Logistic Regression
- **Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix, Calibration Curves

---

## 📧 Contact

For questions or collaboration:
- **Project**: Year 3 Data Mining - Diabetes Prediction
- **Institution**: [Royal University of Phnom Penh]
- **Academic Year**: 2025-2026

---

## 📄 License

This project is for educational purposes as part of university coursework.

---

## 🎓 Acknowledgments

- Course: Data Mining (Year 3)
- Dataset: TAIPEI diabetes dataset
- Libraries: scikit-learn, XGBoost, Streamlit, pandas

---

**Last Updated**: January 21, 2026

---

## 🔍 Quick Start Commands

```bash
# Setup
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run notebook
jupyter notebook "Data_Preparation_n_model (2).ipynb"

# Launch web app
streamlit run app.py
```

---

**⭐ Medical AI for Diabetes Screening | Built with ❤️ and XGBoost**
