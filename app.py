import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_models():
    model = joblib.load('best_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("❌ Model files not found! Please run the notebook to generate 'best_diabetes_model.pkl' and 'scaler.pkl'")
    st.stop()

# Feature engineering function (matches notebook preprocessing)
def engineer_features(df):
    """Apply the same feature engineering as in training"""
    df = df.copy()
    
    # 1. Age groups (binning)
    df['AgeGroup'] = pd.cut(df['Age'], 
                            bins=[0, 30, 40, 50, 100], 
                            labels=['Young', 'MiddleAge', 'Senior', 'Elderly'])
    df['AgeGroup'] = df['AgeGroup'].cat.codes
    
    # 2. BMI categories (binning)
    df['BMI_Category'] = pd.cut(df['BMI'], 
                                bins=[0, 18.5, 25, 30, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['BMI_Category'] = df['BMI_Category'].cat.codes
    
    # 3. Interaction features
    df['Glucose_BMI_interaction'] = df['PlasmaGlucose'] * df['BMI']
    df['Age_BMI_interaction'] = df['Age'] * df['BMI']
    
    # 4. Pregnancy-Age risk
    df['Pregnancy_Age_risk'] = df['Pregnancies'] * df['Age']
    
    return df

def get_risk_level(probability):
    """Categorize risk level based on probability"""
    if probability < 0.3:
        return "🟢 LOW RISK", "#28a745"
    elif probability < 0.7:
        return "🟡 MEDIUM RISK", "#ffc107"
    else:
        return "🔴 HIGH RISK", "#dc3545"

# Title and description
st.title('🏥 Diabetes Prediction System')
st.markdown("""
**AI-powered diabetes risk assessment using XGBoost model**  
*Accuracy: 95.4% | ROC-AUC: 99.1% | Recall: 94.1%*
""")

st.divider()

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "ℹ️ About"])

with tab1:
    st.header('Enter Patient Information')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1, 
                                      help="Number of times pregnant")
        age = st.number_input('Age (years)', min_value=18, max_value=100, value=30)
        bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                             help="Body Mass Index (kg/m²)")
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 
                                           min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                                           help="Family history score (0-3)")
    
    with col2:
        st.subheader("Clinical Measurements")
        plasma_glucose = st.number_input('Plasma Glucose (mg/dL)', min_value=0, max_value=250, value=120,
                                        help="2-hour oral glucose tolerance test")
        diastolic_bp = st.number_input('Diastolic Blood Pressure (mm Hg)', 
                                      min_value=0, max_value=150, value=80)
        triceps_thickness = st.number_input('Triceps Thickness (mm)', 
                                           min_value=0, max_value=100, value=20)
        serum_insulin = st.number_input('Serum Insulin (μU/mL)', 
                                       min_value=0, max_value=900, value=100)
    
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        predict_button = st.button('🔍 Predict', type="primary", use_container_width=True)
    with col_btn2:
        if st.button('🔄 Reset', use_container_width=True):
            st.rerun()
    
    if predict_button:
        # Create input dataframe with base features
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'PlasmaGlucose': [plasma_glucose],
            'DiastolicBloodPressure': [diastolic_bp],
            'TricepsThickness': [triceps_thickness],
            'SerumInsulin': [serum_insulin],
            'BMI': [bmi],
            'DiabetesPedigree': [diabetes_pedigree],
            'Age': [age]
        })
        
        # Apply feature engineering (same as training)
        input_engineered = engineer_features(input_data)
        
        # Scale features
        input_scaled = scaler.transform(input_engineered)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.divider()
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="Diagnosis",
                value="DIABETIC" if prediction == 1 else "NON-DIABETIC",
                delta="Positive" if prediction == 1 else "Negative"
            )
        
        with result_col2:
            st.metric(
                label="Diabetes Probability",
                value=f"{probability:.1%}"
            )
        
        with result_col3:
            risk_label, risk_color = get_risk_level(probability)
            st.markdown(f"### {risk_label}")
        
        # Progress bar for probability
        st.progress(float(probability))
        
        # Recommendations
        st.divider()
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **⚠️ High Risk of Diabetes Detected**
            - Consult with a healthcare professional immediately
            - Get HbA1c test for confirmation
            - Monitor blood glucose levels regularly
            - Consider lifestyle modifications (diet & exercise)
            """)
        else:
            st.success("""
            **✅ Low Risk of Diabetes**
            - Maintain healthy lifestyle habits
            - Regular check-ups recommended
            - Keep BMI in healthy range (18.5-25)
            - Stay physically active
            """)

with tab2:
    st.header('📊 Batch Prediction from CSV')
    st.info("Upload a CSV file with patient data. Required columns: Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age")
    
    # Sample data download
    sample_data = pd.DataFrame({
        'Pregnancies': [1, 6, 2],
        'PlasmaGlucose': [85, 148, 120],
        'DiastolicBloodPressure': [70, 90, 75],
        'TricepsThickness': [19, 32, 25],
        'SerumInsulin': [95, 180, 140],
        'BMI': [22.5, 33.6, 28.0],
        'DiabetesPedigree': [0.150, 0.627, 0.400],
        'Age': [25, 50, 35]
    })
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_diabetes_data.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])
    
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✓ File uploaded successfully! {len(batch_df)} records found.")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button('🔍 Run Batch Prediction', type="primary"):
                with st.spinner('Processing predictions...'):
                    # Store PatientID if exists
                    has_patient_id = 'PatientID' in batch_df.columns
                    if has_patient_id:
                        patient_ids = batch_df['PatientID'].copy()
                        batch_df_clean = batch_df.drop('PatientID', axis=1)
                    else:
                        batch_df_clean = batch_df.copy()
                    
                    # Apply feature engineering
                    batch_engineered = engineer_features(batch_df_clean)
                    
                    # Scale features
                    batch_scaled = scaler.transform(batch_engineered)
                    
                    # Predictions
                    predictions = model.predict(batch_scaled)
                    probabilities = model.predict_proba(batch_scaled)[:, 1]
                    
                    # Create results dataframe
                    results_df = batch_df.copy()
                    results_df['Prediction'] = ['Diabetic' if p == 1 else 'Non-Diabetic' for p in predictions]
                    results_df['Probability'] = probabilities
                    results_df['Risk_Level'] = [get_risk_level(p)[0] for p in probabilities]
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patients", len(results_df))
                    with col2:
                        diabetic_count = (predictions == 1).sum()
                        st.metric("Diabetic Cases", diabetic_count)
                    with col3:
                        st.metric("Non-Diabetic Cases", len(results_df) - diabetic_count)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label='📥 Download Full Results',
                        data=csv,
                        file_name='diabetes_predictions.csv',
                        mime='text/csv',
                        type="primary"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct columns and format.")

with tab3:
    st.header('ℹ️ About This System')
    
    st.markdown("""
    ### 🎯 Model Performance
    
    This diabetes prediction system uses **XGBoost**, a state-of-the-art machine learning algorithm trained on 15,000 patient records.
    
    **Key Metrics:**
    - **Accuracy:** 95.43%
    - **Precision:** 92.35%
    - **Recall:** 94.10% (catches 94 out of 100 diabetic cases)
    - **ROC-AUC:** 99.13% (near-perfect discrimination)
    - **F1-Score:** 93.21%
    
    ### 📊 Dataset Information
    
    **TAIPEI Diabetes Dataset** - 15,000 women patients
    
    **Features Used:**
    - Pregnancies
    - Plasma Glucose (mg/dL)
    - Diastolic Blood Pressure (mm Hg)
    - Triceps Thickness (mm)
    - Serum Insulin (μU/mL)
    - BMI (Body Mass Index)
    - Diabetes Pedigree Function (family history)
    - Age (years)
    
    ### 🔬 Feature Importance
    
    Top predictive features:
    1. **Plasma Glucose** (18.5%) - Primary indicator
    2. **BMI** (14.5%) - Obesity correlation
    3. **Age** (12.8%) - Risk increases with age
    4. **Diabetes Pedigree** (11.2%) - Genetic predisposition
    
    ### ⚠️ Important Note
    
    This system is designed as a **clinical decision support tool** and should not replace professional medical diagnosis. 
    Always consult with healthcare professionals for proper diagnosis and treatment.
    
    ### 👥 Model Details
    
    - **Algorithm:** XGBoost Classifier
    - **Training Set:** 12,000 samples (80%)
    - **Test Set:** 3,000 samples (20%)
    - **Class Balance Handling:** scale_pos_weight
    - **Feature Engineering:** Age groups, BMI categories, interaction features
    
    ---
    
    *Developed by Data Mining Project Team | 2026*
    """)

# Footer
st.divider()
st.caption("⚕️ Diabetes Prediction System v1.0 | For educational and research purposes only")