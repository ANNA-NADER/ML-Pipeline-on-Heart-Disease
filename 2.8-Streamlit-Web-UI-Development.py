# Task 2.8: Streamlit Web UI Development
# Interactive Web Application for Heart Disease Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-positive {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .feature-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained heart disease prediction model"""
    try:
        model = joblib.load('final_heart_disease_model.pkl')
        return model, True
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'final_heart_disease_model.pkl' is in the directory.")
        return None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, False

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        if os.path.exists('heart_disease_cleaned.csv'):
            df = pd.read_csv('heart_disease_cleaned.csv')
            return df, True
        else:
            # Create sample data if file not found
            np.random.seed(42)
            n_samples = 300
            data = {
                'age': np.random.normal(54, 9, n_samples),
                'sex': np.random.choice([0, 1], n_samples),
                'cp': np.random.choice([1, 2, 3, 4], n_samples),
                'trestbps': np.random.normal(132, 18, n_samples),
                'chol': np.random.normal(247, 52, n_samples),
                'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
                'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.48, 0.02]),
                'thalach': np.random.normal(150, 23, n_samples),
                'exang': np.random.choice([0, 1], n_samples, p=[0.67, 0.33]),
                'oldpeak': np.random.exponential(1, n_samples),
                'slope': np.random.choice([1, 2, 3], n_samples, p=[0.46, 0.35, 0.19]),
                'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.23, 0.15, 0.08]),
                'thal': np.random.choice([3, 6, 7], n_samples, p=[0.55, 0.18, 0.27]),
                'target': np.random.choice([0, 1], n_samples, p=[0.54, 0.46])
            }
            df = pd.DataFrame(data)
            st.warning("⚠️ Using simulated data for visualization (original dataset not found)")
            return df, False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

def create_input_form():
    """Create the input form for user data"""
    st.markdown("<h3>📝 Patient Information</h3>", unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
        sex = st.selectbox("Sex", 
                          options=[0, 1], 
                          format_func=lambda x: "Female" if x == 0 else "Male",
                          help="Biological sex")
        
        st.markdown("### Cardiovascular Measurements")
        trestbps = st.slider("Resting Blood Pressure", 80, 220, 130, 
                           help="Resting blood pressure in mm Hg")
        chol = st.slider("Cholesterol", 100, 600, 240,
                        help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                          options=[0, 1],
                          format_func=lambda x: "No" if x == 0 else "Yes",
                          help="Fasting blood sugar greater than 120 mg/dl")
        
        st.markdown("### Heart Rate & Exercise")
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150,
                           help="Maximum heart rate achieved during exercise")
        exang = st.selectbox("Exercise Induced Angina",
                           options=[0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes",
                           help="Exercise induced chest pain")
    
    with col2:
        st.markdown("### Clinical Tests")
        cp = st.selectbox("Chest Pain Type", 
                         options=[1, 2, 3, 4],
                         format_func=lambda x: {
                             1: "Typical Angina",
                             2: "Atypical Angina", 
                             3: "Non-anginal Pain",
                             4: "Asymptomatic"
                         }[x],
                         help="Type of chest pain experienced")
        
        restecg = st.selectbox("Resting ECG",
                             options=[0, 1, 2],
                             format_func=lambda x: {
                                 0: "Normal",
                                 1: "ST-T Wave Abnormality",
                                 2: "Left Ventricular Hypertrophy"
                             }[x],
                             help="Resting electrocardiographic results")
        
        oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1,
                           help="ST depression induced by exercise relative to rest")
        
        slope = st.selectbox("ST Segment Slope",
                           options=[1, 2, 3],
                           format_func=lambda x: {
                               1: "Upsloping",
                               2: "Flat", 
                               3: "Downsloping"
                           }[x],
                           help="Slope of the peak exercise ST segment")
        
        ca = st.selectbox("Major Vessels Colored",
                         options=[0, 1, 2, 3],
                         help="Number of major vessels colored by fluoroscopy")
        
        thal = st.selectbox("Thallium Heart Scan",
                          options=[3, 6, 7],
                          format_func=lambda x: {
                              3: "Normal",
                              6: "Fixed Defect",
                              7: "Reversible Defect"
                          }[x],
                          help="Results of thallium heart scan")
    
    # Create input dictionary
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return input_data

def make_prediction(model, input_data):
    """Make prediction using the loaded model"""
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction_result(prediction, probability):
    """Display the prediction result with styling"""
    if prediction is None:
        return
    
    prob_no_disease = probability[0]
    prob_disease = probability[1]
    
    st.markdown("---")
    st.markdown("<h3>🔬 Prediction Results</h3>", unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Prediction",
            value="Heart Disease" if prediction == 1 else "No Heart Disease",
            delta="High Risk" if prediction == 1 else "Low Risk"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Disease Probability",
            value=f"{prob_disease:.1%}",
            delta=f"{prob_disease - 0.5:+.1%}" if prob_disease > 0.5 else f"{prob_disease - 0.5:+.1%}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Confidence",
            value=f"{max(prob_no_disease, prob_disease):.1%}",
            delta="High" if max(prob_no_disease, prob_disease) > 0.8 else "Medium" if max(prob_no_disease, prob_disease) > 0.6 else "Low"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction box with color coding
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box prediction-positive">
            <h3>⚠️ Heart Disease Predicted</h3>
            <p>The model predicts a <strong>{prob_disease:.1%}</strong> probability of heart disease.</p>
            <p><em>Please consult with a healthcare professional for proper medical advice.</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box prediction-negative">
            <h3>✅ No Heart Disease Predicted</h3>
            <p>The model predicts a <strong>{prob_no_disease:.1%}</strong> probability of no heart disease.</p>
            <p><em>Continue maintaining a healthy lifestyle and regular checkups.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability visualization
    fig = go.Figure(data=[
        go.Bar(
            x=['No Heart Disease', 'Heart Disease'],
            y=[prob_no_disease, prob_disease],
            marker_color=['#4CAF50' if prob_no_disease > prob_disease else '#FFEB3B', 
                         '#F44336' if prob_disease > prob_no_disease else '#FFEB3B'],
            text=[f'{prob_no_disease:.1%}', f'{prob_disease:.1%}'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Condition",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_data_visualizations(df):
    """Create data visualizations for exploration"""
    if df is None:
        return
    
    st.markdown("<h3>📊 Heart Disease Data Exploration</h3>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Demographics", "🩺 Clinical Features", "📋 Correlations", "📊 Statistics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(
                df, x='age', color='target',
                title="Age Distribution by Heart Disease Status",
                labels={'target': 'Heart Disease', 'age': 'Age'},
                marginal="box"
            )
            fig_age.update_layout(
                xaxis_title="Age (years)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Sex distribution
            sex_counts = df.groupby(['sex', 'target']).size().unstack()
            sex_counts.index = ['Female', 'Male']
            sex_counts.columns = ['No Heart Disease', 'Heart Disease']
            
            fig_sex = px.bar(
                sex_counts, 
                title="Heart Disease by Gender",
                labels={'index': 'Gender', 'value': 'Count'},
                color_discrete_map={'No Heart Disease': '#4CAF50', 'Heart Disease': '#F44336'}
            )
            st.plotly_chart(fig_sex, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Chest pain types
            cp_counts = df.groupby(['cp', 'target']).size().unstack()
            cp_counts.index = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
            cp_counts.columns = ['No Heart Disease', 'Heart Disease']
            
            fig_cp = px.bar(
                cp_counts,
                title="Heart Disease by Chest Pain Type",
                labels={'index': 'Chest Pain Type', 'value': 'Count'},
                color_discrete_map={'No Heart Disease': '#4CAF50', 'Heart Disease': '#F44336'}
            )
            fig_cp.update_xaxis(tickangle=45)
            st.plotly_chart(fig_cp, use_container_width=True)
        
        with col2:
            # Blood pressure vs cholesterol
            fig_scatter = px.scatter(
                df, x='trestbps', y='chol', color='target',
                title="Blood Pressure vs Cholesterol",
                labels={'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol', 'target': 'Heart Disease'},
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Correlation matrix
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        corr_matrix = df[numeric_cols + ['target']].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        # Statistical summary
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Heart Disease Cases", (df['target'] == 1).sum())
        with col3:
            st.metric("Average Age", f"{df['age'].mean():.1f}")
        with col4:
            st.metric("Disease Rate", f"{(df['target'] == 1).mean():.1%}")
        
        st.markdown("### 📊 Feature Statistics")
        st.dataframe(df.describe().round(2))

def display_model_info():
    """Display model information and instructions"""
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### Heart Disease Prediction Model
        
        This application uses a machine learning model trained on the UCI Heart Disease dataset 
        to predict the likelihood of heart disease based on clinical features.
        
        **Model Features:**
        - **Algorithm**: Logistic Regression (optimized)
        - **Accuracy**: ~87%
        - **Features Used**: 8 selected features from 13 original features
        - **Training Data**: 303 patients from Cleveland Clinic
        
        **Selected Features:**
        - Age, Sex, Chest Pain Type
        - Resting Blood Pressure, Maximum Heart Rate
        - Exercise-induced Angina, ST Depression
        - Major Vessels Count, Thallium Scan Results
        
        **⚠️ Medical Disclaimer:**
        This tool is for educational purposes only and should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment.
        """)

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("Cannot proceed without the trained model. Please ensure the model file exists.")
        st.stop()
    
    # Load sample data for visualization
    df, data_loaded = load_sample_data()
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Prediction", "📊 Data Explorer", "ℹ️ About"]
    )
    
    if page == "🏠 Prediction":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Instructions")
        st.sidebar.info("""
        1. Fill in the patient information
        2. Click 'Make Prediction' 
        3. View the results and probability
        4. Consult healthcare professional for medical advice
        """)
        
        # Main prediction interface
        input_data = create_input_form()
        
        # Prediction button
        if st.button("🔬 Make Prediction", type="primary"):
            with st.spinner("Analyzing patient data..."):
                prediction, probability = make_prediction(model, input_data)
                display_prediction_result(prediction, probability)
        
        # Model info
        display_model_info()
    
    elif page == "📊 Data Explorer":
        st.markdown("### Explore Heart Disease Trends")
        if data_loaded:
            create_data_visualizations(df)
        else:
            st.warning("Data visualization unavailable - dataset not found")
    
    elif page == "ℹ️ About":
        st.markdown("### About This Project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Project Overview
            This is a comprehensive machine learning project for heart disease prediction, 
            featuring:
            
            - **Data Preprocessing & Cleaning**
            - **Feature Selection & PCA**
            - **Multiple ML Algorithms**
            - **Hyperparameter Tuning**
            - **Model Deployment**
            - **Interactive Web Interface**
            
            #### 📊 Model Performance
            - **Accuracy**: 86.89%
            - **F1-Score**: 86.67%
            - **AUC Score**: 94.59%
            """)
        
        with col2:
            st.markdown("""
            #### 🔬 Technical Stack
            - **Machine Learning**: Scikit-learn
            - **Data Processing**: Pandas, NumPy
            - **Visualization**: Plotly, Streamlit
            - **Model**: Logistic Regression
            - **Deployment**: Streamlit + Ngrok
            
            #### 📅 Project Timeline
            - **Data Analysis**: Complete
            - **Model Training**: Complete
            - **Web Interface**: Complete
            - **Deployment**: Ready
            """)
        
        st.markdown("---")
        st.markdown("#### 🏥 Dataset Information")
        st.info("""
        **UCI Heart Disease Dataset**
        - Source: Cleveland Clinic Foundation
        - Patients: 303 individuals
        - Features: 14 clinical features
        - Target: Binary classification (heart disease presence)
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        f"Heart Disease Prediction System | Built with Streamlit | "
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()