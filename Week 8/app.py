import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Page config
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('best_student_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run the training notebook first.")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('student-por.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please ensure 'student-por.csv' is in the directory.")
        return None

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=80)
    st.title("🎓 Navigation")
    
    pages = {
        "🏠 Home Dashboard": "Home",
        "🎯 Student Prediction": "Prediction",
        "📊 Bulk Prediction": "Bulk",
        "📈 Analytics Dashboard": "Analytics",
        "⚠️ At-Risk Students": "AtRisk",
        "🔬 Model Performance": "Model",
        "👥 Student Comparison": "Comparison",
        "📄 Reports Generator": "Reports"
    }
    
    for display_name, page_key in pages.items():
        if st.button(display_name, key=page_key):
            st.session_state.page = page_key
    
    st.markdown("---")
    st.info("💡 **Tip**: Navigate through different sections to explore student performance analytics and predictions.")

# Main content
model, scaler, label_encoders, feature_names = load_models()
df = load_data()

if df is not None:
    # ====================
    # HOME DASHBOARD
    # ====================
    if st.session_state.page == 'Home':
        st.markdown('<div class="main-header">🎓 Student Performance Prediction & Analytics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Student Performance Analytics Platform
        This intelligent system helps educators identify at-risk students, predict academic outcomes, 
        and make data-driven decisions to improve student success rates.
        """)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Students", f"{len(df):,}")
        with col2:
            avg_grade = df['G3'].mean()
            st.metric("📊 Average Grade", f"{avg_grade:.2f}/20")
        with col3:
            pass_rate = (df['G3'] >= 10).mean() * 100
            st.metric("✅ Pass Rate", f"{pass_rate:.1f}%")
        with col4:
            at_risk = ((df['G3'] < 10) | (df['failures'] > 0)).sum()
            st.metric("⚠️ At-Risk Students", f"{at_risk}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Grade Distribution")
            fig = px.histogram(df, x='G3', nbins=20, 
                             title='Distribution of Final Grades',
                             labels={'G3': 'Final Grade', 'count': 'Number of Students'},
                             color_discrete_sequence=['#1f77b4'])
            fig.add_vline(x=10, line_dash="dash", line_color="red", 
                         annotation_text="Pass Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Pass/Fail Distribution")
            pass_fail = df['G3'].apply(lambda x: 'Pass (≥10)' if x >= 10 else 'Fail (<10)').value_counts()
            fig = px.pie(values=pass_fail.values, names=pass_fail.index,
                        title='Pass vs Fail Rate',
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance by demographics
        st.subheader("👥 Performance by Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender_perf = df.groupby('sex')['G3'].mean().reset_index()
            gender_perf['sex'] = gender_perf['sex'].map({'F': 'Female', 'M': 'Male'})
            fig = px.bar(gender_perf, x='sex', y='G3',
                        title='Average Grade by Gender',
                        labels={'sex': 'Gender', 'G3': 'Average Grade'},
                        color='sex',
                        color_discrete_map={'Female': '#e74c3c', 'Male': '#3498db'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            age_perf = df.groupby('age')['G3'].mean().reset_index()
            fig = px.line(age_perf, x='age', y='G3',
                         title='Average Grade by Age',
                         labels={'age': 'Age', 'G3': 'Average Grade'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            address_perf = df.groupby('address')['G3'].mean().reset_index()
            address_perf['address'] = address_perf['address'].map({'U': 'Urban', 'R': 'Rural'})
            fig = px.bar(address_perf, x='address', y='G3',
                        title='Average Grade by Location',
                        labels={'address': 'Location', 'G3': 'Average Grade'},
                        color='address',
                        color_discrete_map={'Urban': '#f39c12', 'Rural': '#9b59b6'})
            st.plotly_chart(fig, use_container_width=True)

    # ====================
    # STUDENT PREDICTION
    # ====================
    elif st.session_state.page == 'Prediction':
        st.markdown('<div class="main-header">🎯 Individual Student Prediction</div>', unsafe_allow_html=True)
        
        st.info("📝 Enter student details below to predict their final grade and risk level.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            # Demographic information
            with col1:
                st.subheader("📋 Demographics")
                school = st.selectbox("School", ["GP", "MS"])
                sex = st.selectbox("Gender", ["F", "M"])
                age = st.number_input("Age", min_value=15, max_value=22, value=16)
                address = st.selectbox("Address", ["U - Urban", "R - Rural"])
                famsize = st.selectbox("Family Size", ["GT3 - Greater than 3", "LE3 - Less/Equal 3"])
                Pstatus = st.selectbox("Parent Status", ["T - Together", "A - Apart"])
            
            # Family background
            with col2:
                st.subheader("👨‍👩‍👧 Family Background")
                Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], 
                                   format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x])
                Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4],
                                   format_func=lambda x: ["None", "Primary", "5th-9th", "Secondary", "Higher"][x])
                Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
                Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
                reason = st.selectbox("Reason for School", ["course", "home", "reputation", "other"])
                guardian = st.selectbox("Guardian", ["mother", "father", "other"])
            
            # Academic information
            with col3:
                st.subheader("📚 Academic Info")
                traveltime = st.selectbox("Travel Time", [1, 2, 3, 4],
                                         format_func=lambda x: ["<15min", "15-30min", "30min-1h", ">1h"][x-1])
                studytime = st.selectbox("Study Time", [1, 2, 3, 4],
                                        format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
                failures = st.number_input("Past Failures", min_value=0, max_value=4, value=0)
                schoolsup = st.selectbox("School Support", ["yes", "no"])
                famsup = st.selectbox("Family Support", ["yes", "no"])
                paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
            
            col1, col2, col3 = st.columns(3)
            
            # Activities and lifestyle
            with col1:
                st.subheader("🎯 Activities")
                activities = st.selectbox("Extra-curricular", ["yes", "no"])
                nursery = st.selectbox("Attended Nursery", ["yes", "no"])
                higher = st.selectbox("Wants Higher Ed", ["yes", "no"])
                internet = st.selectbox("Internet Access", ["yes", "no"])
                romantic = st.selectbox("Romantic Relationship", ["yes", "no"])
            
            # Social behavior
            with col2:
                st.subheader("👥 Social Behavior")
                famrel = st.slider("Family Relationship", 1, 5, 4)
                freetime = st.slider("Free Time", 1, 5, 3)
                goout = st.slider("Going Out", 1, 5, 3)
                Dalc = st.slider("Weekday Alcohol", 1, 5, 1)
                Walc = st.slider("Weekend Alcohol", 1, 5, 1)
            
            # Health and grades
            with col3:
                st.subheader("💪 Health & Performance")
                health = st.slider("Health Status", 1, 5, 3)
                absences = st.number_input("Absences", min_value=0, max_value=93, value=0)
                G1 = st.number_input("Period 1 Grade (G1)", min_value=0, max_value=20, value=10)
                G2 = st.number_input("Period 2 Grade (G2)", min_value=0, max_value=20, value=10)
            
            submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True)
        
        if submitted and model is not None:
            # Prepare input data
            input_data = {
                'school': school, 'sex': sex, 'age': age,
                'address': address.split(' - ')[0],
                'famsize': famsize.split(' - ')[0],
                'Pstatus': Pstatus.split(' - ')[0],
                'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
                'reason': reason, 'guardian': guardian,
                'traveltime': traveltime, 'studytime': studytime,
                'failures': failures, 'schoolsup': schoolsup,
                'famsup': famsup, 'paid': paid, 'activities': activities,
                'nursery': nursery, 'higher': higher, 'internet': internet,
                'romantic': romantic, 'famrel': famrel, 'freetime': freetime,
                'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
                'health': health, 'absences': absences, 'G1': G1, 'G2': G2
            }
            
            # Create dataframe
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in label_encoders.keys():
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Binary encoding
            binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
            for col in binary_cols:
                input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
            
            # Feature engineering
            input_df['parent_edu_avg'] = (input_df['Medu'] + input_df['Fedu']) / 2
            input_df['parent_edu_max'] = input_df[['Medu', 'Fedu']].max(axis=1)
            input_df['alc_avg'] = (input_df['Dalc'] + input_df['Walc']) / 2
            input_df['social_score'] = (input_df['goout'] + input_df['freetime'] + input_df['romantic']) / 3
            input_df['support_score'] = (input_df['schoolsup'] + input_df['famsup'] + input_df['paid']) / 3
            input_df['grade_improvement'] = input_df['G2'] - input_df['G1']
            input_df['at_risk'] = ((input_df['failures'] > 0) | (input_df['absences'] > 10) | (input_df['G2'] < 10)).astype(int)
            input_df['study_efficiency'] = input_df['G2'] / (input_df['studytime'] + 1)
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Select features in correct order
            input_scaled = scaler.transform(input_df[feature_names])
            
            # Make prediction
            predicted_grade = model.predict(input_scaled)[0]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎓 Predicted Final Grade", f"{predicted_grade:.2f}/20")
            
            with col2:
                if predicted_grade >= 16:
                    risk_level = "Low Risk"
                    risk_color = "🟢"
                elif predicted_grade >= 10:
                    risk_level = "Medium Risk"
                    risk_color = "🟡"
                else:
                    risk_level = "High Risk"
                    risk_color = "🔴"
                st.metric("⚠️ Risk Level", f"{risk_color} {risk_level}")
            
            with col3:
                pass_status = "✅ Pass" if predicted_grade >= 10 else "❌ Fail"
                st.metric("📊 Status", pass_status)
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            
            recommendations = []
            
            if predicted_grade < 10:
                recommendations.append("🔴 **Urgent intervention required** - Student at high risk of failure")
            if failures > 0:
                recommendations.append("📚 **Academic support needed** - History of past failures")
            if absences > 10:
                recommendations.append("⏰ **Improve attendance** - High absence rate affecting performance")
            if studytime < 2:
                recommendations.append("📖 **Increase study time** - Currently studying less than 2 hours/week")
            if schoolsup == "no" and predicted_grade < 12:
                recommendations.append("🏫 **Recommend school support** - May benefit from extra tutoring")
            if internet == "no":
                recommendations.append("💻 **Internet access** - May help with homework and research")
            if G1 < G2 < predicted_grade:
                recommendations.append("✨ **Positive trend** - Student showing improvement over time")
            if health < 3:
                recommendations.append("💊 **Health concerns** - May need medical attention")
            if Dalc > 3 or Walc > 3:
                recommendations.append("⚠️ **Alcohol consumption** - May be affecting academic performance")
            if famsup == "no":
                recommendations.append("👨‍👩‍👧 **Engage parents** - Family support can improve outcomes")
            
            if not recommendations:
                recommendations.append("✅ **Good trajectory** - Continue current study habits")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

    # ====================
    # BULK PREDICTION
    # ====================
    elif st.session_state.page == 'Bulk':
        st.markdown('<div class="main-header">📊 Bulk Student Prediction</div>', unsafe_allow_html=True)
        
        st.info("📁 Upload a CSV file with student data to get batch predictions.")
        
        # Download template
        st.subheader("📥 Download Template")
        template_df = pd.DataFrame(columns=df.columns[:-3])  # Exclude G1, G2, G3
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV Template",
            data=csv,
            file_name="student_template.csv",
            mime="text/csv"
        )
        
        # File upload
        st.subheader("📤 Upload Student Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! {len(upload_df)} students found.")
                
                st.dataframe(upload_df.head(), use_container_width=True)
                
                if st.button("🔮 Generate Predictions", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        # Process data similar to individual prediction
                        # ...existing preprocessing code...
                        
                        # Make predictions
                        predictions = model.predict(scaler.transform(upload_df[feature_names]))
                        upload_df['Predicted_Grade'] = predictions
                        upload_df['Risk_Level'] = upload_df['Predicted_Grade'].apply(
                            lambda x: 'Low' if x >= 16 else 'Medium' if x >= 10 else 'High'
                        )
                        upload_df['Pass_Status'] = upload_df['Predicted_Grade'].apply(
                            lambda x: 'Pass' if x >= 10 else 'Fail'
                        )
                        
                        st.success("✅ Predictions completed!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📚 Total Students", len(upload_df))
                        with col2:
                            st.metric("📊 Avg Predicted Grade", f"{upload_df['Predicted_Grade'].mean():.2f}")
                        with col3:
                            pass_rate = (upload_df['Pass_Status'] == 'Pass').mean() * 100
                            st.metric("✅ Pass Rate", f"{pass_rate:.1f}%")
                        with col4:
                            high_risk = (upload_df['Risk_Level'] == 'High').sum()
                            st.metric("⚠️ High Risk", high_risk)
                        
                        # Display results
                        st.dataframe(upload_df, use_container_width=True)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_result = upload_df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download CSV",
                                csv_result,
                                "predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Excel download
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                upload_df.to_excel(writer, index=False, sheet_name='Predictions')
                            excel_data = buffer.getvalue()
                            
                            st.download_button(
                                "⬇️ Download Excel",
                                excel_data,
                                "predictions.xlsx",
                                "application/vnd.ms-excel",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # ====================
    # ANALYTICS DASHBOARD
    # ====================
    elif st.session_state.page == 'Analytics':
        st.markdown('<div class="main-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance Trends", 
            "👥 Demographics", 
            "📚 Study Habits", 
            "📅 Attendance", 
            "👨‍👩‍👧 Parental Influence"
        ])
        
        # Performance Trends Tab
        with tab1:
            st.subheader("Grade Progression Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Grade progression over periods
                grade_prog = df[['G1', 'G2', 'G3']].mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=['Period 1', 'Period 2', 'Final'],
                    y=grade_prog.values,
                    mode='lines+markers',
                    name='Average Grade',
                    line=dict(width=3, color='#3498db'),
                    marker=dict(size=12)
                ))
                fig.update_layout(
                    title='Average Grade Progression',
                    xaxis_title='Period',
                    yaxis_title='Grade',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pass/Fail trend
                pass_rates = []
                for col in ['G1', 'G2', 'G3']:
                    pass_rates.append((df[col] >= 10).mean() * 100)
                
                fig = go.Figure(data=[
                    go.Bar(x=['Period 1', 'Period 2', 'Final'],
                          y=pass_rates,
                          marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
                          text=[f'{rate:.1f}%' for rate in pass_rates],
                          textposition='auto')
                ])
                fig.update_layout(
                    title='Pass Rate by Period',
                    xaxis_title='Period',
                    yaxis_title='Pass Rate (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Grade distribution by category
            st.subheader("Performance Distribution")
            
            def categorize_grade(grade):
                if grade >= 16: return 'Excellent (16-20)'
                elif grade >= 14: return 'Good (14-15)'
                elif grade >= 10: return 'Average (10-13)'
                else: return 'Poor (<10)'
            
            categories = df['G3'].apply(categorize_grade).value_counts()
            
            fig = px.pie(
                values=categories.values,
                names=categories.index,
                title='Final Grade Distribution by Category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Demographics Tab
        with tab2:
            st.subheader("Performance by Demographics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Gender analysis
                gender_perf = df.groupby('sex')['G3'].agg(['mean', 'count']).reset_index()
                gender_perf['sex'] = gender_perf['sex'].map({'F': 'Female', 'M': 'Male'})
                
                fig = px.bar(
                    gender_perf, x='sex', y='mean',
                    title='Average Grade by Gender',
                    labels={'sex': 'Gender', 'mean': 'Average Grade'},
                    color='sex',
                    text='mean'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age analysis
                age_perf = df.groupby('age')['G3'].mean().reset_index()
                
                fig = px.line(
                    age_perf, x='age', y='G3',
                    title='Average Grade by Age',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Address analysis
                address_perf = df.groupby('address')['G3'].mean().reset_index()
                address_perf['address'] = address_perf['address'].map({'U': 'Urban', 'R': 'Rural'})
                
                fig = px.bar(
                    address_perf, x='address', y='G3',
                    title='Average Grade by Location',
                    color='address'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # School comparison
            st.subheader("School Performance Comparison")
            
            school_stats = df.groupby('school').agg({
                'G3': ['mean', 'median', 'std', 'count']
            }).round(2)
            school_stats.columns = ['Mean Grade', 'Median Grade', 'Std Dev', 'Student Count']
            
            st.dataframe(school_stats, use_container_width=True)
        
        # Study Habits Tab
        with tab3:
            st.subheader("Study Habits Impact Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Study time vs grades
                studytime_perf = df.groupby('studytime')['G3'].mean().reset_index()
                studytime_labels = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
                studytime_perf['studytime_label'] = studytime_perf['studytime'].map(studytime_labels)
                
                fig = px.bar(
                    studytime_perf, x='studytime_label', y='G3',
                    title='Study Time vs Performance',
                    labels={'studytime_label': 'Weekly Study Time', 'G3': 'Average Grade'},
                    color='G3',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Failures impact
                failures_perf = df.groupby('failures')['G3'].mean().reset_index()
                
                fig = px.line(
                    failures_perf, x='failures', y='G3',
                    title='Past Failures Impact on Performance',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Number of Past Failures',
                    yaxis_title='Average Grade'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Going out frequency
            st.subheader("Social Activities Impact")
            
            goout_perf = df.groupby('goout')['G3'].mean().reset_index()
            goout_labels = {1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'}
            goout_perf['goout_label'] = goout_perf['goout'].map(goout_labels)
            
            fig = px.bar(
                goout_perf, x='goout_label', y='G3',
                title='Going Out Frequency vs Performance',
                color='G3',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Attendance Tab
        with tab4:
            st.subheader("Attendance Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: Absences vs Grade
                fig = px.scatter(
                    df, x='absences', y='G3',
                    title='Absences vs Final Grade',
                    trendline='ols',
                    opacity=0.6
                )
                fig.update_layout(
                    xaxis_title='Number of Absences',
                    yaxis_title='Final Grade'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation coefficient
                corr = df['absences'].corr(df['G3'])
                st.metric("Correlation Coefficient", f"{corr:.3f}")
            
            with col2:
                # Absence categories
                df['absence_category'] = pd.cut(
                    df['absences'],
                    bins=[-1, 0, 5, 10, 20, 100],
                    labels=['0', '1-5', '6-10', '11-20', '>20']
                )
                
                absence_perf = df.groupby('absence_category')['G3'].mean().reset_index()
                
                fig = px.bar(
                    absence_perf, x='absence_category', y='G3',
                    title='Performance by Absence Range',
                    color='G3',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            st.subheader("Grade Distribution by Attendance")
            
            fig = px.box(
                df, x='absence_category', y='G3',
                title='Grade Distribution Across Absence Categories',
                color='absence_category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Parental Influence Tab
        with tab5:
            st.subheader("Parental Education Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mother's education
                medu_perf = df.groupby('Medu')['G3'].mean().reset_index()
                edu_labels = {0: 'None', 1: 'Primary', 2: '5th-9th', 3: 'Secondary', 4: 'Higher'}
                medu_perf['Medu_label'] = medu_perf['Medu'].map(edu_labels)
                
                fig = px.line(
                    medu_perf, x='Medu_label', y='G3',
                    title="Mother's Education Impact",
                    markers=True,
                    line_shape='spline'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Father's education
                fedu_perf = df.groupby('Fedu')['G3'].mean().reset_index()
                fedu_perf['Fedu_label'] = fedu_perf['Fedu'].map(edu_labels)
                
                fig = px.line(
                    fedu_perf, x='Fedu_label', y='G3',
                    title="Father's Education Impact",
                    markers=True,
                    line_shape='spline'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Combined parent education
            st.subheader("Combined Parental Education")
            
            df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
            parent_edu_perf = df.groupby(df['parent_edu_avg'].round())['G3'].mean().reset_index()
            
            fig = px.bar(
                parent_edu_perf, x='parent_edu_avg', y='G3',
                title='Average Parent Education vs Student Performance',
                color='G3',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ====================
    # AT-RISK STUDENTS
    # ====================
    elif st.session_state.page == 'AtRisk':
        st.markdown('<div class="main-header">⚠️ At-Risk Students Identification</div>', unsafe_allow_html=True)
        
        # Define at-risk criteria
        at_risk_df = df[
            (df['G3'] < 10) | 
            (df['failures'] > 0) | 
            (df['absences'] > 10) |
            (df['G2'] < 10)
        ].copy()
        
        # Calculate risk score
        def calculate_risk_score(row):
            score = 0
            if row['G3'] < 10: score += 40
            if row['failures'] > 0: score += 30
            if row['absences'] > 10: score += 20
            if row['studytime'] < 2: score += 10
            return min(score, 100)
        
        at_risk_df['risk_score'] = at_risk_df.apply(calculate_risk_score, axis=1)
        at_risk_df['risk_level'] = pd.cut(
            at_risk_df['risk_score'],
            bins=[0, 40, 70, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total At-Risk Students", len(at_risk_df))
        with col2:
            high_risk = (at_risk_df['risk_level'] == 'High Risk').sum()
            st.metric("High Risk", high_risk, delta=f"{high_risk/len(at_risk_df)*100:.1f}%")
        with col3:
            medium_risk = (at_risk_df['risk_level'] == 'Medium Risk').sum()
            st.metric("Medium Risk", medium_risk)
        with col4:
            low_risk = (at_risk_df['risk_level'] == 'Low Risk').sum()
            st.metric("Low Risk", low_risk)
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_dist = at_risk_df['risk_level'].value_counts()
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title='Risk Level Distribution',
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk factors
            risk_factors = {
                'Failing Grade': (at_risk_df['G3'] < 10).sum(),
                'Past Failures': (at_risk_df['failures'] > 0).sum(),
                'High Absences': (at_risk_df['absences'] > 10).sum(),
                'Low Study Time': (at_risk_df['studytime'] < 2).sum()
            }
            
            fig = px.bar(
                x=list(risk_factors.keys()),
                y=list(risk_factors.values()),
                title='Risk Factors Distribution',
                labels={'x': 'Risk Factor', 'y': 'Count'},
                color=list(risk_factors.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Filters
        st.subheader("🔍 Filter At-Risk Students")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_risk = st.multiselect(
                "Risk Level",
                options=['High Risk', 'Medium Risk', 'Low Risk'],
                default=['High Risk']
            )
        
        with col2:
            min_absences = st.slider("Minimum Absences", 0, int(at_risk_df['absences'].max()), 0)
        
        with col3:
            max_grade = st.slider("Maximum Grade", 0, 20, 10)
        
        # Apply filters
        filtered_df = at_risk_df[
            (at_risk_df['risk_level'].isin(selected_risk)) &
            (at_risk_df['absences'] >= min_absences) &
            (at_risk_df['G3'] <= max_grade)
        ]
        
        # Display table
        st.subheader(f"At-Risk Students List ({len(filtered_df)} students)")
        
        display_columns = ['school', 'sex', 'age', 'G1', 'G2', 'G3', 'failures', 
                          'absences', 'studytime', 'risk_score', 'risk_level']
        
        st.dataframe(
            filtered_df[display_columns].sort_values('risk_score', ascending=False),
            use_container_width=True
        )
        
        # Intervention recommendations
        st.subheader("💡 Recommended Interventions")
        
        for idx, row in filtered_df.head(5).iterrows():
            with st.expander(f"Student (Age: {row['age']}, Grade: {row['G3']}) - {row['risk_level']}"):
                st.write(f"**Risk Score:** {row['risk_score']}/100")
                st.write("**Recommendations:**")
                
                if row['G3'] < 10:
                    st.write("- 🔴 **Urgent**: Failing grade - immediate tutoring required")
                if row['failures'] > 0:
                    st.write("- 📚 Academic support program enrollment")
                if row['absences'] > 10:
                    st.write("- ⏰ Attendance monitoring and parent engagement")
                if row['studytime'] < 2:
                    st.write("- 📖 Study skills workshop")
                
                st.write("- 👨‍👩‍👧 Schedule parent-teacher conference")
                st.write("- 🎯 Create personalized learning plan")

    # ====================
    # MODEL PERFORMANCE
    # ====================
    elif st.session_state.page == 'Model':
        st.markdown('<div class="main-header">🔬 Model Performance Analysis</div>', unsafe_allow_html=True)
        
        try:
            # Load model results
            regression_results = pd.read_csv('regression_model_results.csv')
            classification_results = pd.read_csv('classification_model_results.csv')
            
            tab1, tab2, tab3 = st.tabs(["📊 Regression Models", "🎯 Classification Models", "📈 Visualizations"])
            
            # Regression Models Tab
            with tab1:
                st.subheader("Regression Model Comparison")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display results table
                    st.dataframe(
                        regression_results.style.highlight_max(subset=['Test R²'], color='lightgreen')
                                               .highlight_min(subset=['Test RMSE', 'Test MAE'], color='lightgreen'),
                        use_container_width=True
                    )
                
                with col2:
                    # Best model metrics
                    best_model = regression_results.iloc[0]
                    st.metric("🏆 Best Model", best_model['Model'])
                    st.metric("R² Score", f"{best_model['Test R²']:.4f}")
                    st.metric("RMSE", f"{best_model['Test RMSE']:.4f}")
                    st.metric("MAE", f"{best_model['Test MAE']:.4f}")
                
                # Model comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        regression_results.sort_values('Test R²', ascending=True),
                        x='Test R²', y='Model',
                        title='Models by R² Score',
                        orientation='h',
                        color='Test R²',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        regression_results.sort_values('Test RMSE'),
                        x='Test RMSE', y='Model',
                        title='Models by RMSE (Lower is Better)',
                        orientation='h',
                        color='Test RMSE',
                        color_continuous_scale='Reds_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Classification Models Tab
            with tab2:
                st.subheader("Classification Model Comparison")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(
                        classification_results.style.highlight_max(
                            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            color='lightgreen'
                        ),
                        use_container_width=True
                    )
                
                with col2:
                    best_clf = classification_results.iloc[0]
                    st.metric("🏆 Best Classifier", best_clf['Model'])
                    st.metric("Accuracy", f"{best_clf['Accuracy']:.4f}")
                    st.metric("F1-Score", f"{best_clf['F1-Score']:.4f}")
                
                # Metrics comparison
                fig = go.Figure()
                
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=classification_results['Model'],
                        y=classification_results[metric]
                    ))
                
                fig.update_layout(
                    title='Classification Metrics Comparison',
                    barmode='group',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualizations Tab
            with tab3:
                st.subheader("Model Performance Visualizations")
                
                if model is not None:
                    # Make predictions
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    
                    # Prepare data
                    X = df_processed.drop(['G3'], axis=1)
                    y = df_processed['G3']
                    
                    # Encode and process
                    df_viz = df_processed.copy()
                    for col in label_encoders.keys():
                        if col in df_viz.columns:
                            df_viz[col] = label_encoders[col].transform(df[col])
                    
                    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
                    for col in binary_cols:
                        if col in df_viz.columns:
                            df_viz[col] = df[col].map({'yes': 1, 'no': 0})
                    
                    # Feature engineering
                    df_viz['parent_edu_avg'] = (df_viz['Medu'] + df_viz['Fedu']) / 2
                    df_viz['parent_edu_max'] = df_viz[['Medu', 'Fedu']].max(axis=1)
                    df_viz['alc_avg'] = (df_viz['Dalc'] + df_viz['Walc']) / 2
                    df_viz['social_score'] = (df_viz['goout'] + df_viz['freetime'] + df_viz['romantic']) / 3
                    df_viz['support_score'] = (df_viz['schoolsup'] + df_viz['famsup'] + df_viz['paid']) / 3
                    df_viz['grade_improvement'] = df_viz['G2'] - df_viz['G1']
                    df_viz['at_risk'] = ((df_viz['failures'] > 0) | (df_viz['absences'] > 10) | (df_viz['G2'] < 10)).astype(int)
                    df_viz['study_efficiency'] = df_viz['G2'] / (df_viz['studytime'] + 1)
                    
                    X_viz = df_viz.drop(['G3'], axis=1)
                    y_viz = df_viz['G3']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_viz, y_viz, test_size=0.3, random_state=42)
                    
                    scaler_viz = StandardScaler()
                    X_test_scaled = scaler_viz.fit_transform(X_test[feature_names])
                    
                    y_pred = model.predict(X_test_scaled)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Actual vs Predicted
                        fig = px.scatter(
                            x=y_test, y=y_pred,
                            title='Actual vs Predicted Grades',
                            labels={'x': 'Actual Grade', 'y': 'Predicted Grade'},
                            opacity=0.6
                        )
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Residuals
                        residuals = y_test - y_pred
                        fig = px.scatter(
                            x=y_pred, y=residuals,
                            title='Residual Plot',
                            labels={'x': 'Predicted Grade', 'y': 'Residuals'},
                            opacity=0.6
                        )
                        fig.add_hline(y=0, line_dash='dash', line_color='red')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Error distribution
                    fig = px.histogram(
                        residuals,
                        nbins=30,
                        title='Prediction Error Distribution',
                        labels={'value': 'Prediction Error', 'count': 'Frequency'}
                    )
                    fig.add_vline(x=residuals.mean(), line_dash='dash', line_color='red',
                                 annotation_text=f'Mean: {residuals.mean():.2f}')
                    st.plotly_chart(fig, use_container_width=True)
        
        except FileNotFoundError:
            st.error("⚠️ Model results files not found. Please run the training notebook first.")

    # ====================
    # STUDENT COMPARISON
    # ====================
    elif st.session_state.page == 'Comparison':
        st.markdown('<div class="main-header">👥 Student Comparison Tool</div>', unsafe_allow_html=True)
        
        st.info("Compare up to 3 students side-by-side to analyze performance differences.")
        
        # Sample students for comparison
        col1, col2, col3 = st.columns(3)
        
        students = []
        
        with col1:
            st.subheader("Student 1")
            if st.checkbox("Use random student 1", key='random1'):
                student1 = df.sample(1).iloc[0]
                st.write(f"**Age:** {student1['age']}")
                st.write(f"**Final Grade:** {student1['G3']}")
                students.append(student1)
        
        with col2:
            st.subheader("Student 2")
            if st.checkbox("Use random student 2", key='random2'):
                student2 = df.sample(1).iloc[0]
                st.write(f"**Age:** {student2['age']}")
                st.write(f"**Final Grade:** {student2['G3']}")
                students.append(student2)
        
        with col3:
            st.subheader("Student 3")
            if st.checkbox("Use random student 3", key='random3'):
                student3 = df.sample(1).iloc[0]
                st.write(f"**Age:** {student3['age']}")
                st.write(f"**Final Grade:** {student3['G3']}")
                students.append(student3)
        
        if len(students) >= 2:
            # Radar chart comparison
            st.subheader("📊 Multi-Feature Comparison")
            
            features = ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']
            
            fig = go.Figure()
            
            for i, student in enumerate(students):
                fig.add_trace(go.Scatterpolar(
                    r=[student[f] for f in features],
                    theta=features,
                    fill='toself',
                    name=f'Student {i+1}'
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
                showlegend=True,
                title='Student Performance Radar Chart',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.subheader("📋 Detailed Comparison")
            
            comparison_data = {
                'Metric': ['Age', 'Study Time', 'Failures', 'Absences', 'Period 1', 'Period 2', 'Final Grade']
            }
            
            for i, student in enumerate(students):
                comparison_data[f'Student {i+1}'] = [
                    student['age'],
                    student['studytime'],
                    student['failures'],
                    student['absences'],
                    student['G1'],
                    student['G2'],
                    student['G3']
                ]
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

    # ====================
    # REPORTS GENERATOR
    # ====================
    elif st.session_state.page == 'Reports':
        st.markdown('<div class="main-header">📄 Reports Generator</div>', unsafe_allow_html=True)
        
        st.subheader("Generate Comprehensive Reports")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Individual Student Report", "Class Performance Report", "At-Risk Students Report"]
        )
        
        if report_type == "Individual Student Report":
            st.write("### Individual Student Report")
            
            # Select random student or use filters
            if st.button("Generate Sample Student Report"):
                student = df.sample(1).iloc[0]
                
                # Create report
                st.markdown("---")
                st.markdown(f"## Student Performance Report")
                st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Student Information")
                    st.write(f"**School:** {student['school']}")
                    st.write(f"**Gender:** {'Female' if student['sex'] == 'F' else 'Male'}")
                    st.write(f"**Age:** {student['age']}")
                    st.write(f"**Address:** {'Urban' if student['address'] == 'U' else 'Rural'}")
                
                with col2:
                    st.markdown("### Academic Performance")
                    st.metric("Period 1 Grade", student['G1'])
                    st.metric("Period 2 Grade", student['G2'])
                    st.metric("Final Grade", student['G3'])
                
                # Performance chart
                grades = [student['G1'], student['G2'], student['G3']]
                fig = go.Figure(data=[
                    go.Bar(x=['Period 1', 'Period 2', 'Final'], y=grades,
                          marker_color=['#3498db', '#2ecc71', '#e74c3c'])
                ])
                fig.update_layout(title='Grade Progression', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Study habits
                st.markdown("### Study Habits & Behavior")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Study Time", ['<2h', '2-5h', '5-10h', '>10h'][student['studytime']-1])
                with col2:
                    st.metric("Absences", student['absences'])
                with col3:
                    st.metric("Past Failures", student['failures'])
                
                # Download report
                st.markdown("---")
                st.download_button(
                    "📥 Download Report (CSV)",
                    data=pd.DataFrame([student]).to_csv(index=False),
                    file_name=f"student_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        elif report_type == "Class Performance Report":
            st.write("### Class Performance Summary")
            
            if st.button("Generate Class Report"):
                # Overall statistics
                st.markdown("## Class Performance Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Students", len(df))
                with col2:
                    st.metric("Average Grade", f"{df['G3'].mean():.2f}")
                with col3:
                    st.metric("Pass Rate", f"{(df['G3'] >= 10).mean()*100:.1f}%")
                with col4:
                    st.metric("At-Risk", f"{((df['G3'] < 10) | (df['failures'] > 0)).sum()}")
                
                # Grade distribution
                fig = px.histogram(df, x='G3', nbins=20, title='Grade Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download class report
                class_summary = df.describe().T
                st.download_button(
                    "📥 Download Class Report",
                    data=class_summary.to_csv(),
                    file_name=f"class_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:  # At-Risk Students Report
            st.write("### At-Risk Students Report")
            
            if st.button("Generate At-Risk Report"):
                at_risk = df[
                    (df['G3'] < 10) | 
                    (df['failures'] > 0) | 
                    (df['absences'] > 10)
                ]
                
                st.markdown(f"## At-Risk Students Report")
                st.markdown(f"**Total At-Risk Students:** {len(at_risk)}")
                
                # Display at-risk students
                st.dataframe(
                    at_risk[['school', 'sex', 'age', 'G3', 'failures', 'absences', 'studytime']],
                    use_container_width=True
                )
                
                # Download at-risk report
                st.download_button(
                    "📥 Download At-Risk Report",
                    data=at_risk.to_csv(index=False),
                    file_name=f"at_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

else:
    st.error("⚠️ Unable to load data. Please check if 'student-por.csv' exists in the project directory.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎓 Student Performance Prediction System | Built with Streamlit</p>
        <p>© 2024 - Educational Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)
