from codebase.dashboard_graphs import MaternalHealthDashboard
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import numpy as np
from io import StringIO
import requests

# Load models
maternal_model = pickle.load(open("model/finalized_maternal_model.sav", 'rb'))
fetal_model = pickle.load(open("model/best_xgboost.pkl", 'rb'))
import joblib

heart_model = joblib.load("model/heart.pkl")  # Correct way to load

diabetes_model = pickle.load(open("model/diabetes.pkl", 'rb'))
kidney_model=pickle.load(open("model/kidney.pkl",'rb'))
# Add Breast Cancer Model Loading
breast_cancer_model = pickle.load(open("model/breastcancer.pkl", 'rb'))



NewBMI_Overweight = 0
NewBMI_Underweight = 0
NewBMI_Obesity_1 = 0
NewBMI_Obesity_2 = 0
NewBMI_Obesity_3 = 0
NewInsulinScore_Normal = 0
NewGlucose_Low = 0
NewGlucose_Normal = 0
NewGlucose_Overweight = 0
NewGlucose_Secret = 0

# Custom CSS for consistent black, white, and neon green color palette
st.markdown("""
    <style>
    /* General styling */
    body {
        color: white;
        background-color: black;
    }
    .stSidebar {
        background-color: black;  /* Black background */
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #39FF14;  /* Neon green border */
    }
    .stSidebar .stTitle {
        color: #39FF14;  /* Neon green text for the title */
        font-size: 24px;
        font-weight: bold;
    }
    .stSidebar .stMarkdown {
        color: white;  /* White text for the content */
        font-size: 16px;
    }
    .stSidebar .stButton>button {
        background-color: #39FF14;  /* Neon green button */
        color: black;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stSidebar .stButton>button:hover {
        background-color: #32CD32;  /* Slightly darker neon green on hover */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #39FF14;  /* Neon green headings */
    }
    .stMarkdown p {
        color: white;  /* White text for paragraphs */
        font-size: 16px;
        line-height: 1.6;
    }
    .stTextInput>div>div>input {
        background-color: black;
        color: white;
        border: 1px solid #39FF14;
        border-radius: 5px;
        padding: 10px;
    }
    .stColumn {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add Breast Cancer Model Loading
breast_cancer_model = pickle.load(open("model/breastcancer.pkl", 'rb'))


# Update Sidebar
with st.sidebar:
    st.title("Aarogyakendra - Made by Rajnish Singh")
    st.write("Welcome to Aarogya Kendra")
    st.write("Choose an option from the menu below to get started:")

    selected = option_menu(
        'AarogyaKendra',
        ['About AarogyaKendra', 'AarogyaGarbha', 'AarogyaNidhi'],
        icons=['house', 'info-circle', 'hospital'],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "black", "border": "2px solid #39FF14"},
            "icon": {"color": "#39FF14", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
            "nav-link-selected": {"background-color": "#39FF14", "color": "black"},
        }
    )

    if selected == 'AarogyaGarbha':
        selected = option_menu(
            'AarogyaGarbha',
            ['About AarogyaGarbha', 'Pregnancy Risk Prediction', 'Fetal Health Prediction', 'Dashboard'],
            icons=['info-circle', 'heart-pulse', 'heart', 'bar-chart'],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "black", "border": "2px solid #39FF14"},
                "icon": {"color": "#39FF14", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#39FF14", "color": "black"},
            }
        )

    if selected == 'AarogyaNidhi':
        selected = option_menu(
            'AarogyaNidhi',
            ['About AarogyaNidhi', 'Diabetes Prediction', 'Heart Disease Prediction', 'Kidney Disease Prediction', 'Breast Cancer Prediction'],
            menu_icon='hospital-fill',
            icons=['info-circle', 'activity', 'heart-pulse', 'person', 'gender-female'],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "black", "border": "2px solid #39FF14"},
                "icon": {"color": "#39FF14", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#39FF14", "color": "black"},
            }
        )

# Landing Page: About AarogyaKendra
if selected == 'About AarogyaKendra':
    st.title("Welcome to Aarogyakendra - Made by Rajnish Singh")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>At Aarogyakendra, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. Our platform is designed to address the intricate aspects of maternal, fetal, heart, kidney, diabetes, and breast cancer health, providing accurate predictions and proactive risk management.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)

    st.header("AarogyaGarbha")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <p style='color: white;'>AarogyaGarbha focuses on maternal and fetal health, offering predictive tools to ensure the well-being of both mother and child.</p>
        </div>
        </br>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Pregnancy Risk Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict potential risks during pregnancy by analyzing parameters like age, blood sugar levels, and blood pressure.</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_container_width=True)

    with col2:
        st.subheader("2. Fetal Health Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Assess the health status of the fetus using advanced algorithms and comprehensive data analysis.</p>
            </div>
                    <br>
        """, unsafe_allow_html=True)
        st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_container_width=True)

    with col3:
        st.subheader("3. Dashboard Statistics")

        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Monitor and manage health data with our interactive dashboard, designed for ease of use and accessibility.</p>
                 <br>
            </div>
             <br>
        """, unsafe_allow_html=True)
        st.image("graphics/dashboardimage.png", caption="Dashboard", use_container_width=True)

    st.header("AarogyaNidhi")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <p style='color: white;'>AarogyaNidhi focuses on predicting and managing chronic diseases like heart disease, kidney disease, diabetes, and breast cancer.</p>
        </div>
        </br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3,col4= st.columns(2)
    with col1:
        st.subheader("1. Heart Disease Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict the risk of heart disease using parameters like age, blood pressure, and cholesterol levels.</p>
            </div>
                <br>
        """, unsafe_allow_html=True)
        st.image("graphics/heartimage.webp", caption="Heart Disease Prediction", use_container_width=True)

    with col2:
        st.subheader("2. Kidney Disease Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Assess kidney health by analyzing factors like blood urea, serum creatinine, and more.</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("graphics/kidneyimage.jpg", caption="Kidney Disease Prediction", use_container_width=True)

    with col3:
        st.subheader("3. Diabetes Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict diabetes risk using parameters like glucose levels, BMI, and insulin levels.</p>
            </div>
                     <br>
        """, unsafe_allow_html=True)
        st.image("graphics/diabetesimage.jpg", caption="Diabetes Prediction", use_container_width=True)

    with col4:
        st.subheader("4. Breast Cancer Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict breast cancer risk using parameters like mean radius, mean texture, and mean perimeter.</p>
            </div>
                    <br>
        """, unsafe_allow_html=True)
        st.image("graphics/breastcancer.webp", caption="Breast Cancer Prediction", use_container_width=True)

    st.write("Thank you for choosing Aarogyakendra. We are committed to advancing healthcare through technology and predictive analytics. Feel free to explore our features and take advantage of the insights we provide.")

# Rest of the code remains the same...

if selected == 'About AarogyaGarbha':
    st.title("Welcome to AarogyaGarbha - Made by Rajnish Singh")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h3 style='color: white;'>At Aarogyakendra, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate predictions and proactive risk management.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Pregnancy Risk Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of potential risks during pregnancy.</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_container_width=True)

    with col2:
        st.header("2. Fetal Health Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, and genetic factors, we deliver insights into the well-being of the unborn child.</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_container_width=True)

    st.header("3. Dashboard")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <p style='color: white;'>Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard is designed for ease of use and accessibility.</p>
        </div>
    """, unsafe_allow_html=True)
    st.image("graphics/dashboardimage.png", caption="Dashboard", use_container_width=True)

    st.write("Thank you for choosing AarogyaGarbha. We are committed to advancing healthcare through technology and predictive analytics. Feel free to explore our features and take advantage of the insights we provide.")


# Pregnancy Risk Prediction Page
if selected == 'Pregnancy Risk Prediction':
    st.title('Pregnancy Risk Prediction')
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age of the Person', key="age")
    with col2:
        diastolicBP = st.text_input('Diastolic BP in mmHg')
    with col3:
        BS = st.text_input('Blood glucose in mmol/L')
    with col1:
        bodyTemp = st.text_input('Body Temperature in Celsius')
    with col2:
        heartRate = st.text_input('Heart rate in beats per minute')

    if st.button('Predict Pregnancy Risk'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = pickle.load(open("model/scaler1.pkl", "rb"))
            user_input = np.array([int(age), int(diastolicBP), float(BS), float(bodyTemp), int(heartRate)]).reshape(1, -1)
            columns = ['Age', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
            user_df = pd.DataFrame(user_input, columns=columns)
            scaled_input = scaler.transform(user_df)
            predicted_risk = maternal_model.predict(scaled_input)

            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.success("Low Risk ✅")
            elif predicted_risk[0] == 1:
                st.warning("Medium Risk ⚠️")
            else:
                st.error("High Risk ❗ Please consult a doctor.")

    if st.button("Clear", key="clear_pregnancy"):
        st.rerun()

# Fetal Health Prediction Page
if selected == 'Fetal Health Prediction':
    st.title('Fetal Health Prediction')
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>Cardiotocograms (CTGs) are a simple and cost-accessible option to assess fetal health, allowing healthcare professionals to take action to prevent child and maternal mortality.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        BaselineValue = st.text_input('Baseline Value')
    with col2:
        Accelerations = st.text_input('Accelerations')
    with col3:
        fetal_movement = st.text_input('Fetal Movement')
    with col1:
        uterine_contractions = st.text_input('Uterine Contractions')
    with col2:
        light_decelerations = st.text_input('Light Decelerations')
    with col3:
        severe_decelerations = st.text_input('Severe Decelerations')
    with col1:
        prolongued_decelerations = st.text_input('Prolongued Decelerations')
    with col2:
        abnormal_short_term_variability = st.text_input('Abnormal Short Term Variability')
    with col3:
        mean_value_of_short_term_variability = st.text_input('Mean Value Of Short Term Variability')
    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.text_input('Percentage Of Time With ALTV')
    with col2:
        mean_value_of_long_term_variability = st.text_input('Mean Value Long Term Variability')
    with col3:
        histogram_width = st.text_input('Histogram Width')
    with col1:
        histogram_min = st.text_input('Histogram Min')
    with col2:
        histogram_max = st.text_input('Histogram Max')
    with col3:
        histogram_number_of_peaks = st.text_input('Histogram Number Of Peaks')
    with col1:
        histogram_number_of_zeroes = st.text_input('Histogram Number Of Zeroes')
    with col2:
        histogram_mode = st.text_input('Histogram Mode')
    with col3:
        histogram_mean = st.text_input('Histogram Mean')
    with col1:
        histogram_median = st.text_input('Histogram Median')
    with col2:
        histogram_variance = st.text_input('Histogram Variance')
    with col3:
        histogram_tendency = st.text_input('Histogram Tendency')

    if st.button('Predict Fetal Health'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = pickle.load(open("./model/scaler2.pkl", "rb"))
            user_input = np.array([
                float(BaselineValue), float(Accelerations), float(fetal_movement),
                float(uterine_contractions), float(light_decelerations), float(severe_decelerations),
                float(prolongued_decelerations), float(abnormal_short_term_variability),
                float(mean_value_of_short_term_variability),
                float(percentage_of_time_with_abnormal_long_term_variability),
                float(mean_value_of_long_term_variability), float(histogram_width),
                float(histogram_min), float(histogram_max), float(histogram_number_of_peaks),
                float(histogram_number_of_zeroes), float(histogram_mode), float(histogram_mean),
                float(histogram_median), float(histogram_variance), float(histogram_tendency)
            ]).reshape(1, -1)

            columns = [
                'baseline value', 'accelerations', 'fetal_movement',
                'uterine_contractions', 'light_decelerations', 'severe_decelerations',
                'prolongued_decelerations', 'abnormal_short_term_variability',
                'mean_value_of_short_term_variability',
                'percentage_of_time_with_abnormal_long_term_variability',
                'mean_value_of_long_term_variability', 'histogram_width',
                'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
                'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                'histogram_median', 'histogram_variance', 'histogram_tendency'
            ]

            user_df = pd.DataFrame(user_input, columns=columns)
            scaled_input = scaler.transform(user_df)
            predicted_risk = fetal_model.predict(scaled_input)

            st.subheader("Health Status:")
            if predicted_risk[0] == 1:
                st.success("✅ Result: **Normal**")
            elif predicted_risk[0] == 2:
                st.warning("⚠️ Result: **Suspect**")
            else:
                st.error("❗ Result: **Pathological** - Please consult a doctor.")

    if st.button("Clear", key="clear_fetal"):
        st.rerun()

if selected == "Dashboard":
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    
    # Dashboard Header
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h1 style='color: #39FF14;'>Dashboard</h1>
        </div>
        </br>
    """, unsafe_allow_html=True)

    # Dashboard Content
    content = """
        Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions. 
        The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. 
        It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives.
    """
    st.markdown(f"""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <p style='color: white;'>{content}</p>
        </div>
        </br>
    """, unsafe_allow_html=True)
    st.image("graphics/dashboardimage.png", caption="Dashboard", use_container_width=True)

    # Initialize Dashboard
    dashboard = MaternalHealthDashboard(api_endpoint)

    # Bubble Chart
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: #39FF14;'>Bubble Chart: Regional Performance</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    dashboard.create_bubble_chart()

    # Expander for Bubble Chart Data
    with st.expander("Show Bubble Chart Data", expanded=False):
        st.markdown("""
            <style>
            .stExpander {
                background-color: black;
                border: 2px solid #39FF14;
                border-radius: 10px;
            }
            .stExpander > div > div > div {
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        bubble_chart_data = dashboard.get_bubble_chart_data()
        st.markdown(f"<div style='color: white;'>{bubble_chart_data}</div>", unsafe_allow_html=True)

    # Pie Chart
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: #39FF14;'>Pie Chart: Regional Distribution</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    dashboard.create_pie_chart()

    # Expander for Pie Chart Data
    with st.expander("Show Pie Chart Data", expanded=False):
        st.markdown("""
            <style>
            .stExpander {
                background-color: black;
                border: 2px solid #39FF14;
                border-radius: 10px;
            }
            .stExpander > div > div > div {
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        pie_chart_data = dashboard.get_pie_graph_data()
        st.markdown(f"<div style='color: white;'>{pie_chart_data}</div>", unsafe_allow_html=True)

if selected == 'About AarogyaNidhi':
    st.title("Welcome to AarogyaNidhi - Made by Rajnish Singh")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>At AarogyaNidhi, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. Our platform is specifically designed to address the intricate aspects of heart, kidney, and diabetes and Breast health, providing accurate predictions and proactive risk management.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3,col4= st.columns(2)
    with col1:
        st.subheader("1. Heart Disease Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict the risk of heart disease using parameters like age, blood pressure, and cholesterol levels.</p>
            </div>
                <br>
        """, unsafe_allow_html=True)
        st.image("graphics/heartimage.webp", caption="Heart Disease Prediction", use_container_width=True)

    with col2:
        st.subheader("2. Kidney Disease Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Assess kidney health by analyzing factors like blood urea, serum creatinine, and more.</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("graphics/kidneyimage.jpg", caption="Kidney Disease Prediction", use_container_width=True)

    with col3:
        st.subheader("3. Diabetes Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict diabetes risk using parameters like glucose levels, BMI, and insulin levels.</p>
            </div>
                     <br>
        """, unsafe_allow_html=True)
        st.image("graphics/diabetesimage.jpg", caption="Diabetes Prediction", use_container_width=True)

    with col4:
        st.subheader("4. Breast Cancer Prediction")
        st.markdown("""
            <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
                <p style='color: white;'>Predict breast cancer risk using parameters like mean radius, mean texture, and mean perimeter.</p>
            </div>
             <br>
        """, unsafe_allow_html=True)
        st.image("graphics/breastcancer.webp", caption="Breast Cancer Prediction", use_container_width=True)


    st.write("Thank you for choosing AarogyaNidhi. We are committed to advancing healthcare through technology and predictive analytics. Feel free to explore our features and take advantage of the insights we provide.")

if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction Using Machine Learning")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>Predicting diabetes involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding diabetes.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    st.image("graphics/diabetesimage.jpg", caption="Diabetes Prediction", use_container_width=True)

    # Load the trained model and scaler
    with open("model/scalerdiabetes.pkl", "rb") as scaler_file:
        scaler3 = pickle.load(scaler_file)
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", "0")
    with col2:
        Glucose = st.text_input("Glucose Level", "0")
    with col3:
        BloodPressure = st.text_input("BloodPressure Value", "0")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value", "0")
    with col2:
        Insulin = st.text_input("Insulin Value", "0")
    with col3:
        BMI = st.text_input("BMI Value", "0")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value", "0")
    with col2:
        Age = st.text_input("Age", "0")



    if st.button("Diabetes Test Result"):
        try:
            # Convert inputs to float
            Pregnancies = float(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            SkinThickness = float(SkinThickness)
            Insulin = float(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = float(Age)

            # One-hot encoding for BMI categories
            NewBMI_Underweight = 1 if BMI <= 18.5 else 0
            NewBMI_Overweight = 1 if 24.9 < BMI <= 29.9 else 0
            NewBMI_Obesity_1 = 1 if 29.9 < BMI <= 34.9 else 0
            NewBMI_Obesity_2 = 1 if 34.9 < BMI <= 39.9 else 0
            NewBMI_Obesity_3 = 1 if BMI > 39.9 else 0

            # One-hot encoding for Insulin categories
            NewInsulinScore_Normal = 1 if 16 <= Insulin <= 166 else 0

            # One-hot encoding for Glucose categories
            NewGlucose_Low = 1 if Glucose <= 70 else 0
            NewGlucose_Normal = 1 if 70 < Glucose <= 99 else 0
            NewGlucose_Overweight = 1 if 99 < Glucose <= 126 else 0
            NewGlucose_Secret = 1 if Glucose > 126 else 0

            # Combine all features
            user_input = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                   BMI, DiabetesPedigreeFunction, Age, NewBMI_Underweight,
                                   NewBMI_Overweight, NewBMI_Obesity_1, NewBMI_Obesity_2, 
                                   NewBMI_Obesity_3, NewInsulinScore_Normal, NewGlucose_Low, 
                                   NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret]).reshape(1, -1)

            # Scale the input
            user_input_scaled = scaler3.transform(user_input)

            proba = diabetes_model.predict_proba(user_input_scaled)[0][1]  # Probability of diabetes
            threshold = 0.7  # Adjust if needed

            if proba >= threshold:
                 st.error(" 🔴 The person has diabetes. Consult a doctor!!")
            else:
                 st.success(" 🟢 The person does not have diabetes.")

        
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

    

if "selected" in locals() and selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction Using Machine Learning")
    st.markdown(
        """
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #FF5733;'>
            <h4 style='color: white;'>Heart disease prediction analyzes multiple health parameters such as age, blood pressure, cholesterol levels, and more to assess the risk of cardiovascular issues.</h4>
        </div>
        </br>
        """,
        unsafe_allow_html=True,
    )
    st.image("graphics/heartimage.webp", caption="Heart Disease Prediction", use_container_width=True)

    # Input Fields for Heart Disease Prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
        sex = st.text_input("Sex (0 = Female, 1 = Male)")
        cp = st.text_input("Chest Pain Type (0-3)")
        trestbps = st.text_input("Resting Blood Pressure")
        chol = st.text_input("Serum Cholesterol")
    with col2:
        fbs = st.text_input("Fasting Blood Sugar (0 = False, 1 = True)")
        restecg = st.text_input("Resting ECG Results (0-2)")
        thalach = st.text_input("Max Heart Rate Achieved")
        exang = st.text_input("Exercise-Induced Angina(0 =No, 1 =Yes)") 
    with col3:
        oldpeak = st.text_input("ST Depression Induced by Exercise")
        slope = st.text_input("Slope of Peak Exercise ST Segment (0-2)")
        ca = st.text_input("Number of Major Vessels (0-4)")
        thal = st.text_input("Thalassemia (0-3)")

    # Prediction Logic
    if st.button("Predict Heart Disease Risk"):
        try:
            # Prepare input data (No Scaling)
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]

            # Convert input data to numpy array and reshape
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Make prediction without scaling
            prediction = heart_model.predict(input_data_reshaped)[0]
            prediction = 1 - prediction  # Reverse prediction like Jupyter Notebook

            # Display result with correct labeling
            if prediction == 1:
                st.error("🔴 High Risk of Heart Disease! Consult a Doctor Immediately.")
            else:
                st.success("🟢 Low Risk of Heart Disease.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("Clear", key="clear_heart_disease"):
        st.rerun()


if selected == 'Kidney Disease Prediction':
    st.title("Kidney Disease Prediction using ML")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>Predicting kidney disease involves analyzing several parameters, including age, blood pressure, blood urea, serum creatinine, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding kidney disease.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)
    st.image("graphics/kidneyimage.jpg", caption="Kidney Disease Prediction", use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input('Age')

    with col2:
        blood_pressure = st.text_input('Blood Pressure')

    with col3:
        specific_gravity = st.text_input('Specific Gravity')

    with col4:
        albumin = st.text_input('Albumin')

    with col5:
        sugar = st.text_input('Sugar')

    with col1:
        red_blood_cells = st.text_input('Red Blood Cell')

    with col2:
        pus_cell = st.text_input('Pus Cell')

    with col3:
        pus_cell_clumps = st.text_input('Pus Cell Clumps')

    with col4:
        bacteria = st.text_input('Bacteria')

    with col5:
        blood_glucose_random = st.text_input('Blood Glucose Random')

    with col1:
        blood_urea = st.text_input('Blood Urea')

    with col2:
        serum_creatinine = st.text_input('Serum Creatinine')

    with col3:
        sodium = st.text_input('Sodium')

    with col4:
        potassium = st.text_input('Potassium')

    with col5:
        haemoglobin = st.text_input('Haemoglobin')

    with col1:
        packed_cell_volume = st.text_input('Packet Cell Volume')

    with col2:
        white_blood_cell_count = st.text_input('White Blood Cell Count')

    with col3:
        red_blood_cell_count = st.text_input('Red Blood Cell Count')

    with col4:
        hypertension = st.text_input('Hypertension')

    with col5:
        diabetes_mellitus = st.text_input('Diabetes Mellitus')

    with col1:
        coronary_artery_disease = st.text_input('Coronary Artery Disease')

    with col2:
        appetite = st.text_input('Appetitte')

    with col3:
        peda_edema = st.text_input('Peda Edema')
    with col4:
        aanemia = st.text_input('Aanemia')

   
    if st.button("Kidney's Test Result"):
        user_input = [age, blood_pressure, specific_gravity, albumin, sugar,
                    red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                    blood_glucose_random, blood_urea, serum_creatinine, sodium,
                    potassium, haemoglobin, packed_cell_volume,
                    white_blood_cell_count, red_blood_cell_count, hypertension,
                    diabetes_mellitus, coronary_artery_disease, appetite,
                    peda_edema, aanemia]

        # Convert input values to float
        user_input = [float(x) for x in user_input]
        
     

        # Make prediction
        prediction = kidney_model.predict([user_input])

        #reverse
        pred=1-prediction

        if pred[0] == 1:
           st.error("The person has Kidney's disease")
        else:
            st.success("The person does not have Kidney's disease")

    



# Add Breast Cancer Prediction Page
if "selected" in locals() and selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction Using Machine Learning")
    st.markdown(
        """
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h4 style='color: white;'>Predicting breast cancer involves analyzing several parameters, including mean radius, mean texture, mean perimeter, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding breast cancer.</h4>
        </div>
        </br>
        """,
        unsafe_allow_html=True,
    )
    st.image("graphics/breastcancer.webp", caption="Breast Cancer Prediction", use_container_width=True)

    # Input Fields for Breast Cancer Prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        mean_radius = st.text_input("Mean Radius")
        mean_texture = st.text_input("Mean Texture")
        mean_perimeter = st.text_input("Mean Perimeter")
        mean_area = st.text_input("Mean Area")
        mean_smoothness = st.text_input("Mean Smoothness")
        mean_compactness = st.text_input("Mean Compactness")
        mean_concavity = st.text_input("Mean Concavity")
        mean_concave_points = st.text_input("Mean Concave Points")
        mean_symmetry = st.text_input("Mean Symmetry")
        mean_fractal_dimension = st.text_input("Mean Fractal Dimension")
    with col2:
        radius_error = st.text_input("Radius Error")
        texture_error = st.text_input("Texture Error")
        perimeter_error = st.text_input("Perimeter Error")
        area_error = st.text_input("Area Error")
        smoothness_error = st.text_input("Smoothness Error")
        compactness_error = st.text_input("Compactness Error")
        concavity_error = st.text_input("Concavity Error")
        concave_points_error = st.text_input("Concave Points Error")
        symmetry_error = st.text_input("Symmetry Error")
        fractal_dimension_error = st.text_input("Fractal Dimension Error")
    with col3:
        worst_radius = st.text_input("Worst Radius")
        worst_texture = st.text_input("Worst Texture")
        worst_perimeter = st.text_input("Worst Perimeter")
        worst_area = st.text_input("Worst Area")
        worst_smoothness = st.text_input("Worst Smoothness")
        worst_compactness = st.text_input("Worst Compactness")
        worst_concavity = st.text_input("Worst Concavity")
        worst_concave_points = st.text_input("Worst Concave Points")
        worst_symmetry = st.text_input("Worst Symmetry")
        worst_fractal_dimension = st.text_input("Worst Fractal Dimension")

    # Prediction Logic
    if st.button("Predict Breast Cancer Risk"):
        try:
            # Prepare input data without scaling
            input_data = [
                float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area),
                float(mean_smoothness), float(mean_compactness), float(mean_concavity), float(mean_concave_points),
                float(mean_symmetry), float(mean_fractal_dimension), float(radius_error), float(texture_error),
                float(perimeter_error), float(area_error), float(smoothness_error), float(compactness_error),
                float(concavity_error), float(concave_points_error), float(symmetry_error), float(fractal_dimension_error),
                float(worst_radius), float(worst_texture), float(worst_perimeter), float(worst_area),
                float(worst_smoothness), float(worst_compactness), float(worst_concavity), float(worst_concave_points),
                float(worst_symmetry), float(worst_fractal_dimension)
            ]

            # Convert input data to numpy array and reshape
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Make prediction without scaling
            prediction = breast_cancer_model.predict(input_data_reshaped)

            # Display result
            if prediction[0] == 0:
                st.error("🔴 Breast Cancer Risk Detected! Consult a Doctor.")
            else:
                st.success("🟢 No Risk of Breast Cancer.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("Clear", key="clear_breast_cancer"):
        st.rerun()
