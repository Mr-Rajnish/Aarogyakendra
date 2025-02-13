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

# Custom CSS for black, white, and neon green color palette
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

# Sidebar for navigation
with st.sidebar:
    st.title("AarogyaGarbha-          Made by Rajnish Singh")
    st.write("Welcome to AarogyaGarbha")
    st.write("Choose an option from the menu below to get started:")

    selected = option_menu(
        'AarogyaGarbha',
        ['About Us', 'Pregnancy Risk Prediction', 'Fetal Health Prediction', 'Dashboard'],
        icons=['info-circle', 'heart-pulse', 'heart', 'bar-chart'],  # Added icon for Fetal Health Prediction
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "black", "border": "2px solid #39FF14"},
            "icon": {"color": "#39FF14", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
            "nav-link-selected": {"background-color": "#39FF14", "color": "black"},
        }
    )

# About Us Page
if selected == 'About Us':
    st.title("Welcome to AarogyaGarbha- Made by Rajnish Singh")
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h3 style='color: white;'>At AarogyaGarbha, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate predictions and proactive risk management.</h3>
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

    st.write("Thank you for choosing AarogyaGarbha. We are committed to advancing healthcare through technology and predictive analytics. Feel free to explore our features and take advantage of the insights we provide.")

# Pregnancy Risk Prediction Page
if selected == 'Pregnancy Risk Prediction':
    st.title('Pregnancy Risk Prediction')
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h3 style='color: white;'>Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)

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
            <h3 style='color: white;'>Cardiotocograms (CTGs) are a simple and cost-accessible option to assess fetal health, allowing healthcare professionals to take action to prevent child and maternal mortality.</h3>
        </div>
        </br>
    """, unsafe_allow_html=True)

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

    # Initialize Dashboard
    dashboard = MaternalHealthDashboard(api_endpoint)

    # Bubble Chart
    st.markdown("""
        <div style='background-color: black; padding: 20px; border-radius: 10px; border: 2px solid #39FF14;'>
            <h3 style='color: #39FF14;'>Bubble Chart: Regional Performance</h3>
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
            <h3 style='color: #39FF14;'>Pie Chart: Regional Distribution</h3>
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