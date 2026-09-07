import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("decision_tree_co_model.pkl")


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌍 Air Quality Prediction System")

st.write(
    "This application uses a Machine Learning Decision Tree "
    "Regression model to predict CO(GT) concentration."
)

st.divider()


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Enter Air Quality Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    pt08_s1 = st.number_input(
        "PT08.S1(CO)",
        min_value=0.0,
        value=1000.0
    )

    nmhc = st.number_input(
        "NMHC(GT)",
        min_value=0.0,
        value=100.0
    )

    c6h6 = st.number_input(
        "C6H6(GT)",
        min_value=0.0,
        value=10.0
    )

    pt08_s2 = st.number_input(
        "PT08.S2(NMHC)",
        min_value=0.0,
        value=900.0
    )

with col2:
    nox = st.number_input(
        "NOx(GT)",
        min_value=0.0,
        value=200.0
    )

    pt08_s3 = st.number_input(
        "PT08.S3(NOx)",
        min_value=0.0,
        value=800.0
    )

    no2 = st.number_input(
        "NO2(GT)",
        min_value=0.0,
        value=100.0
    )

    pt08_s4 = st.number_input(
        "PT08.S4(NO2)",
        min_value=0.0,
        value=1000.0
    )

with col3:
    pt08_s5 = st.number_input(
        "PT08.S5(O3)",
        min_value=0.0,
        value=1000.0
    )

    temperature = st.number_input(
        "Temperature (T)",
        value=25.0
    )

    humidity = st.number_input(
        "Relative Humidity (RH)",
        min_value=0.0,
        value=50.0
    )

    absolute_humidity = st.number_input(
        "Absolute Humidity (AH)",
        min_value=0.0,
        value=1.0
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict CO(GT)", type="primary"):

    input_data = pd.DataFrame([[
        pt08_s1,
        nmhc,
        c6h6,
        pt08_s2,
        nox,
        pt08_s3,
        no2,
        pt08_s4,
        pt08_s5,
        temperature,
        humidity,
        absolute_humidity
    ]], columns=[
        "PT08.S1(CO)",
        "NMHC(GT)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH"
    ])

    prediction = model.predict(input_data)

    st.success("Prediction completed successfully!")

    st.metric(
        label="Predicted CO(GT)",
        value=f"{prediction[0]:.3f}"
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Model: Decision Tree Regression | "
    "Air Quality Prediction Project"
)
