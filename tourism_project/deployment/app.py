
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Create a Streamlit app that loads the model committed to the repository
MODEL_PATH = Path(__file__).resolve().parent / "tourism_model.pkl"
model = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏨",
    layout="wide"
)

# Header section
st.markdown(
    """
    <style>
    .main-title {font-size: 36px; font-weight: bold; color: #2C3E50;}
    .sub-title {font-size: 18px; color: #7F8C8D;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🏨 Tourism Package Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fill in customer details to predict package purchase likelihood</p>', unsafe_allow_html=True)

# Tabs for better navigation
tab1, tab2 = st.tabs(["📋 Customer Details", "📊 Prediction Results"])

with tab1:
    st.header("Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 35)
        typeofcontact = st.radio("Type of Contact", ["Self Enquiry", "Company Invited"])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        durationofpitch = st.slider("Duration of Pitch (minutes)", 0, 60, 10)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.radio("Gender", ["Male", "Female"])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
        numberoffollowups = st.slider("Number of Followups", 0, 10, 3)
        productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

    with col2:
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
        maritalstatus = st.radio("Marital Status", ["Single", "Married", "Divorced"])
        numberoftrips = st.slider("Number of Trips", 0, 20, 2)
        passport = st.radio("Passport", [0, 1])
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        owncar = st.radio("Own Car", [0, 1])
        numberofchildrenvisiting = st.slider("Number of Children Visiting", 0, 5, 1)
        designation = st.selectbox("Designation", ["AVP", "VP", "Manager", "Senior Manager", "Executive"])
        monthlyincome = st.number_input("Monthly Income", min_value=0, value=25000)

    # Get the inputs and save them into a dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [typeofcontact],
        "CityTier": [citytier],
        "DurationOfPitch": [durationofpitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [numberofpersonvisiting],
        "NumberOfFollowups": [numberoffollowups],
        "ProductPitched": [productpitched],
        "PreferredPropertyStar": [preferredpropertystar],
        "MaritalStatus": [maritalstatus],
        "NumberOfTrips": [numberoftrips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitchsatisfactionscore],
        "OwnCar": [owncar],
        "NumberOfChildrenVisiting": [numberofchildrenvisiting],
        "Designation": [designation],
        "MonthlyIncome": [monthlyincome]
    })

with tab2:
    st.header("Prediction Results")
    if st.button("🔮 Predict Package Purchase"):
        expected_columns = model.feature_names_in_
        for column in expected_columns:
            if column not in input_data.columns:
                input_data[column] = 0
        input_data = input_data[expected_columns]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.success("🎉 Customer is likely to purchase the tourism package.")
        else:
            st.warning("Customer is unlikely to purchase the tourism package.")

        if probability is not None:
            st.metric("Purchase Probability", f"{probability:.2%}")
            st.progress(int(probability * 100))

        st.subheader("Customer Details")
        st.dataframe(input_data, use_container_width=True)
