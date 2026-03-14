import streamlit as st
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Wine Classifier",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #722F37;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5A242B;
        border-color: #5A242B;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR
with st.sidebar:
    st.title("🍷 About the App")
    st.info(
        """
        This ML app predicts the **class** of wine based on 13 chemical properties.

        The model distinguishes between:
        - **Class 0** – Barolo
        - **Class 1** – Grignolino
        - **Class 2** – Barbera
        """
    )
    st.write("---")
    st.caption("Built with Streamlit & Cloud Run")

# 4. MAIN INTERFACE
st.title("🍷 Wine Class Predictor")
st.markdown("Adjust the sliders below to input the wine measurements.")

col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.slider('Alcohol', 11.0, 15.0, 13.0)
    malic_acid = st.slider('Malic Acid', 0.5, 6.0, 1.7)
    ash = st.slider('Ash', 1.0, 4.0, 2.4)
    alcalinity = st.slider('Alcalinity of Ash', 10.0, 30.0, 15.0)
    magnesium = st.slider('Magnesium', 70.0, 165.0, 100.0)

with col2:
    total_phenols = st.slider('Total Phenols', 0.9, 4.0, 2.3)
    flavanoids = st.slider('Flavanoids', 0.3, 6.0, 2.0)
    nonflavanoid = st.slider('Nonflavanoid Phenols', 0.1, 0.7, 0.3)
    proanthocyanins = st.slider('Proanthocyanins', 0.4, 4.0, 1.5)

with col3:
    color_intensity = st.slider('Color Intensity', 1.0, 13.0, 5.0)
    hue = st.slider('Hue', 0.5, 1.8, 1.0)
    od280 = st.slider('OD280/OD315', 1.2, 4.0, 3.0)
    proline = st.slider('Proline', 270.0, 1700.0, 750.0)

st.write("---")

# 5. PREDICTION
if st.button('🔍 Predict Wine Class'):
    with st.spinner('Analyzing wine data...'):
        data = {
            "alcohol": alcohol,
            "malic_acid": malic_acid,
            "ash": ash,
            "alcalinity_of_ash": alcalinity,
            "magnesium": magnesium,
            "total_phenols": total_phenols,
            "flavanoids": flavanoids,
            "nonflavanoid_phenols": nonflavanoid,
            "proanthocyanins": proanthocyanins,
            "color_intensity": color_intensity,
            "hue": hue,
            "od280_od315": od280,
            "proline": proline
        }

        try:
            # Change this URL to your Cloud Run URL after deployment
            response = requests.post('http://127.0.0.1:8080/predict', json=data)

            if response.status_code == 200:
                prediction = response.json()['prediction']
                st.success("Prediction Complete!")
                st.header(f"🍷 {prediction}")
                st.markdown(f"""
                Based on the measurements provided:
                * **Alcohol:** {alcohol} | **Malic Acid:** {malic_acid}
                * **Flavanoids:** {flavanoids} | **Proline:** {proline}
                """)
                st.balloons()
            else:
                st.error(f'Server Error: {response.status_code}')

        except requests.exceptions.RequestException as e:
            st.error('Connection Error: Could not reach the prediction service.')
