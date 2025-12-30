import streamlit as st
import joblib
import re
import numpy as np

model = joblib.load("model/phishing_model.pkl")

def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('%'),
        url.count('/'),
        1 if re.search(r"\d", url) else 0
    ]

st.title("Phishing Website Detection System")
st.write("Machine Learning based phishing detection")

url = st.text_input("Enter Website URL")

if st.button("Check Website"):
    if not re.match(r"https?://", url):
        st.error("Invalid website address format")
    else:
        features = np.array(extract_features(url)).reshape(1, -1)
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.error("⚠️ Phishing Website Detected")
        else:
            st.success("✅ Legitimate Website")