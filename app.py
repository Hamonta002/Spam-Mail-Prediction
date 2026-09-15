import streamlit as st
import joblib

# Load the trained machine learning pipeline
model = joblib.load("advanced_spam_classifier.joblib")

# Page configuration
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Spam Mail Detector")

st.write(
    "Enter an email message below and the machine learning model "
    "will classify it as Spam or Ham."
)

# Email input
user_input = st.text_area(
    "Email Content",
    height=250,
    placeholder="Paste your email message here..."
)

# Analyze button
if st.button("Analyze Email"):

    if user_input.strip():

        # The saved pipeline handles TF-IDF + model prediction
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.success("✅ Ham — This appears to be a normal email.")
        else:
            st.error("🚨 Spam — This email appears to be spam.")

    else:
        st.warning("⚠️ Please enter an email message first.")
        
