
# frontend/app.py
import streamlit as st
import requests

st.title("🧠 AI Medical Prescription Verifier")

prescription_text = st.text_area("Paste Prescription Text Here")

if st.button("Verify Prescription"):
    response = requests.post("http://localhost:8000/verify", json={"text": prescription_text})
    result = response.json()

    st.subheader("Extracted Drug Info")
    st.write(result["drugs"])

    st.subheader("Interaction Warnings")
    st.write(result["interactions"])

    st.subheader("Alternative Suggestions")
    st.write(result["alternatives"])