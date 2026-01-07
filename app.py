import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan label encoder
model = joblib.load('stress_level_predictor_model.pkl')
le = joblib.load('label_encoder.pkl')

# Styling
st.set_page_config(page_title="Stress Level Predictor", layout="centered")

st.title("Stress Level Predictor")
st.write("Prediksi tingkat stres berdasarkan kebiasaan digital dan tidurmu 📱😴")

# Form Input
with st.form("user_input_form"):
    st.subheader("Masukkan Data Harianmu:")
    screen_time = st.slider("Berapa jam kamu menatap layar per hari?", 0.0, 24.0, step=0.5)
    tiktok_time = st.slider("Berapa jam main TikTok?", 0.0, 24.0, step=0.5)
    sleep_duration = st.slider("Berapa jam tidurmu semalam?", 0.0, 12.0, step=0.5)

    submitted = st.form_submit_button("Prediksi Sekarang 🚀")

if submitted:
    input_data = np.array([[screen_time, tiktok_time, sleep_duration]])

    # Prediksi
    prediction = model.predict(input_data)
    predicted_label = le.inverse_transform(prediction)[0]

    st.success(f"Tingkat stres kamu diprediksi: **{predicted_label.upper()}** 😌")

    if predicted_label.lower() == "high":
        st.warning("⚠️ Coba kurangi screen time, perbanyak tidur, dan jaga mood kamu ya.")
    elif predicted_label.lower() == "medium":
        st.info("✨ Kamu sedang di titik tengah. Yuk lebih mindful dengan keseharianmu.")
    else:
        st.balloons()
        st.success("🎉 Kamu terlihat cukup stabil, pertahankan gaya hidup sehatmu!")

# Footer
st.markdown("---")
st.markdown("©2025 | All rights reserved")


