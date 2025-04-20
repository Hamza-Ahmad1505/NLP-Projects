import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to check whether it's **Real** or **Fake**.")

# Input box
news_text = st.text_area("News Article Text", height=200)

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: {result}")
