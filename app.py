import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

news_text = st.text_area("Enter news text:")

if st.button("Check"):
    if news_text:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        result = "🟥 Fake News" if prediction == 1 else "🟩 Real News"
        st.subheader(result)
    else:
        st.warning("Please enter some text")

