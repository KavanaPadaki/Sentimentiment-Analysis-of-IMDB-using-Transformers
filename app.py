import streamlit as st
from transformers import pipeline

# Load your fine-tuned model from local folder or Hugging Face Hub
MODEL_NAME = "KavanaPadaki/distilbert-imdb"  # or "./distilbert-imdb-finetuned"
classifier = pipeline("text-classification", model=MODEL_NAME)

# Streamlit UI
st.title("🎬 IMDb Sentiment Classifier")
st.write("Enter a movie review and see if it's positive or negative.")

# Text input
user_input = st.text_area("Movie Review", height=150)

if st.button("Classify"):
    if user_input.strip():
        result = classifier(user_input, truncation=True, max_length=512)[0]
        label = result['label']
        score = result['score']
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {score:.2%}")
    else:
        st.warning("Please enter a review before classifying.")
