import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import os


MODEL_NAME = "KavanaPadaki/distilbert-imdb"  

@st.cache_resource
def load_model(model_name):
    try:
        clf = pipeline("text-classification", model=model_name)
        return clf, None
    except Exception as e:
        return None, str(e)

classifier, load_error = load_model(MODEL_NAME)


st.title("🎬 IMDb Sentiment Classifier")
st.write("Enter a movie review and see if it's **Positive** or **Negative**.")

if load_error:
    st.error(f"Model could not be loaded: {load_error}")
else:
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

# Sidebar Information
st.sidebar.header("ℹ️ Model Info")
st.sidebar.write(f"Model: `{MODEL_NAME}`")
st.sidebar.write("Fine-tuned on IMDb dataset for sentiment analysis.")
st.sidebar.write("Built with 🤗 Transformers + Streamlit")
