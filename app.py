import streamlit as st
import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬"
)

st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Enter a movie review and the model will predict whether it is Positive or Negative."
)

@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

review = st.text_area(
    "Enter Movie Review"
)

if st.button("Analyze"):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = (
        probs.max().item()
        * 100
    )

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    st.success(
        f"Prediction: {sentiment}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )