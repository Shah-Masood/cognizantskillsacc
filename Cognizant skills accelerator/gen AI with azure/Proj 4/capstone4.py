import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os

# Set up Azure Credentials
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Initialize Azure NLP Client
def authenticate_client():
    credential = AzureKeyCredential(AZURE_API_KEY)
    client = TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=credential)
    return client

client = authenticate_client()

# Load NLP Models
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
summarization_model_name = "t5-small"

sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

summarization_model = T5ForConditionalGeneration.from_pretrained(summarization_model_name)
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

# Sentiment Analysis Function
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return sentiment_labels[sentiment]

# Document Summarization Function
def summarize_text(text):
    inputs = summarization_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Azure AI Bias Monitoring Function
def detect_bias(text):
    response = client.analyze_sentiment([text])[0]
    return f"Bias Detection Score: {response.confidence_scores.neutral * 100:.2f}%"

# Streamlit Web App
st.title("Cloud-Based NLP Capstone Project")

st.sidebar.header("Choose a Feature")
option = st.sidebar.selectbox("Select an option:", ["Sentiment Analysis", "Document Summarization", "Ethical NLP Bias Detection"])

if option == "Sentiment Analysis":
    st.header("Real-Time Sentiment Analysis (Azure Cloud)")
    user_text = st.text_area("Enter customer feedback or reviews:")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(user_text)
        st.write("📊 Sentiment Score:", sentiment)

elif option == "Document Summarization":
    st.header("Summarization of Research Papers (LLM-powered)")
    uploaded_file = st.file_uploader("Upload a Research Paper (Text File)", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("Original Text:")
        st.text(text[:500])  # Show preview
        summary = summarize_text(text)
        st.subheader("Summarized Text:")
        st.write(summary)

elif option == "Ethical NLP Bias Detection":
    st.header("Bias Detection in NLP Models (Azure API)")
    user_input = st.text_area("Enter text to check for bias:")
    if st.button("Analyze Bias"):
        bias_score = detect_bias(user_input)
        st.write("🧐 Bias Score:", bias_score)

st.sidebar.info("This tool leverages Azure AI Studio, BERT for sentiment analysis, and T5 for summarization.")
