import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, T5ForConditionalGeneration

# Load Pretrained Models
chatbot_model_name = "microsoft/DialoGPT-small"
summarization_model_name = "t5-small"
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_model_name)
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)

summarization_model = T5ForConditionalGeneration.from_pretrained(summarization_model_name)
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

# Chatbot Function
def chatbot_response(user_input):
    inputs = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors="pt")
    response_ids = chatbot_model.generate(inputs, max_length=100, pad_token_id=chatbot_tokenizer.eos_token_id)
    response = chatbot_tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Summarization Function
def summarize_text(text):
    inputs = summarization_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Sentiment Analysis Function
def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    return "Positive" if sentiment == 1 else "Negative"

# Streamlit Web App
st.title("LLM-Based AI Capstone Project")

st.sidebar.header("Choose a Feature")
option = st.sidebar.selectbox("Select an option:", ["Chatbot (Customer Service)", "Summarization (Research Papers)", "Sentiment Analysis (Social Media)"])

if option == "Chatbot (Customer Service)":
    st.header("AI Chatbot for Customer Service")
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = chatbot_response(user_input)
        st.write("🤖 Chatbot:", response)

elif option == "Summarization (Research Papers)":
    st.header("AI-Powered Research Paper Summarization")
    uploaded_file = st.file_uploader("Upload a Research Paper (Text File)", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.write("Original Text:")
        st.text(text[:500])  # Show preview
        summary = summarize_text(text)
        st.subheader("Summarized Text:")
        st.write(summary)

elif option == "Sentiment Analysis (Social Media)":
    st.header("AI Sentiment Analysis for Social Media")
    user_text = st.text_area("Enter a social media comment:")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(user_text)
        st.write("📝 Sentiment:", sentiment)

st.sidebar.info("This AI tool utilizes GPT-based chatbots, T5 for summarization, and BERT for sentiment analysis.")
