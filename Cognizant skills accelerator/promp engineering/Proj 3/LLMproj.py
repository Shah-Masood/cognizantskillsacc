#Proj 1

# Step 1: Import Necessary Modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Step 2: Choose a Short Paragraph
text = """The quick brown fox jumps over the lazy dog. This classic sentence contains every letter of the alphabet, making it a favorite for typists and font designers."""

# Step 3: Tokenize the Text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Display Tokenization Results
print(f"Original Text: {text}\n")
print(f"Tokenized Output: {tokens}\n")
print(f"Number of Tokens: {len(tokens)}\n")

# Identify Words Split into Subwords
subword_tokens = [token for token in tokens if "##" in token]
print(f"Subword Tokens: {subword_tokens}\n")

# Step 4: Extract Embeddings Using BERT
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Extract Token Embeddings
embeddings = outputs.last_hidden_state.squeeze(0).numpy()

# Step 5: Reduce Dimensionality Using PCA
pca = PCA(n_components=2)
reduced_embeddings_pca = pca.fit_transform(embeddings)

# Step 6: Reduce Dimensionality Using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings_tsne = tsne.fit_transform(embeddings)

# Step 7: Visualize Token Embeddings
plt.figure(figsize=(12, 5))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], c='blue', alpha=0.6)
for i, token in enumerate(tokens):
    plt.annotate(token, (reduced_embeddings_pca[i, 0], reduced_embeddings_pca[i, 1]), fontsize=8)
plt.title("Token Embeddings Visualization (PCA)")

# t-SNE Plot
plt.subplot(1, 2, 2)
plt.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], c='red', alpha=0.6)
for i, token in enumerate(tokens):
    plt.annotate(token, (reduced_embeddings_tsne[i, 0], reduced_embeddings_tsne[i, 1]), fontsize=8)
plt.title("Token Embeddings Visualization (t-SNE)")

plt.show()

#Proj 2

import openai
# Define the API key 
openai.api_key = "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"

# Example text for summarization
article_text = """Artificial intelligence (AI) is transforming industries by automating complex tasks, improving decision-making, 
and optimizing efficiency. From healthcare to finance, AI-driven innovations are reshaping the way businesses operate. 
For example, machine learning models are being used to diagnose diseases, detect fraudulent transactions, and personalize customer experiences. 
As AI continues to evolve, ethical concerns such as bias, data privacy, and accountability must be addressed to ensure responsible use."""

# Define three different prompts for summarizing the article
generic_prompt = "Summarize the following article:\n" + article_text

detailed_prompt = """You are an AI assistant specializing in summarization. Your task is to summarize the following article in 3-4 sentences.
Focus on the main ideas and key takeaways. Avoid unnecessary details and maintain clarity.\n""" + article_text

specific_prompt = """You are an expert AI tasked with summarizing the following article in exactly three sentences.
Ensure that:
- The first sentence introduces the topic.
- The second sentence highlights real-world applications.
- The third sentence discusses ethical concerns.
Do not exceed three sentences.\n""" + article_text

# Function to generate response from LLM
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Generate responses for each prompt
generic_response = generate_response(generic_prompt)
detailed_response = generate_response(detailed_prompt)
specific_response = generate_response(specific_prompt)

# Print and compare outputs
print("\n--- Generic Prompt Response ---\n", generic_response)
print("\n--- Detailed Prompt Response ---\n", detailed_response)
print("\n--- Specific Prompt Response ---\n", specific_response)

# Analyze differences in clarity and usefulness
responses = {
    "Generic": generic_response,
    "Detailed": detailed_response,
    "Specific": specific_response
}

# Display responses in a structured format
import pandas as pd
df_responses = pd.DataFrame(responses.items(), columns=["Prompt Type", "Output"])
import ace_tools as tools
tools.display_dataframe_to_user(name="Prompt Response Comparison", dataframe=df_responses)

#Proj 3

import streamlit as st
import openai
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# Set OpenAI API Key
openai.api_key = "key"

# Sample FAQ Dataset (Question-Answer Pairs)
faq_data = {
    "What is AI?": "AI, or artificial intelligence, refers to computer systems that mimic human intelligence to perform tasks.",
    "How does machine learning work?": "Machine learning is a subset of AI where models learn from data to make predictions or decisions.",
    "What is deep learning?": "Deep learning is a type of machine learning that uses neural networks with many layers to process data.",
    "What is tokenization?": "Tokenization is the process of breaking text into smaller units, such as words or subwords, for NLP processing."
}

# Convert FAQ Data to List
faq_questions = list(faq_data.keys())
faq_answers = list(faq_data.values())

# Load BERT Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to Generate Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate Embeddings for FAQ Questions
faq_embeddings = np.array([get_embedding(q) for q in faq_questions])

# Initialize FAISS Index for Similarity Search
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings)

# Streamlit Web App
st.title("AI-Powered FAQ Assistant 🤖")
user_query = st.text_input("Ask a question:")

if user_query:
    query_embedding = get_embedding(user_query).reshape(1, -1)
    _, closest_match = index.search(query_embedding, 1)
    best_match_idx = closest_match[0][0]
    best_match_question = faq_questions[best_match_idx]
    best_match_answer = faq_answers[best_match_idx]

    # Generate LLM Response Based on Matched Answer
    prompt = f"You are a helpful AI assistant. Answer the following question based on existing knowledge: {best_match_question}\n\nAnswer: {best_match_answer}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("### AI Response:")
    st.write(response["choices"][0]["message"]["content"])
