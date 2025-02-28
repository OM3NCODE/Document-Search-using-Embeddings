import os
import torch
import faiss
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the embedding model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(texts):
    # Ensure texts are strings and remove NaN values
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    texts = [text for text in texts if text.strip()]  # Remove empty strings

    if not texts:  # If the batch is empty after cleaning, return an empty array
        return np.array([])

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def read_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return pd.DataFrame(f.readlines(), columns=["text"]).dropna()
    elif ext == "csv":
        df = pd.read_csv(file_path)
        return df.iloc[:, [0]].dropna()  # Ensure only text column is considered
    elif ext == "pdf":
        return read_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Use TXT, CSV, or PDF.")

def read_pdf(file_path):
    text_data = []
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_data.append(text)
        else:
            with pdfplumber.open(file_path) as pdf:
                text_plumber = pdf.pages[page.number].extract_text()
                if text_plumber:
                    text_data.append(text_plumber)
    return pd.DataFrame(text_data, columns=["text"]).dropna()

# Load a limited subset of 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = pd.DataFrame(newsgroups.data[:650], columns=["text"]).dropna()

# Chunking to optimize processing
chunk_size = 1000
documents["chunks"] = documents["text"].apply(lambda doc: [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)])
document_chunks = documents.explode("chunks").drop(columns=["text"]).dropna().reset_index(drop=True)

# Drop NaN and empty text chunks before embedding
document_chunks = document_chunks.dropna(subset=["chunks"]).reset_index(drop=True)

start_time = time.time()

# Compute embeddings
print("🛠️ Computing embeddings for documents...")
batch_size = 64
embeddings = []
valid_indices = []

for i in range(0, len(document_chunks), batch_size):
    batch_texts = document_chunks["chunks"].iloc[i:i+batch_size].tolist()
    batch_embeddings = get_embedding(batch_texts)

    if batch_embeddings.size > 0:  # Only add non-empty embeddings
        embeddings.extend(batch_embeddings)
        valid_indices.extend(range(i, i + len(batch_embeddings)))  # Track valid indices

# Ensure embeddings align with valid rows
document_chunks = document_chunks.iloc[valid_indices].reset_index(drop=True)
document_chunks["embedding"] = embeddings

dimension = len(embeddings[0])
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(np.vstack(embeddings))

def search_similar(query, top_k=3):
    query_embedding = get_embedding([query])
    distances, indices = index.search(query_embedding, top_k)
    print("\n🔍 Query:", query)
    print("\n🔎 Top Matches:\n")
    for i, idx in enumerate(indices[0]):
        print(f"🔥 Match {i+1}:")
        print(document_chunks.iloc[idx]["chunks"])  # Retrieve text from DataFrame
        print("-" * 80)

end_time = time.time()
print(f"⏳ Execution Time: {end_time - start_time:.2f} seconds")

query = input("Enter your Query: ")
search_similar(query)