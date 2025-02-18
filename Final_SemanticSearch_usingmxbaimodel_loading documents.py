import os
import torch
import faiss
import numpy as np
import pandas as pd
import fitz  # PyMuPDF for install [use pip install --upgrade pymupdf]
import pdfplumber
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the embedding model
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token as embedding

# Function to read a document (TXT, CSV, PDF)
def read_document(file_path):
    ext = file_path.split('.')[-1].lower()

    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.readlines()
    
    elif ext == "csv":
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].tolist()  # Assume first column contains text
    
    elif ext == "pdf":
        return read_pdf(file_path)  # Use our custom PDF reader

    else:
        raise ValueError("Unsupported file format. Use TXT, CSV, or PDF.")

# Function to read a PDF using PyMuPDF and pdfplumber
def read_pdf(file_path):
    text_data = []

    # First, try extracting text with PyMuPDF
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text("text")
        if text.strip():  # If PyMuPDF extracts text, use it
            text_data.append(text)
        else:  # If PyMuPDF fails, use pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text_plumber = pdf.pages[page.number].extract_text()
                if text_plumber:
                    text_data.append(text_plumber)

    return text_data

# Load document and extract text
file_path = "Covid.pdf"  # Change this to your file
documents = read_document(file_path)

# Create embeddings for document
document_embeddings = np.vstack([get_embedding(text) for text in documents])

# Build FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Function to search the document
def search_similar(query, top_k=1):
    query_embedding = get_embedding(query)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(documents[idx])

    return results

# Test search
query = input("Enter your Query?")
results = search_similar(query)

print("\n🔍 Query:", query)
print("🔎 Top Matches:", results)
