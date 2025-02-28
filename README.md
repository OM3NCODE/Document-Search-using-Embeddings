# Document Search using Embeddings

## Overview
This project, developed by **Adhish Bharadwaj** as a mini-project, is a **document search system** using **FAISS** and **transformer-based embeddings**. It allows users to search for relevant text within documents (**TXT, CSV, and PDF**) based on semantic similarity. While it is designed to handle all document formats, it currently utilizes the **20 Newsgroups dataset (first 650 rows) for testing purposes.**

## Features
- Reads and processes **TXT, CSV, and PDF** files.
- Uses **transformer-based embeddings** for document representation.
- Employs **FAISS (Facebook AI Similarity Search)** for efficient similarity search.
- Supports multiple queries and returns the **most relevant matches**.
- **Handles missing values (NaNs) and empty texts** efficiently.
- **Improved chunking mechanism** for better processing of large documents.
- **Uses the 20 Newsgroups dataset for testing (first 650 rows) but is designed to handle any document format.**

## Installation
Follow these steps to set up the project:

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <project-folder>
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**
Run the following command to install required libraries from `requirements.txt`:
```bash
pip install -r Requirements.txt
```

### **4. Ensure Additional Packages are Installed**
If `requirements.txt` is missing, install dependencies manually:
```bash
pip install torch faiss-cpu numpy pandas pymupdf pdfplumber transformers scikit-learn
```
**Note**: 
If pymupdf does not install, use the below command: 
```bash
pip install --upgrade pymupdf
```

## Usage
1. **Prepare your documents:** Place your test documents (`TXT`, `CSV`, or `PDF`) inside the project directory.
2. **Run the script:**
   ```bash
   python Final_SemanticSearch_usingmxbaimodel_loading_documents.py
   ```
3. **Enter your query** when prompted.
4. The script will return the **top matching sections** from the documents based on their semantic similarity.

## How It Works
1. **Loads a pre-trained embedding model from HuggingFace** (`mixedbread-ai/mxbai-embed-large-v1`).
2. **Processes documents and handles missing values**.
3. **Chunks large documents into 1000-character segments**.
4. **Extracts embeddings** from the document chunks.
5. **Indexes embeddings** using FAISS for efficient retrieval.
6. **Encodes the user's query** and searches for the closest matches.
7. **Displays the most relevant document sections** with high similarity scores.
8. **Uses the first 650 rows of the 20 Newsgroups dataset for testing while ensuring flexibility for all document formats.**

## File Structure
```
📂 Project Folder
├── 📄 Final_SemanticSearch_usingmxbaimodel_loading_documents.py  # Main script
├── 📄 Requirements.txt   # List of dependencies
├── 📄 README.md          # Project documentation
├── 📄 Bitcoin.txt        # Sample document
├── 📄 Covid.pdf          # Sample document
```

## Notes
- This project is optimized for **CPU usage**. If running on **GPU**, install `faiss-gpu` instead of `faiss-cpu` for better performance.
- **Improved indexing and query handling** for better accuracy.
- Tested with sample **TXT and PDF** documents available in the project directory.

## Credits
Project by **Adhish Bharadwaj** 🚀
