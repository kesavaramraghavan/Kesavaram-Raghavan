from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Load the PDF
pdf_path = "./sample_data/TOOTH_FAIRY.pdf"  # Replace with your PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
)

# 4. Create the FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# 5. Save the index locally
vector_store.save_local("faiss_index")
print("FAISS index saved successfully to folder 'faiss_index'.")

# 6. Load the FAISS index safely
new_vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print("FAISS index loaded successfully!")
