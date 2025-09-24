from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from typing import List, Dict, Any

# --- App Initialization ---
app = FastAPI(title="RAG API", version="1.0")

# --- Load the Vector Index and Create Retrievers ---
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# For BM25, we need the raw text. We can get it from the vector store's docstore.
docstore_dict = vector_store.docstore._dict
texts = [doc.page_content for doc in docstore_dict.values()]
bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 3

# Combine the two retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

# --- Define the Prompt Template ---
prompt_template = """
You are a helpful AI assistant. Answer the user's question based ONLY on the following context. If the answer is not in the context, say "I don't know based on the provided information."

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- Pydantic Model for Request ---
class QueryRequest(BaseModel):
    question: str

# --- API Endpoint ---
@app.post("/query")
def query_rag(request: QueryRequest):
    question = request.question

    # 1. Retrieve relevant chunks
    retrieved_docs = ensemble_retriever.invoke(question)

    # 2. Format the context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3. FOR DEMO: We'll just return the context and a placeholder.
    # In a full version, you would send {context} and {question} to an LLM like GPT-3.5 or Ollama.
    # For now, we'll return the context to prove it's working.

    # 4. Prepare source information
    sources = []
    for doc in retrieved_docs:
        source_info = {"content": doc.page_content[:150] + "..."} # First 150 chars
        if hasattr(doc, 'metadata'):
            source_info.update(doc.metadata)
        sources.append(source_info)

    return {
        "question": question,
        "retrieved_context": context, # This shows what chunks were found
        "llm_answer": "This is where the full LLM's answer would appear. For this demo, see the 'retrieved_context' to verify the search is working.", # Placeholder
        "source_documents": sources
    }

@app.get("/")
def read_root():
    return {"message": "RAG API is running. Send a POST request to /query with a JSON body containing your 'question'."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=True)