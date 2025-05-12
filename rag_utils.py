# rag_utils.py

import faiss
import numpy as np
import textwrap
import requests
from llama_index.embeddings.gemini import GeminiEmbedding

API_KEY = "AIzaSyCTlKkq1eYW5lHoss8i6xEpgjmvg8ETHOc"  # Move to environment variable in production

def initialize_embedding_model():
    return GeminiEmbedding(model_name="models/embedding-001", api_key=API_KEY, title="Resume RAG")

def get_embeddings(model, texts):
    return model.get_text_embedding_batch(texts)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_similar_texts(model, index, query, texts, top_k=1):
    q_embed = model.get_text_embedding_batch([query])
    q_embed = np.array(q_embed)
    _, indices = index.search(q_embed, top_k)
    return [texts[i] for i in indices[0]]

def split_document(doc, max_size=9000):
    return textwrap.wrap(doc, max_size)

def generate_response(retrieved_docs, query):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}'

    if isinstance(retrieved_docs, list):
        retrieved_docs = "\n".join(retrieved_docs)

    doc_chunks = split_document(retrieved_docs)
    response_text = ""

    for chunk in doc_chunks:
        prompt = f"""
        Documents: {chunk}
        Query: {query}
        Instructions:
        - Only use information from the documents.
        - If info is insufficient, say so clearly.

        Response:"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            response_text += content + "\n"
        else:
            response_text += f"[Error {response.status_code}] {response.text}\n"

    return response_text
