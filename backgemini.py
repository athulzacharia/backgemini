from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow frontend access
import pymupdf as fitz  # Ensure correct import
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure Google Gemini AI
genai.configure(api_key="AIzaSyB4WZudVc6NS00M_y9z4SuK415mLlADbNw")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chat_history = []

# Path to the PDF file
PDF_PATH = "C:\\Users\\athul\\Downloads\\mini project\\chatbot under working\\temp.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings, text_chunks

# Load and process the PDF at startup
text = extract_text_from_pdf(PDF_PATH)
text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
index, embeddings, chunks = create_vector_store(text_chunks)

# Chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json["query"]
    
    # Embed the query
    query_embedding = embed_model.encode([user_query], convert_to_numpy=True)
    
    # Retrieve relevant context
    D, I = index.search(query_embedding, k=3)
    retrieved_text = " ".join([chunks[i] for i in I[0]])
    
    # Create a prompt
    prompt = f"Context: {retrieved_text}\nUser: {user_query}\nAI:"

    # Generate response using Gemini AI
    model = genai.GenerativeModel("gemini-2.0-pro-exp")
    response = model.generate_content(prompt)
    chat_response = response.text if response.text else "Sorry, I couldn't generate a response."

    # Store chat history
    chat_history.append({"user": user_query, "ai": chat_response})

    return jsonify({"response": chat_response, "chat_history": chat_history})

# API to get chat history
@app.route("/history", methods=["GET"])
def get_chat_history():
    return jsonify({"chat_history": chat_history})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000
    app.run(host="0.0.0.0", port=port, debug=True)

