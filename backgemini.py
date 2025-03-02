from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow frontend access
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Allow all origins for /chat endpoint

# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_DEFAULT_API_KEY"))

# Load sentence transformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chat history
chat_history = []

# Set PDF path (relative path for deployment)
PDF_PATH = os.path.join(os.getcwd(), "temp.pdf")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to create FAISS vector store
def create_vector_store(text_chunks):
    try:
        embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index, embeddings, text_chunks
    except Exception as e:
        return None, None, None

# Load and process the PDF
text = extract_text_from_pdf(PDF_PATH)
text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
index, embeddings, chunks = create_vector_store(text_chunks)

# Initialize Gemini model (Loaded once for efficiency)
gemini_model = genai.GenerativeModel("gemini-2.0-pro")

# Chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # Embed the query
        query_embedding = embed_model.encode([user_query], convert_to_numpy=True)

        # Retrieve relevant context
        D, I = index.search(query_embedding, k=3)
        retrieved_text = " ".join([chunks[i] for i in I[0]])

        # Create a prompt
        prompt = f"Context: {retrieved_text}\nUser: {user_query}\nAI:"

        # Generate response using Gemini AI
        response = gemini_model.generate_content(prompt)
        chat_response = response.text if response.text else "Sorry, I couldn't generate a response."

        # Store chat history
        chat_history.append({"user": user_query, "ai": chat_response})

        return jsonify({"response": chat_response, "chat_history": chat_history})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to get chat history
@app.route("/history", methods=["GET"])
def get_chat_history():
    return jsonify({"chat_history": chat_history})

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000
    app.run(host="0.0.0.0", port=port, debug=True)
