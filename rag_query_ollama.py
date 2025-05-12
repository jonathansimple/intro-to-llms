import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import requests

# ---------- CONFIG ----------
API_KEY = "fake-key"
LM_STUDIO_SERVER_URL = "http://localhost:1234/v1" # for LM Studio
MODEL_NAME = "gemma-3-1b-it-qat" # LM Studio uses a different name for the same model

LLAMA_SERVER_URL = "http://localhost:11434/v1" # for ollama
OLLAMA_MODEL_NAME = "gemma3:1b-it-qat" # the model identifier for the ollama model you are using

# Function to generate an answer using LM Studio's model
def generate_with_lm_studio(query, context):
    # prompt initialization
    conversations = [
        {"role": "system", "content": 'You are a helpful AI assistant named Bob capable of RAG (retrieval augmented generation). Use the provided context to accurately address the user\'s query. You may be provided with extra, unnecessary information. Only use relevant context when augmenting your answers.'},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer:"},
    ]
    
    client = openai.OpenAI(base_url=LM_STUDIO_SERVER_URL, api_key=API_KEY)

    # Stream the LLM output on the terminal
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversations,
        max_tokens=512,
        temperature=0.0,
        stream=True
    )
    
    llm_answer = ""
    print("Response:")
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            llm_answer += chunk.choices[0].delta.content
    print("\n")
    
    return llm_answer


def rag_system(query, documents, index, embedder):
    # Step 1: Perform retrieval (use FAISS for example)
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=2)
    retrieved_docs = [documents[i] for i in I[0]]

    # Step 2: Generate answer with LM Studio
    context = " ".join(retrieved_docs)
    answer = generate_with_lm_studio(query, context)
    return answer


# Example documents
documents = [
    "Ollama is way faster than transformers.",
    "Donald Trump won the 2024 presidential election and remains president in 2025.",
    "Most new models can perform function calling out-of-the-box."
]

# Embed documents using SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Creating vector store...")
document_embeddings = embedder.encode(documents)

# Create a FAISS index
dimension = document_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings, dtype=np.float32))

# Example usage
query = "Who is the president of the United States in 2025?"
answer = rag_system(query, documents, index, embedder)
