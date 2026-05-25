import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
import pickle

# ---------- CONFIG ----------
API_KEY = "fake-key"
SERVER_URL = "http://210.61.209.139:45276/v1" # for ollama
MODEL_NAME = "Qwen/Qwen3-30B-A3B" # the model identifier for the vLLM model you are using

# Function to generate an answer using LM Studio's model
def inference(query, context):
    # prompt initialization
    conversations = [
        {"role": "system", "content": 'You are a helpful AI assistant named Bob capable of RAG (retrieval augmented generation).\
        Use the provided context to accurately address the user\'s query.\
        You may be provided with extra, unnecessary information. Only use relevant context when augmenting your answers.\
        If the context is irrelevant, use your own knowledge.'},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer:"},
    ]
    
    print("==> Query: " + query)

    client = openai.OpenAI(base_url=SERVER_URL, api_key=API_KEY)

    # Stream the LLM output on the terminal
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversations,
        max_tokens=8192,
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
    print("Retrieved context:", retrieved_docs)

    # Step 2: Generate answer with Ollama or Lemonade
    context = " ".join(retrieved_docs)
    answer = inference(query, context)
    return answer

if __name__ == '__main__':
    # Example documents
    documents = [
        "Ollama and vLLM are way faster than transformers.",
        "Donald Trump won the 2024 presidential election and remains president until today.",
        "Most new models can perform function calling out-of-the-box.",
        "MCP is a cleaner way to organize agents with long tool lists.",
        "AI agents have severe security flaws, so you should use them in a sandboxed environment.",
        "Keria renewed his contract with T1 and will remain on the team until 2029.",
        "You can receive programming certificates from AMD for free by participating in their AI Developer Program."
    ]

    # Embed documents using SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists("vector_store.index"):
        # Load FAISS index
        index = faiss.read_index("vector_store.index")

        # Load documents
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)

        print(documents)
    else:
        print("Creating vector store...")
        document_embeddings = embedder.encode(documents)

        # Create a FAISS index
        dimension = document_embeddings.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(document_embeddings, dtype=np.float32))

        faiss.write_index(index, "vector_store.index")
        with open("documents.pkl", "wb") as f:
            pickle.dump(documents, f)

    # Example usage
    query = "Who is the president of the United States in 2025?"
    answer = rag_system(query, documents, index, embedder)
    print("-"*100)

    while True:
        # Wait for user input
        query_str = input("How may I help you?\n")
        
        # Skip if query is empty
        if len(query_str) == 0:
            continue
    
        # Inference by sending an API request to vLLM server
        llm_answer = rag_system(query_str, documents, index, embedder)
        
        print("-"*100)
