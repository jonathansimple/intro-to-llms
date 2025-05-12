import requests
import openai


# ---------- CONFIG ----------
API_KEY = "sk-no-key-required" # Fake key (llama-server doesn’t validate keys)

# Change this to your Ollama/LM Studio's host url or use localhost if running on the same machine
LLAMA_SERVER_URL = "http://localhost:11434/v1" # for ollama
MODEL_NAME = "gemma3:1b-it-qat" # the model identifier for the model you are using

LM_STUDIO_SERVER_URL = "http://localhost:1234/v1" # for LM Studio
MODEL_NAME_LM = "gemma-3-1b-it-qat" # LM Studio uses a different name for the same model

def inference(query_str: str):
    
    # prompt initialization
    conversations = [
        {"role": "system", "content": 'You are a helpful AI assistant named Bob.'},
        {"role": "user", "content": query_str},
    ]
    
    client = openai.OpenAI(base_url=LLAMA_SERVER_URL, api_key=API_KEY)

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


if __name__ == '__main__':
    while True:
        # Wait for user input
        query_str = input("How may I help you?\n")
        
        # Skip if query is empty
        if len(query_str) == 0:
            continue
    
        # Inference by sending an API request to either the Ollama or LM Studio server
        llm_answer = inference(query_str)
        
        print("-"*100)
        
