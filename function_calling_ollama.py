import time
import os
import json
import requests
import torch
import openai

from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# ---------- CONFIG ----------
API_KEY = "sk-no-key-required" # Fake key (llama-server doesn’t validate keys)

# Change this to your Ollama/LM Studio's host url or use localhost if running on the same machine
LLAMA_SERVER_URL = "http://localhost:11434/v1" # for ollama
MODEL_NAME = "gemma2:2b" # the model identifier for the model you are using

system_prompt = r"""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_weather",
    "description": "Get the current temperature for the provided coordinates in Celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "The latitude of the location the user asked for."},
            "longitude": {"type": "number", "description": "The longitude of the location the user asked for."}
        },
        "required": ["latitude", "longitude"]
    }
}
If a function should be used, respond with the function name and its parameters in json format like so:
{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": "value_2", ...}}

User Query: """

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m")
    data = response.json()
    return data['current']['temperature_2m']
    
def parse_llm(llm_text):
    if r"```json" in llm_text:
        llm_text = r"[" + llm_text.split(r"```json")[1].split(r"```")[0] + r"]"
    elif r"```tool_code" in llm_text:
        llm_text = r"[" + llm_text.split(r"```tool_code")[1].split(r"```")[0] + r"]"
    elif r"<functioncall>" in llm_text:
        llm_text = r"[" + llm_text.split(r"<functioncall>")[1].split(r"</functioncall>")[0] + r"]"
    else:
        llm_text = r"[" + llm_text + r"]"

    print("==> Parsed result: ", llm_text)

    try:
        data = json.loads(llm_text)
        print("==> Parsed list:", data)

        return data
    except:
        print("Cannot be parsed")
        return None

def run_conversation(input_text):
    print("==> User Input:", input_text)
    
    conversations = [
        {"role": "user", "content": system_prompt + input_text},
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
    print("==> Response:")
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            llm_answer += chunk.choices[0].delta.content
    print("\n")

    data = parse_llm(llm_answer)
    temperature = get_weather(data[0]["arguments"]["latitude"], data[0]["arguments"]["longitude"])
    print(f"The current temperature is {temperature} Celsius")
    print("-"*100)
    
if __name__ == '__main__':
    for input_text in ["What is the weather in Taipei?"]:
        run_conversation(input_text)
