from transformers import AutoModelForCausalLM , AutoTokenizer
import time
import os
import json
import requests
import torch

from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained("./gemma-ft", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("./gemma-ft", trust_remote_code=True, device_map="auto")

system_prompt = r"""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_weather",
    "description": "Get the current temperature for the provided coordinates in Celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
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

max_new_tokens = 1024
generation_config = dict(
    #temperature=0.0,
    #top_k=30,
    #top_p=0.6,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)

def run_conversation(input_text):
    print("==> User Input:", input_text)
    
    history_messages = [
        {"role": "user", "content": system_prompt + input_text},
    ]
    
    inputs = tokenizer.apply_chat_template(
            history_messages,
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt"
    )

    # For CUDA or ROCm enabled devices
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    start_time = time.time()
    function_calling_output = model.generate(
            input_ids = inputs, 
            **generation_config
    )
    end_time = time.time()

    generated_tokens = function_calling_output.shape[-1] - inputs.shape[-1]
    elapsed_time = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed_time

    function_calling_output = tokenizer.decode(function_calling_output[0],skip_special_tokens=True)

    print("==> Gemma output:", function_calling_output)

    data = parse_llm(function_calling_output)
    temperature = get_weather(data[0]["arguments"]["latitude"], data[0]["arguments"]["longitude"])
    print(f"The current temperature is {temperature} Celsius")
    print(f"Generated {generated_tokens} tokens in {elapsed_time:.2f}s → {tokens_per_sec:.2f} tokens/sec")
    print("-"*100)
    
if __name__ == '__main__':
    for input_text in ["What is the weather in Taipei?"]:
        run_conversation(input_text)
