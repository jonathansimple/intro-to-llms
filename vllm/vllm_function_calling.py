from openai import OpenAI
import json
import requests

client = OpenAI(
    base_url="http://210.61.209.139:45276/v1",   # your vLLM server
    api_key="token-abc123",                # same token you started vLLM with
)

system_prompt = r"""You are a helpful assistant with access to the following functions. Come up with the latitude and longitude information yourself. Use them if required -
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
"""

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m", verify=False)
    data = response.json()
    return data['current']['temperature_2m']

def run_conversation(query):
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",  # replace with your served model name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=8192,
        stream=True,
    )

    llm_answer = ""
    print("==> Query:", query)
    print("==> Response:")
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            llm_answer += chunk.choices[0].delta.content
    print("\n")

    # If function/tool is used, the LLM output will be a json we can parse
    try:
        # A more complicated project would use a function/tool map/list to guide control flow
        data = json.loads(llm_answer)
        temperature = get_weather(data["arguments"]["latitude"], data["arguments"]["longitude"])
        print(f"The current temperature is {temperature} Celsius")
    except:
        # If the LLM responded in plain text, no additional processing is needed
        pass



if __name__ == '__main__':
    # Function calling flow demo
    # Case 1: function call is triggered
    # Case 2: no relevant tool is present so plain text response is received
    for input_text in ["What is the weather in Taipei?", "Who is the president of the United States?"]:
        run_conversation(input_text)
        print("-"*100)

    while True:
        # Wait for user input
        query_str = input("How may I help you?\n")
        
        # Skip if query is empty
        if len(query_str) == 0:
            continue
    
        # Inference by sending an API request to either the Ollama or LM Studio server
        llm_answer = run_conversation(query_str)
        
        print("-"*100)
