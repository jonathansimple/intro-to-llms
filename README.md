# Introduction
This repo serves as an introductory resource and guide for those interested in deploying LLM solutions on AMD platforms. It covers a high level discussion on the foundation of LLMs and includes practical implementations and slides users can follow along. Source code can also be run on Nvidia platforms without modification.

# Target Audience
- Programmers familiar with Python
- Computer Science or Electrical Engineering students interested in building LLM solutions on ROCm or AMD AI PCs
- Anyone looking to build their first LLM project

# Included Materials
- Setting up your AMD environment (ROCm or AI PC/NPU) [**PDF**]
- Deploying a LLM server using Ollama and LM Studio [**PDF**]
- LLM inferencing using transformers [```inference_transformers.py```]
- LLM inferencing using API (Ollama and LM Studio) [```inference_ollama.py```]
- Fine tuning your model with custom dataset(s) to perform function calling [```gemma_function_calling_finetune.py```]
- Function calling using Open-Meteo as an example [```function_calling_transformers.py```, ```function_calling_ollama.py```]
- End-to-end RAG example using LM Studio to host query generator [```rag_query_ollama.py```] 
- Lecture slides with high level discussions on LLM fundamentals [**PDF**]

# Usage
Set up your conda environment using the requirements.txt most suited for your machine. Use requirements_nvidia_compatible.txt for Nvidia devices and AMD machines with ROCm compatibility. Use requirements_ai_pc.txt for AMD AI PCs with NPU + iGPU. (For AMD AI PCs, I STRONGLY RECOMMEND following AMD's [NPU driver installation tutorial](https://ryzenai.docs.amd.com/en/latest/inst.html) instead of using this requirements.txt. You can manually install the missing libraries from the AMD installer afterwards.)

After providing your Huggingface access tokens, you can run the transformers-based code without additional arguments.

```console
python inference_transformers.py
python gemma_function_calling_finetune.py
python function_calling_transformers.py
```

After installing Ollama or LM Studio and starting a LLM server with your chosen model (refer to the lecture slides for detailed walkthrough), you can run the API based code without additional arguments.

```console
python inference_ollama.py
python rag_query_ollama.py
python function_calling_ollama.py
```
