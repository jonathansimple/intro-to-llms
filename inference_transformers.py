from transformers import AutoModelForCausalLM , AutoTokenizer
import os
import torch

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True, device_map="auto")

model.eval()

generation_config = dict(
    #temperature=0.9,
    #top_k=30,
    #top_p=0.6,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=256
)

if __name__ == '__main__':

    for user_input in ["Who is the president of the United States?"]:    
        print("==> User input:", user_input)
        history_messages = [
            {"role": "user", "content": user_input},
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

        print("generating...")
        output = model.generate(
            input_ids = inputs, 
            **generation_config
        )[0]

        output = tokenizer.decode(output, skip_special_tokens=True)

        print("==> Gemma output:", output[output.find("model\n")+6:])



