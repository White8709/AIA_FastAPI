from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation = [{"role": "system", "content": "You are a helpful assistant."}]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

async def generate_response(user_input: str) -> str:
    conversation.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = output_ids[0][model_inputs.input_ids.shape[-1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    conversation.append({"role": "assistant", "content": response_text})
    return response_text

@app.post("/api/v0/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if len(conversation) > 3:
        conversation.pop(1)
        conversation.pop(1)

    response = await generate_response(request.message)
    print(conversation)
    return ChatResponse(response=response)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
