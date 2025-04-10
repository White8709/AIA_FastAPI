from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import asyncio
from threading import Thread

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda()
model.eval()

conversation = [{"role": "system", "content": "你是有用的智慧助理，都用繁體中文回答"}]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    if len(conversation) > 10:
        conversation.pop(1)
        conversation.pop(1)
    conversation.append({"role": "user", "content": request.message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=2048, do_sample=True, temperature=0.7)

    def generate():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        response_text = ""
        for new_text in streamer:
            response_text += new_text
            yield new_text
        conversation.append({"role": "assistant", "content": response_text})
    print(conversation)
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)