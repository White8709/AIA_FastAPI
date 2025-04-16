from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, AutoModel
import torch
from threading import Thread
from pathlib import Path
from scipy.spatial.distance import cosine
import pickle
import numpy as np

app = FastAPI()

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

#conversation = [{"role": "system", "content": "你是有用的智慧助理，都用繁體中文回答"}]

class ChatRequest(BaseModel):
    message: str

with open("D:\python\FastAPI\ChatAIServer\embedded.pkl", "rb") as f:
    df = pickle.load(f)
    #print(df)

model_name_embed = "BAAI/bge-small-zh"  # 或 bge-large-zh
tokenizer_embed = AutoTokenizer.from_pretrained(model_name_embed)
model_embed = AutoModel.from_pretrained(model_name_embed)
def embed(text):
    inputs = tokenizer_embed(text, padding=True, truncation=True, return_tensors="pt").to(model_embed.device)
    with torch.no_grad():
        embedding = model_embed(**inputs).last_hidden_state[:, 0]  # CLS token
    return embedding[0].detach().cpu().numpy().flatten()  # 強制變 1D

def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(np.array(embedding1).flatten(), np.array(embedding2).flatten())

def search_docs(df, user_query, similarity_threshold=0.85, top_n=1, to_print=True):
    embedding = embed(
        user_query
    )
    df["similarities"] = df.bge_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df[df['similarities'] >= similarity_threshold].sort_values("similarities", ascending=False).head(top_n)
    )
    if to_print:
        print(res)
    return res

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = Path("D:/python/FastAPI/ChatAIServer_stream/templates/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/chat")
async def chat(request: ChatRequest):
    '''
    if len(conversation) > 10:
        conversation.pop(1)
        conversation.pop(1)
    '''
    res = search_docs(df, request.message, similarity_threshold=0.90, top_n=1, to_print=True)
    if not res.empty:
        conversation = [{"role": "system", "content": "你是有用的智慧助理，都用繁體中文回答"}]
        conversation.append({"role": "user", "content": f"根據網址{res['結果'][0]}回答問題:{request.message}，簡短回答user："})
        print(conversation)
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
            #conversation.append({"role": "assistant", "content": response_text})
            #print(conversation)
        return StreamingResponse(generate(), media_type="text/plain")
    else:
        print("機器人: 抱歉，這個問題我無法提供建議。")
'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
    '''