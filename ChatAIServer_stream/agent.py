import requests
from bs4 import BeautifulSoup
import re
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

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

with open(r"D:\python\FastAPI\ChatAIServer_stream\tools.pkl", "rb") as f:
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

async def Weather():
    url = "https://tw.news.yahoo.com/weather/?location-picker=%E6%9D%BF%E6%A9%8B"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    temp_span = soup.find("span", class_=re.compile(r"celsius.*celsius_D\(b\)"))

    if temp_span:
        return("目前氣溫：", temp_span.text.strip())
    else:
        return("找不到氣溫")
    
async def Stock():
    url = "https://tw.stock.yahoo.com/quote/2330.TW"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    price_span = soup.find("span", class_=re.compile(r"Fz\(32px\).*Fw\(b\).*"))
    if price_span:
        return("股價：", price_span.text.strip())
    else:
        return("找不到股價")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = Path("D:/python/FastAPI/ChatAIServer_stream/templates/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/chat")
async def chat(request: ChatRequest):
    res = search_docs(df, request.message, similarity_threshold=0.90, top_n=1, to_print=True)
    if not res.empty:
        conversation = [{"role": "system", "content": "你是有用的智慧助理，都用繁體中文回答"}]
        res_unique = res.drop_duplicates(subset='方法')
        res_f = res_unique["方法"].iloc[0]
        if res_f == "Weather":
            resault = await Weather()
            print(resault)
            conversation.append({"role": "user", "content": f"根據以下資訊回答問題：{resault} 問題：{request.message}回答："})
        elif res_f == "Stock":
            resault = await Stock()
            print(resault)
            conversation.append({"role": "user", "content": f"根據以下資訊回答問題：{resault} 問題：{request.message}回答："})
        else:
            conversation.append({"role": "user", "content": request.message})
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
        return StreamingResponse(generate(), media_type="text/plain")
    else:
        print("機器人: 抱歉，這個問題我無法提供建議。")
