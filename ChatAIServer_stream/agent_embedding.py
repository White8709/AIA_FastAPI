import os
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

model_name = "BAAI/bge-small-zh"  # 或 bge-large-zh
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0]
    return embeddings

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), r"D:\python\FastAPI\ChatAIServer_stream\tools.csv"))
    tqdm.pandas()
    df["bge_embedding"] = df["工具"].astype(str).apply(lambda x: embed(x))
    print(df)
    # 儲存成 pickle
    with open("tools.pkl", "wb") as f:
        pickle.dump(df, f)