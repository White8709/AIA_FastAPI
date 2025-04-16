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

'''
def search_docs(df, user_query, similarity_threshold=0.85, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" 
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df[df['similarities'] >= similarity_threshold].sort_values("similarities", ascending=False).head(top_n)
    )
    if to_print:
        print(res)
    return res
'''
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "D:\python\FastAPI\ChatAIServer_stream\data.csv"))
    tqdm.pandas()
    df["bge_embedding"] = df["語句"].astype(str).apply(lambda x: embed(x))
    print(df)
    # 儲存成 pickle
    with open("embedded.pkl", "wb") as f:
        pickle.dump(df, f)