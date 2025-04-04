# FastAPI
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/resource")
def read_root():
    return {"message": "Ok!"}

if __name__ == "__main__":
    uvicorn.run("FastAPI:app", host="127.0.0.1", port=8000)