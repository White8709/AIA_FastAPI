# FastAPI
from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="我的 API",
    description="提供資源",
    version="1.0.0",
    contact={
        "name": "Daniel Lee",
        "email": "daniel@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

tags_metadata = [
    {
        "name": "Resourse",
        "description": "資源",
    }
]

@app.get(
    "/resource",
    tags=["Resource"],
    summary="取得資源狀態",
    description="確認伺服器資源的狀態是否正常。",
    response_description="成功時回傳確認訊息" 
)
def read_root():
    return {"message": "Ok!"}