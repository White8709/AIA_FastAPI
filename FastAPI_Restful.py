# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 定义数据模型
class Item(BaseModel):
    id: int
    name: str
    description: str = None
    price: float
    on_offer: bool = False

# 模拟数据库
fake_db = {
    1: Item(id=1, name="Item 1", description="Description for Item 1", price=10.0, on_offer=True),
    2: Item(id=2, name="Item 2", description="Description for Item 2", price=20.0, on_offer=False),
    3: Item(id=3, name="Item 3", description="Description for Item 3", price=30.0, on_offer=True),
}

# 创建资源
@app.post("/resource", response_model=Item)
def create_item(item: Item):
    if item.id in fake_db:
        raise HTTPException(status_code=400, detail="Item already exists")
    fake_db[item.id] = item
    return item

# 读取所有资源
@app.get("/resource", response_model=List[Item])
def read_items():
    return list(fake_db.values())

# 读取单个资源
@app.get("/resource/{item_id}", response_model=Item)
def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]

# 更新资源
@app.put("/resource/{item_id}", response_model=Item)
def update_item(item_id: int, item: Item):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    fake_db[item_id] = item
    return item

# 删除资源
@app.delete("/resource/{item_id}")
def delete_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del fake_db[item_id]
    return {"message": "Item deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("FastAPI:app", host="127.0.0.1", port=8000)