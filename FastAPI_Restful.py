import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Fruit(BaseModel):
    id: int
    name: str
    description: str = None
    price: float
    on_offer: bool = False

fake_db = {
    1: Fruit(id=1, name="香蕉", description="這是香蕉", price=41.9, on_offer=True),
    2: Fruit(id=2, name="蘋果", description="這是蘋果", price=36.0, on_offer=False),
    3: Fruit(id=3, name="芭樂", description="這是芭樂", price=39.7, on_offer=True),
}

@app.post("/fruit", response_model=Fruit, tags=["Fruit"])
def create_Fruit(fruit: Fruit):
    if any(existing_fruit.name == fruit.name for existing_fruit in fake_db.values()):
        raise HTTPException(status_code=400, detail="fruit already exists")
    fake_db[fruit.id] = fruit
    return fruit

@app.get("/fruit", response_model=List[Fruit], tags=["Fruit"])
def query_Fruits():
    return list(fake_db.values())

@app.get("/fruit/{fruit_id}", response_model=Fruit, tags=["Fruit"])
def query_Fruit(fruit_id: int):
    if fruit_id not in fake_db:
        raise HTTPException(status_code=404, detail="Fruit not found")

    return fake_db[fruit_id]

@app.put("/fruit/{fruit_id}", response_model=Fruit, tags=["Fruit"])
def update_Fruit(fruit_id: int, fruit: Fruit):
    if fruit_id not in fake_db:
        raise HTTPException(status_code=404, detail="Fruit not found")
    fake_db[fruit_id] = fruit
    return fruit

@app.delete("/fruit/{fruit_id}", tags=["Fruit"])
def delete_Fruit(fruit_id: int):
    if fruit_id not in fake_db:
        raise HTTPException(status_code=404, detail="Fruit not found")
    del fake_db[fruit_id]
    return {"message": "Fruit deleted successfully"}

if __name__ == "__main__":
    uvicorn.run("FastAPI_Restful:app", host="127.0.0.1", port=8000, reload=True)