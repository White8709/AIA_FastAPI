import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List , Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

app = FastAPI()

DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class Fruit(Base):
    __tablename__ = "fruit"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, default=None)
    price = Column(Float)
    on_offer = Column(Boolean, default=False)

class FruitCreate(BaseModel):
    id: int = None
    name: str
    description: Optional[str] = None
    price: float
    on_offer: bool = False

async def get_db():
    async with SessionLocal() as session:
        yield session

# 创建水果
@app.post("/fruit", response_model=FruitCreate, tags=["Fruit"])
async def create_Fruit(fruit: FruitCreate, db: AsyncSession = Depends(get_db)):
    # 检查水果是否存在
    result = await db.execute(select(Fruit).filter(Fruit.name == fruit.name))
    existing_fruit = result.scalars().first()
    if existing_fruit:
        raise HTTPException(status_code=400, detail="Fruit already exists")
    
    # 添加新水果
    db_fruit = Fruit(name=fruit.name, description=fruit.description, price=fruit.price, on_offer=fruit.on_offer)
    db.add(db_fruit)
    await db.commit()
    await db.refresh(db_fruit)
    return db_fruit

# 查询所有水果
@app.get("/fruit", response_model=List[FruitCreate], tags=["Fruit"])
async def query_Fruits(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Fruit))
    fruits = result.scalars().all()
    return fruits

# 查询单个水果
@app.get("/fruit/{fruit_id}", response_model=FruitCreate, tags=["Fruit"])
async def query_Fruit(fruit_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Fruit).filter(Fruit.id == fruit_id))
    fruit = result.scalars().first()
    if not fruit:
        raise HTTPException(status_code=404, detail="Fruit not found")
    return fruit

# 更新水果
@app.put("/fruit/{fruit_id}", response_model=FruitCreate, tags=["Fruit"])
async def update_Fruit(fruit_id: int, fruit: FruitCreate, db: AsyncSession = Depends(get_db)):
    db_fruit = await db.execute(select(Fruit).filter(Fruit.id == fruit_id))
    db_fruit = db_fruit.scalars().first()
    if not db_fruit:
        raise HTTPException(status_code=404, detail="Fruit not found")
    
    db_fruit.name = fruit.name
    db_fruit.description = fruit.description
    db_fruit.price = fruit.price
    db_fruit.on_offer = fruit.on_offer

    db.add(db_fruit)
    await db.commit()
    await db.refresh(db_fruit)
    return db_fruit

# 删除水果
@app.delete("/fruit/{fruit_id}", tags=["Fruit"])
async def delete_Fruit(fruit_id: int, db: AsyncSession = Depends(get_db)):
    db_fruit = await db.execute(select(Fruit).filter(Fruit.id == fruit_id))
    db_fruit = db_fruit.scalars().first()
    if not db_fruit:
        raise HTTPException(status_code=404, detail="Fruit not found")
    
    await db.delete(db_fruit)
    await db.commit()
    return {"message": "Fruit deleted successfully"}

if __name__ == "__main__":
    uvicorn.run("FastAPI_Restful:app", host="127.0.0.1", port=8000, reload=True)
