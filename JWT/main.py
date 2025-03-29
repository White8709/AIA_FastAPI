import uvicorn  # 匯入 uvicorn 模組
from datetime import datetime, timedelta, timezone  # 從 datetime 模組匯入 datetime、timedelta 和 timezone
from typing import Annotated  # 從 typing 模組匯入 Annotated

import jwt  # 匯入 jwt 模組
from fastapi import Depends, FastAPI, HTTPException, status  # 從 fastapi 模組匯入 Depends、FastAPI、HTTPException 和 status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm  # 從 fastapi.security 模組匯入 OAuth2PasswordBearer 和 OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError  # 從 jwt.exceptions 模組匯入 InvalidTokenError
from passlib.context import CryptContext  # 從 passlib.context 模組匯入 CryptContext
from pydantic import BaseModel  # 從 pydantic 模組匯入 BaseModel

# 要獲取這樣的字符串，請運行：
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # 設定密鑰
ALGORITHM = "HS256"  # 設定加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 設定訪問令牌過期時間（分鐘）

fake_users_db = {  # 假用戶資料庫
    "johndoe": {
        "username": "johndoe",  # 用戶名
        "full_name": "John Doe",  # 全名
        "email": "johndoe@example.com",  # 電子郵件
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # 哈希密碼
        "disabled": False,  # 是否禁用
    }
}

class Token(BaseModel):  # 定義 Token 類
    access_token: str  # 訪問令牌
    token_type: str  # 令牌類型

class TokenData(BaseModel):  # 定義 TokenData 類
    username: str | None = None  # 用戶名

class User(BaseModel):  # 定義 User 類
    username: str  # 用戶名
    email: str | None = None  # 電子郵件
    full_name: str | None = None  # 全名
    disabled: bool | None = None  # 是否禁用

class UserInDB(User):  # 定義 UserInDB 類，繼承自 User
    hashed_password: str  # 哈希密碼

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  # 設定密碼加密上下文

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # 設定 OAuth2 密碼承載者

app = FastAPI()  # 創建 FastAPI 應用

def verify_password(plain_password, hashed_password):  # 驗證密碼
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):  # 獲取密碼哈希值
    return pwd_context.hash(password)

def get_user(db, username: str):  # 獲取用戶
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):  # 驗證用戶
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):  # 創建訪問令牌
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):  # 獲取當前用戶
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):  # 獲取當前活躍用戶
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token")  # 定義 /token 路由
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:  # 登錄以獲取訪問令牌
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me/", response_model=User)  # 定義 /users/me/ 路由
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):  # 讀取當前用戶信息
    return current_user

@app.get("/users/me/items/")  # 定義 /users/me/items/ 路由
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)],
):  # 讀取當前用戶的項目
    return [{"item_id": "Foo", "owner": current_user.username}]

if __name__ == "__main__":  # 主程序入口
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)  # 運行 uvicorn 服務器
