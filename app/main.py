from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from .db import Base, engine
from .auth import router as auth_router 
from .settings import settings


app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    
api_router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origins] if settings.allowed_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router.include_router(auth_router, prefix="/auth")

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"Hello": "World"}
